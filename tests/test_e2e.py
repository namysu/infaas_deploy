"""End-to-end without a GPU: real controller + worker gRPC services, a fake runtime.

Covers: Front-End -> Dispatcher -> worker on-demand load -> Active path,
the Lumina-compatible API, Placement (downgrade across GPU types), the
monitoring window's state update, and the VM-Autoscaler's discovery/recovery.
"""
import threading
import time
from concurrent import futures

import fakeredis
import grpc
import pytest

from infaas.common.naming import parse_variant
from infaas.controller.dispatcher import Dispatcher
from infaas.controller.frontend import ExecutorCompat, Frontend
from infaas.controller.placement import Placement
from infaas.controller.state import Executors, Registry, WorkerClients
from infaas.controller.vm_autoscaler import VMAutoscaler
from infaas.controller.k8s_adapter import PodInfo
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.policy import states
from infaas.policy.scaling import ScaleOption
from infaas.proto import podexec_pb2, podexec_pb2_grpc
from infaas.proto.internal import placement_pb2, query_pb2_grpc, sys_monitor_pb2_grpc
from infaas.worker.context import WorkerContext
from infaas.worker.executor import QueryService, SysStatusService
from infaas.worker.monitor import Monitor

GB = 1 << 30


class FakeInstance:
    dali_cfg = None
    dali_note = "fake"
    inflight = 0

    def __init__(self, variant, delay):
        self.variant, self.delay = variant, delay
        self.loaded_at = time.time()

    def infer(self, image):
        time.sleep(self.delay)
        ms = self.delay * 1000
        return "goldfish", [], (224, 224), {"decode": 0.1, "preprocess": 0.1, "pre_wait": 0,
                                            "lock_wait": 0, "forward": ms, "postprocess": 0.1,
                                            "total": ms, "dali": 0}


class FakeRuntime:
    def __init__(self, hw, load_s=0.05, delay=0.005):
        self.hw, self.load_s, self.delay = hw, load_s, delay
        self._i = {}
        self._lock = threading.Lock()

    def get(self, v):
        return self._i.get(v)

    def loaded(self):
        return sorted(self._i)

    def is_loading(self, v):
        return False

    def load(self, v):
        assert parse_variant(v)[1] == self.hw
        with self._lock:
            if v in self._i:
                return self._i[v], 0.0, False
            time.sleep(self.load_s)
            self._i[v] = FakeInstance(v, self.delay)
            return self._i[v], self.load_s * 1000, True

    def unload(self, v, drain_timeout_s=5.0):
        return self._i.pop(v, None) is not None


def _profile(model, hw, lat, load, sat):
    return {"variant": f"{model}__{hw}", "model": model, "hardware": hw, "inf_latency_ms": lat,
            "load_latency_ms": load, "sat_qps": sat, "peak_memory_bytes": GB // 2}


def _start(servicers):
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    for add, obj in servicers:
        add(obj, s)
    port = s.add_insecure_port("localhost:0")
    s.start()
    return s, f"localhost:{port}"


@pytest.fixture
def cluster():
    server = fakeredis.FakeServer()
    new_md = lambda: RedisMetadata(fakeredis.FakeRedis(server=server, decode_responses=True))  # noqa: E731
    md = new_md()
    for hw, lat, load, sat in (("2080ti", 40, 1500, 30), ("a5000", 30, 1500, 40),
                               ("a30", 25, 1200, 50)):
        md.add_model(_profile("resnet-50", hw, lat, load, sat))
    workers, servers = {}, []
    for name, hw in (("t0", "2080ti"), ("x0", "a30")):
        ctx = WorkerContext(name, hw, new_md(), FakeRuntime(hw))
        s, addr = _start([(query_pb2_grpc.add_QueryServicer_to_server, QueryService(ctx)),
                          (sys_monitor_pb2_grpc.add_SysStatusServicer_to_server,
                           SysStatusService(ctx, None))])
        servers.append(s)
        md.add_executor(name, addr, hw)
        md.update_worker_stats(name, 5.0, 5.0, 20 * GB, 24 * GB)
        workers[name] = (ctx, addr)
    registry, clients = Registry(md), WorkerClients()
    fe = Frontend(Dispatcher(md, registry, Executors(md), mode=6), registry, clients)
    yield md, fe, workers, registry, clients
    for s in servers:
        s.stop(0)


def test_cold_then_active(cluster):
    md, fe, workers, _, _ = cluster
    r1 = fe.serve("resnet-50", b"jpeg", 100.0)
    assert r1.ok and r1.path == "inactive" and r1.hardware == "a30" and r1.worker == "x0"
    assert r1.timings["worker.load"] > 0 and r1.timings["worker.swap"] > 0
    assert md.get_instance_states("resnet-50__a30") == {"x0": states.ACTIVE}
    r2 = fe.serve("resnet-50", b"jpeg", 100.0)
    assert r2.ok and r2.path == "active" and r2.timings["worker.load"] == 0


def test_reject_kinds(cluster):
    _, fe, _, _, _ = cluster
    r = fe.serve("resnet-50", b"jpeg", 10.0)
    assert not r.ok and r.reject_kind == "no_variant" and r.suggestion == "resnet-50__a30"
    r = fe.serve("unknown-model", b"jpeg", 100.0)
    assert not r.ok and r.reject_kind == "bad_request"


def test_lumina_compat_api(cluster):
    _, fe, _, _, _ = cluster
    s, addr = _start([(podexec_pb2_grpc.add_ExecutorServicer_to_server, ExecutorCompat(fe))])
    try:
        stub = podexec_pb2_grpc.ExecutorStub(grpc.insecure_channel(addr))
        rep = stub.Infer(podexec_pb2.InferRequest(model="resnet-50", image=b"jpeg", slo=100.0))
        assert not rep.rejected and rep.gpu == "a30" and rep.inference == "goldfish"
        assert dict(rep.timings)["worker.swap"] > 1.0          # trace_bench's swap%
        assert set(rep.estimates) == {"2080ti", "a5000", "a30"}
        rep = stub.Infer(podexec_pb2.InferRequest(model="resnet-50", image=b"jpeg", slo=5.0))
        assert rep.rejected and rep.reject_kind == "no_variant" and rep.gpu == ""
        rep = stub.Infer(podexec_pb2.InferRequest(model="resnet-50", image=b"x", slo=1,
                                                  dispatch="predict"))
        assert rep.rejected
    finally:
        s.stop(0)


def test_placement_downgrade_moves_instance(cluster):
    md, fe, workers, registry, clients = cluster
    fe.serve("resnet-50", b"jpeg", 100.0)                  # a30 on x0
    pl = Placement(md, registry, clients)
    md.set_pending("resnet-50__a30")
    pl._execute(placement_pb2.ScaleActionRequest(src_worker="x0", reason="test", options=[
        placement_pb2.ScaleOption(kind="downgrade", src_variant="resnet-50__a30",
                                  dst_variant="resnet-50__2080ti", count=1, cost=-1.0)]))
    assert md.get_instance_states("resnet-50__2080ti") == {"t0": states.ACTIVE}
    assert md.get_instance_states("resnet-50__a30") == {}
    assert not md.is_pending("resnet-50__a30")
    r = fe.serve("resnet-50", b"jpeg", 100.0)
    assert r.path == "active" and r.hardware == "2080ti"


def test_placement_infeasible_raises_vm_flag(cluster):
    md, fe, workers, registry, clients = cluster
    fe.serve("resnet-50", b"jpeg", 100.0)                  # a30 on x0 (the only a30)
    pl = Placement(md, registry, clients)
    pl._execute(placement_pb2.ScaleActionRequest(src_worker="x0", reason="test", options=[
        placement_pb2.ScaleOption(kind="replicate", src_variant="resnet-50__a30",
                                  dst_variant="resnet-50__a30", count=1, cost=1.0)]))
    assert md.vm_scale_flags() == ["a30"]


def test_monitor_window_marks_overloaded(cluster):
    md, fe, workers, _, _ = cluster
    fe.serve("resnet-50", b"jpeg", 100.0)
    ctx, _ = workers["x0"]
    ctx.runtime.get("resnet-50__a30").loaded_at -= 10    # up for the whole window
    mon = Monitor(ctx)
    for _ in range(120):                                    # 60 qps over a 2 s window > sat 50
        ctx.counters_for("resnet-50__a30").begin(100.0)
        ctx.counters_for("resnet-50__a30").end(25.0)
    mon.window(2.0)
    assert md.get_instance_states("resnet-50__a30") == {"x0": states.OVERLOADED}
    # the serve() above is in this window too: (120 + 1) / 2 s
    assert md.get_model_qps("x0", "resnet-50__a30") == pytest.approx(60.5)
    assert mon.min_slo("resnet-50__a30") == 100.0


class FakeK8s:
    def __init__(self, pods):
        self.pods, self.created, self.deleted = pods, [], []

    def list_workers(self):
        return dict(self.pods)

    def create_worker(self, hw):
        self.created.append(hw)
        return f"new-{hw}"

    def delete_pod(self, name):
        self.deleted.append(name)


def test_vm_autoscaler_discovery_failure_and_recovery(cluster):
    md, fe, workers, _, _ = cluster
    fe.serve("resnet-50", b"jpeg", 100.0)                  # a30 on x0
    _, x0_addr = workers["x0"]
    _, t0_addr = workers["t0"]
    host, _, port = x0_addr.rpartition(":")
    pods = {"t0": PodInfo("t0", "2080ti", "localhost", True, "Running", False, False)}
    k = FakeK8s(pods)
    vm = VMAutoscaler(md, k, mode="static")
    vm.sync(k.list_workers())                               # x0 vanished -> failure
    assert "x0" not in md.get_all_executors()
    assert md.pop_recover("a30") == ["resnet-50__a30"]


def test_vm_autoscaler_dynamic_scale_up_on_flag(cluster):
    md, fe, workers, _, _ = cluster
    pods = {n: PodInfo(n, ctx.hw, "localhost", True, "Running", True, False)
            for n, (ctx, _) in workers.items()}
    k = FakeK8s(pods)
    vm = VMAutoscaler(md, k, mode="dynamic")
    md.set_vm_scale("a30")
    vm.scale(pods)
    assert k.created == ["a30"]
    vm.scale(pods)                                           # backoff: nothing new
    assert k.created == ["a30"]


class _StubMonitor:
    def __init__(self, qps, slo):
        self.qps, self.slo = qps, slo

    def last_qps(self, v):
        return self.qps.get(v, 0.0)

    def min_slo(self, v):
        return self.slo

    def observed(self, v):
        return True


def test_autoscaler_tick_falls_through_to_upgrade(cluster):
    """2080ti overloaded, no second 2080ti: replicate is cheapest but infeasible,
    so the controller carries out the next option, upgrade to a30."""
    from infaas.proto.internal import placement_pb2_grpc
    from infaas.worker.autoscaler import ModelAutoscaler
    md, fe, workers, registry, clients = cluster
    ctx, _ = workers["t0"]
    ctx.ensure_loaded("resnet-50__2080ti")
    md.update_instance("t0", "resnet-50__2080ti", states.OVERLOADED, 40.0, 60.0)
    pl = Placement(md, registry, clients)
    s, addr = _start([(placement_pb2_grpc.add_PlacementServicer_to_server, pl)])
    try:
        ctx._placement = placement_pb2_grpc.PlacementStub(grpc.insecure_channel(addr))
        a = ModelAutoscaler(ctx, _StubMonitor({"resnet-50__2080ti": 40.0}, 100.0))
        a.tick(time.time())
        deadline = time.time() + 5
        while time.time() < deadline and md.get_instance_states("resnet-50__2080ti"):
            time.sleep(0.05)
        assert md.get_instance_states("resnet-50__a30") == {"x0": states.ACTIVE}
        assert md.get_instance_states("resnet-50__2080ti") == {}
        assert not md.is_pending("resnet-50__2080ti")
    finally:
        s.stop(0)


def test_monitor_first_window_counts_from_load(cluster):
    """A load inside the window: QPS is over the time since the load."""
    md, fe, workers, _, _ = cluster
    fe.serve("resnet-50", b"jpeg", 100.0)
    ctx, _ = workers["x0"]
    ctx.runtime.get("resnet-50__a30").loaded_at = time.time() - 0.5
    mon = Monitor(ctx)
    assert not mon.observed("resnet-50__a30")
    mon.window(2.0)
    assert md.get_model_qps("x0", "resnet-50__a30") == pytest.approx(2.0, rel=0.05)   # 1 req / 0.5 s
    assert mon.observed("resnet-50__a30")
