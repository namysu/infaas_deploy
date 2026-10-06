"""Docker deployment: compose generation from GPU settings, and static worker discovery."""
import importlib.util
import sys
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from infaas.controller import static_workers
from infaas.controller.static_workers import StaticWorkers
from infaas.proto.internal import infaas_request_status_pb2 as ist, query_pb2, query_pb2_grpc

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gen_compose", ROOT / "deploy/docker/gen_compose.py")
gc = importlib.util.module_from_spec(_spec)
sys.modules["gen_compose"] = gc     # dataclasses look the module up
_spec.loader.exec_module(gc)

SMI = [{"index": 0, "uuid": "GPU-aaa", "name": "NVIDIA A30", "mem_mib": 24576},
       {"index": 1, "uuid": "GPU-bbb", "name": "NVIDIA A30", "mem_mib": 24576},
       {"index": 2, "uuid": "GPU-ccc", "name": "NVIDIA GeForce RTX 2080 Ti", "mem_mib": 11264},
       {"index": 3, "uuid": "GPU-ddd", "name": "NVIDIA GeForce RTX 2080 Ti", "mem_mib": 11264}]
BASE = {"HW_COST": "2080ti:1,a5000:2,a30:4", "PREPROCESS_MODES": "a30:gpu_only",
        "MODEL_DIR": "/m", "DECISION_MODE": "6"}


def build(**kw):
    return gc.build({**BASE, **kw}, SMI)


def test_gpu_type_from_name():
    assert gc.gpu_type_from_name("NVIDIA A30") == "a30"
    assert gc.gpu_type_from_name("NVIDIA GeForce RTX 2080 Ti") == "2080ti"
    assert gc.gpu_type_from_name("NVIDIA RTX A5000") == "a5000"
    assert gc.gpu_type_from_name("NVIDIA L40S") == "l40s"
    assert gc.gpu_type_from_name("NVIDIA A100-SXM4-80GB") == "a100"
    assert gc.gpu_type_from_name("NVIDIA GeForce RTX 4090") == "4090"


def test_explicit_gpus_make_one_worker_each():
    compose, env, warnings, info = build(GPUS="0:a30,1:a30,2:2080ti,3:2080ti")
    svcs = compose["services"]
    workers = [n for n in svcs if "-worker-" in n]
    assert workers == ["infaas-worker-0", "infaas-worker-1", "infaas-worker-2", "infaas-worker-3"]
    w2 = svcs["infaas-worker-2"]
    dev = w2["deploy"]["resources"]["reservations"]["devices"][0]
    assert dev["device_ids"] == ["2"] and dev["capabilities"] == ["gpu"]
    assert w2["environment"]["GPU_TYPE"] == "2080ti"
    assert w2["environment"]["PREPROCESS_MODE"] == "cpu_decode_gpu"
    assert svcs["infaas-worker-0"]["environment"]["PREPROCESS_MODE"] == "gpu_only"
    assert w2["volumes"][0] == "/m:/models" and w2["volumes"][1].endswith(":/logs")
    assert env["ORCHESTRATOR"] == "static" and env["WORKER_MODE"] == "static"
    assert env["HW_TYPES"] == "2080ti,a30"                      # cost order, used types only
    assert env["MAX_WORKERS"] == "2080ti:2,a30:2"
    assert env["STATIC_WORKERS"].split(",")[2] == "infaas-worker-2=2080ti@infaas-worker-2:9000"
    assert env["DECISION_MODE"] == "6" and "GPUS" not in env and "MODEL_DIR" not in env
    assert {"infaas-redis", "infaas-controller", "infaas-vm-autoscaler"} <= set(svcs)
    assert not warnings


def test_subset_uuid_and_cpuset():
    compose, env, _, _ = build(GPUS="GPU-ccc:2080ti,0:a30", CPUSETS="2:16-23")
    svcs = compose["services"]
    assert svcs["infaas-worker-2"]["deploy"]["resources"]["reservations"]["devices"][0][
        "device_ids"] == ["GPU-ccc"]
    assert svcs["infaas-worker-2"]["cpuset"] == "16-23"
    assert "cpuset" not in svcs["infaas-worker-0"]


def test_auto_uses_nvidia_smi_names():
    _, env, _, info = build(GPUS="auto")
    assert [g.hw for g in info["gpus"]] == ["a30", "a30", "2080ti", "2080ti"]
    with pytest.raises(gc.ConfigError):
        gc.build({**BASE, "GPUS": "auto"}, None)


def test_errors():
    with pytest.raises(gc.ConfigError, match="index 7"):
        build(GPUS="7:a30")
    with pytest.raises(gc.ConfigError, match="twice"):
        build(GPUS="0:a30,0:a30")
    with pytest.raises(gc.ConfigError, match="no price"):
        build(GPUS="0:a30,2:h100")
    with pytest.raises(gc.ConfigError, match="remove from the env file"):
        build(GPUS="0:a30", STATIC_WORKERS="x")
    with pytest.raises(gc.ConfigError, match="WORKER_MODE"):
        build(GPUS="0:a30", WORKER_MODE="dynamic")


def test_warnings_type_mismatch_and_single_type():
    _, _, warnings, _ = build(GPUS="2:a30")
    assert any("check the type" in w for w in warnings)
    assert any("one GPU type" in w for w in warnings)
    _, env, _, _ = gc.build({"GPUS": "0:a30,1:a30"}, SMI)        # no HW_COST, one type
    assert env["HW_COST"] == "a30:1"


def test_host_network_ports():
    compose, env, _, _ = build(GPUS="0:a30,2:2080ti", NETWORK="host", WORKER_BASE_PORT="9100")
    w = compose["services"]["infaas-worker-2"]
    assert w["network_mode"] == "host" and w["environment"]["WORKER_PORT"] == "9102"
    assert w["environment"]["REDIS_HOST"] == "127.0.0.1"
    assert "infaas-worker-2=2080ti@127.0.0.1:9102" in env["STATIC_WORKERS"]
    assert "networks" not in compose


def test_roles_for_more_hosts():
    compose, env, _, info = build(GPUS="0:a30", NETWORK="host", ROLE="workers",
                                  CONTROLLER_HOST="10.0.0.1", ADVERTISE_HOST="10.0.0.2",
                                  WORKER_PREFIX="nodeb")
    assert set(compose["services"]) == {"nodeb-0"}
    assert compose["services"]["nodeb-0"]["environment"]["CONTROLLER_ADDR"] == "10.0.0.1:50054"
    assert info["local"] == [("nodeb-0", "a30", "10.0.0.2:9000")]
    compose, env, _, _ = build(GPUS="0:a30", EXTRA_WORKERS="nodeb-0=2080ti@10.0.0.2:9000")
    assert env["STATIC_WORKERS"].endswith("nodeb-0=2080ti@10.0.0.2:9000")
    assert env["MAX_WORKERS"] == "2080ti:1,a30:1"
    assert any(p.endswith(":50054:50054") for p in compose["services"]["infaas-controller"]["ports"])


def test_read_env(tmp_path):
    f = tmp_path / "x.env"
    f.write_text("# c\nGPUS=\"0:a30\"\n\nLAMBDA=1.0\n")
    assert gc.read_env(f) == {"GPUS": "0:a30", "LAMBDA": "1.0"}


# ------------------------------------------------------------ static discovery
def test_parse_static_workers():
    ws = static_workers.parse("w0=a30@infaas-worker-0:9000, w1=2080ti@10.0.0.2:9101")
    assert [(w.name, w.hw, w.addr) for w in ws] == [("w0", "a30", "infaas-worker-0:9000"),
                                                    ("w1", "2080ti", "10.0.0.2:9101")]
    for bad in ("w0=a30", "w0@h:1", "w0=a30@h", "w0=a30@h:x"):
        with pytest.raises(ValueError):
            static_workers.parse(bad)
    with pytest.raises(ValueError, match="duplicate"):
        static_workers.parse("w=a30@h:1,w=a30@h:2")


class _HB(query_pb2_grpc.QueryServicer):
    def Heartbeat(self, request, context):
        return query_pb2.HeartbeatResponse(status=ist.InfaasRequestStatus(status=ist.SUCCESS))


def test_static_workers_follow_heartbeat():
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    query_pb2_grpc.add_QueryServicer_to_server(_HB(), s)
    port = s.add_insecure_port("localhost:0")
    s.start()
    sw = StaticWorkers(static_workers.parse(f"up=a30@localhost:{port},down=a30@localhost:1"),
                       fail_threshold=2, timeout_s=0.5)
    pods = sw.list_workers()
    assert pods["up"].ready and pods["up"].addr == f"localhost:{port}"
    assert not pods["down"].ready and pods["down"].phase == "Unreachable"
    s.stop(0)
    assert sw.list_workers()["up"].ready          # one miss is tolerated
    assert not sw.list_workers()["up"].ready      # the second is not
    assert sw.create_worker("a30") is None


def test_host_port_offset():
    compose, _, _, _ = build(GPUS="0:a30", HOST_PORT_OFFSET="10000")
    assert compose["services"]["infaas-controller"]["ports"] == [
        "0.0.0.0:60052:50052", "0.0.0.0:60053:50053", "0.0.0.0:18081:8081"]
    assert compose["services"]["infaas-redis"]["ports"] == ["127.0.0.1:26379:16379"]
