"""Smoke run without GPUs: six fake workers behind the real controller services,
driven by Lumina's trace_bench (bench/run_trace.py) on the compat API.

The worker runtime is faked (fixed latencies, 0.3 s loads); everything else —
Dispatcher, bin packing, monitor windows, Model-Autoscaler, Placement, Redis
schema (fakeredis) — is the real code. Numbers are not results; the point is that
the whole loop runs and Lumina's scorer reads the replies.

    LUMINA_DIR=../new_podexecutor python tests/smoke_fake_cluster.py \
        --trace ../new_podexecutor/traces/t30.csv --seconds 6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from concurrent import futures
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import fakeredis  # noqa: E402
import grpc  # noqa: E402

import infaas.worker.monitor as monitor_mod  # noqa: E402
from infaas.cli.state import dump  # noqa: E402
from infaas.common import config  # noqa: E402
from infaas.controller.dispatcher import Dispatcher  # noqa: E402
from infaas.controller.frontend import ExecutorCompat, Frontend  # noqa: E402
from infaas.controller.placement import Placement  # noqa: E402
from infaas.controller.state import Executors, Registry, WorkerClients  # noqa: E402
from infaas.metadata.redis_metadata import RedisMetadata  # noqa: E402
from infaas.proto import podexec_pb2_grpc  # noqa: E402
from infaas.proto.internal import placement_pb2_grpc, query_pb2_grpc, sys_monitor_pb2_grpc  # noqa: E402
from infaas.worker.autoscaler import ModelAutoscaler  # noqa: E402
from infaas.worker.context import WorkerContext  # noqa: E402
from infaas.worker.executor import QueryService, SysStatusService  # noqa: E402
from infaas.worker.monitor import Monitor  # noqa: E402
from test_e2e import FakeRuntime, _start  # noqa: E402

GB = 1 << 30
monitor_mod.GpuSampler.memory = lambda self: (4 * GB, 20 * GB, 24 * GB)   # no NVML here


def main() -> int:
    lumina = Path(os.environ.get("LUMINA_DIR", HERE.parents[1] / "new_podexecutor"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(lumina / "traces/t30.csv"))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--image", default=str(lumina / "goldfish.jpg"))
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="infaas-smoke-"))
    head = tmp / "trace.csv"
    with open(args.trace) as src, open(head, "w", newline="") as dst:
        rows = csv.DictReader(src)
        w = csv.DictWriter(dst, fieldnames=rows.fieldnames)
        w.writeheader()
        for r in rows:
            if float(r["t_offset_ms"]) < args.seconds * 1000:
                w.writerow(r)
    meta = json.loads(Path(args.trace).with_suffix(".meta.json").read_text())

    server = fakeredis.FakeServer()
    new_md = lambda: RedisMetadata(fakeredis.FakeRedis(server=server, decode_responses=True))  # noqa: E731
    md = new_md()
    for m in meta["model_list"]:
        for hw, lat, sat in (("2080ti", 30, 40), ("a5000", 22, 55), ("a30", 18, 70)):
            md.add_model({"variant": f"{m}__{hw}", "model": m, "hardware": hw,
                          "inf_latency_ms": lat, "load_latency_ms": 300.0, "sat_qps": sat,
                          "peak_memory_bytes": GB // 2})
    registry, clients = Registry(md), WorkerClients()
    keep = []
    ps, paddr = _start([(placement_pb2_grpc.add_PlacementServicer_to_server,
                         Placement(md, registry, clients))])
    keep.append(ps)
    config.CONTROLLER_ADDR = paddr
    for i, hw in enumerate(["2080ti", "2080ti", "a5000", "a5000", "a30", "a30"]):
        name = f"w-{hw}-{i}"
        ctx = WorkerContext(name, hw, new_md(), FakeRuntime(hw, load_s=0.3, delay=0.012))
        mon = Monitor(ctx)
        s, addr = _start([(query_pb2_grpc.add_QueryServicer_to_server, QueryService(ctx)),
                          (sys_monitor_pb2_grpc.add_SysStatusServicer_to_server,
                           SysStatusService(ctx, mon))])
        keep.append(s)
        md.add_executor(name, addr, hw)
        md.update_worker_stats(name, 5.0, 5.0, 20 * GB, 24 * GB)
        mon.start()
        ModelAutoscaler(ctx, mon).start()

    fe = Frontend(Dispatcher(md, registry, Executors(md), mode=6), registry, clients)
    cs = grpc.server(futures.ThreadPoolExecutor(max_workers=64))
    podexec_pb2_grpc.add_ExecutorServicer_to_server(ExecutorCompat(fe), cs)
    port = cs.add_insecure_port("localhost:0")
    cs.start()
    keep.append(cs)

    rc = subprocess.call([sys.executable, str(HERE.parent / "bench/run_trace.py"),
                          "--server", f"localhost:{port}", "--label", "infaas-fake",
                          "--image", args.image, "--trace", str(head), "--warmup-s", "1",
                          "--out", str(tmp / "raw.csv"), "--summary", str(tmp / "summary.csv")],
                         env={**os.environ, "LUMINA_DIR": str(lumina)})
    dump(md)
    print(f"outputs in {tmp}", flush=True)
    os._exit(rc)


if __name__ == "__main__":
    main()
