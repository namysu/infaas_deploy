"""One online query through the native API (infaas_online_query).

    python -m infaas.cli.online_query --controller <cp-ip>:50052 \
        --model resnet-50 --image frame_1080p.jpg --slo 200 [-n 5]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import grpc

from infaas.proto import queryfe_pb2, queryfe_pb2_grpc, request_reply_pb2 as rr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True)
    ap.add_argument("--model", required=True, help="parent model (Lumina short name)")
    ap.add_argument("--variant", default="", help="force a model-variant")
    ap.add_argument("--image", required=True)
    ap.add_argument("--slo", type=float, required=True, help="latency SLO in ms")
    ap.add_argument("-n", type=int, default=1)
    args = ap.parse_args()
    stub = queryfe_pb2_grpc.QueryStub(grpc.insecure_channel(args.controller))
    img = Path(args.image).read_bytes()
    rc = 0
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = stub.QueryOnline(queryfe_pb2.QueryOnlineRequest(
            raw_input=[img], parent_model=args.model, model_variant=args.variant,
            slo=queryfe_pb2.QuerySLO(LatencyInUSec=int(args.slo * 1000))), timeout=180)
        e2e = (time.perf_counter() - t0) * 1000.0
        ok = r.status.status == rr.SUCCESS
        rc |= not ok
        tm = dict(r.timings)
        print(f"{'OK ' if ok else 'REJ'} e2e={e2e:7.1f}ms variant={r.variant or '-':32s} "
              f"worker={r.worker or '-':32s} path={r.path:8s} label={r.label!r} "
              f"load={tm.get('worker.load', 0):.0f}ms decide={tm.get('srv.decide', 0):.2f}ms"
              + ("" if ok else f" [{r.reject_kind}] {r.status.msg} suggest={r.suggested_variant}"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
