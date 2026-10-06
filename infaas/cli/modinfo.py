"""Registered models and their variants (infaas_modarch / infaas_modinfo).

    python -m infaas.cli.modinfo --controller <cp-ip>:50052            # parents
    python -m infaas.cli.modinfo --controller <cp-ip>:50052 resnet-50  # variants
"""
from __future__ import annotations

import argparse
import sys

import grpc

from infaas.proto import queryfe_pb2, queryfe_pb2_grpc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True, help="host:port of the Front-End")
    ap.add_argument("model", nargs="?")
    args = ap.parse_args()
    stub = queryfe_pb2_grpc.QueryStub(grpc.insecure_channel(args.controller))
    if args.model:
        r = stub.QueryModelInfo(queryfe_pb2.QueryModelInfoRequest(model=args.model), timeout=10)
        names = r.reply.all_models
    else:
        r = stub.AllParentInfo(queryfe_pb2.AllParRequest(), timeout=10)
        names = r.reply.all_models
    for n in names:
        print(n)
    return 0 if names else 1


if __name__ == "__main__":
    sys.exit(main())
