"""Register models (infaas_modelregistration, [C cli-tools/infaas_modelregistration.cc]).

Profiles every (model, GPU type) variant on a worker of that type unless a stored
profile exists, or imports stored profiles from a directory.

    # profile all 43 Lumina models on every GPU type (workers must be up, no traffic).
    # PROFILE_MODE=original (default) needs no image; service needs --image
    python -m infaas.cli.register --controller <cp-ip>:50053 --all
    python -m infaas.cli.register --controller <cp-ip>:50053 --image frame_1080p.jpg --all
    # a few models, again even if profiled before
    python -m infaas.cli.register --controller <cp-ip>:50053 --image frame_1080p.jpg \
        --models resnet-50,mit-b1 --reprofile
    # register saved profile JSON files (no GPU work)
    python -m infaas.cli.register --controller <cp-ip>:50053 --import-dir profiles/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import grpc

from infaas.common import profiles
from infaas.proto import modelreg_pb2, modelreg_pb2_grpc, request_reply_pb2 as rr
from infaas.vendor.lumina import models


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", required=True, help="host:port of the Model Registrar")
    ap.add_argument("--image", help="JPEG to profile with (PROFILE_MODE=service only; "
                                    "use the experiment image)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="all 43 Lumina models")
    g.add_argument("--models", help="comma-separated short names")
    g.add_argument("--import-dir", help="register every <variant>.json in this directory")
    ap.add_argument("--hardware", default="", help="comma-separated GPU types (default: all)")
    ap.add_argument("--reprofile", action="store_true")
    args = ap.parse_args()

    stub = modelreg_pb2_grpc.ModelRegStub(grpc.insecure_channel(args.controller))
    if args.import_dir:
        n = 0
        for p in profiles.load_all(args.import_dir):
            r = stub.RegisterModel(modelreg_pb2.ModelRegRequest(
                submitter="tester", profile_json=json.dumps(p)), timeout=30)
            print(f"{p['variant']:45s} {rr.RequestReplyEnum.Name(r.status.status)} {r.status.msg}")
            n += r.status.status == rr.SUCCESS
        print(f"registered {n} profiles")
        return 0

    image = Path(args.image).read_bytes() if args.image else b""
    names = sorted(models.SHORT_TO_ID) if args.all else [m.strip() for m in args.models.split(",")]
    hws = [h.strip() for h in args.hardware.split(",") if h.strip()]
    failed = 0
    for m in names:
        t0 = time.time()
        r = stub.RegisterModel(modelreg_pb2.ModelRegRequest(
            parent_model=m, url=models.resolve(m) or "", profile_image=image,
            hardware=hws, reprofile=args.reprofile), timeout=3600)
        ok = r.status.status == rr.SUCCESS
        failed += not ok
        print(f"{m:40s} {'OK ' if ok else 'ERR'} {time.time() - t0:6.1f}s "
              f"{list(r.variants)} {'' if ok else r.status.msg}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
