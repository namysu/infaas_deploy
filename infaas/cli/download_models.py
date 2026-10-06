"""Download model weights into the HuggingFace cache the workers read (HF_HOME).

Port of new_podexecutor/scripts/download_models.py @8eb588c, so a deployment
without the Lumina repository can fetch the same 43 models. Workers run with
HF_HUB_OFFLINE=1, so every model they serve must be in this cache first.
Idempotent: cached models are not fetched again.

    # inside the worker image (deploy/docker/infaas-docker.sh download does this)
    HF_HOME=/models python -m infaas.cli.download_models [--models resnet-50,mit-b1] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from infaas.vendor.lumina import models


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="", help="comma-separated short names (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    ids = [models.resolve(m) for m in wanted] if wanted else list(models.MODEL_AUTOCLASS)
    if None in ids:
        bad = [m for m, i in zip(wanted, ids) if i is None]
        raise SystemExit(f"unsupported models: {bad}")
    print(f"HF_HOME = {Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser()}")
    print(f"models: {len(ids)}\n")
    if args.dry_run:
        for i in ids:
            print(f"  {i}  ({models.MODEL_AUTOCLASS[i]})")
        return 0

    import transformers
    from transformers import AutoImageProcessor, AutoProcessor

    ok = fail = 0
    for n, model_id in enumerate(ids, 1):
        t0 = time.perf_counter()
        print(f"[{n:>2}/{len(ids)}] {model_id} ...", end=" ", flush=True)
        try:
            getattr(transformers, models.MODEL_AUTOCLASS[model_id]).from_pretrained(model_id)
            # the worker loads the processor the same way (runtime._get_processor)
            try:
                AutoImageProcessor.from_pretrained(model_id)
            except Exception:  # noqa: BLE001
                AutoProcessor.from_pretrained(model_id)
            print(f"OK ({time.perf_counter() - t0:.1f}s)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL ({time.perf_counter() - t0:.1f}s) {e}")
            fail += 1
    print(f"\ndone: ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
