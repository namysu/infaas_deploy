"""Dump the Metadata Store's live view: workers, instances and their states.

    python -m infaas.cli.state --redis <cp-ip>:16379 [--watch 2]
"""
from __future__ import annotations

import argparse
import sys
import time

from infaas.metadata.redis_metadata import RedisMetadata, connect


def dump(md: RedisMetadata) -> None:
    snap = md.worker_snapshot()
    print(time.strftime("%H:%M:%S"), f"workers={len(snap.workers)} flags={md.vm_scale_flags()}")
    for w in sorted(snap.workers.values(), key=lambda w: (w.hw, w.name)):
        insts = [i for insts in snap.instances.values() for i in insts if i.worker == w.name]
        print(f"  {w.name:40s} {w.hw:7s} util={w.util:5.1f}% "
              f"free={w.mem_free / 2**30:5.1f}/{w.mem_total / 2**30:4.1f}GiB"
              f"{' BLACKLISTED' if w.blacklisted else ''}")
        for i in sorted(insts, key=lambda i: i.variant):
            print(f"      {i.variant:40s} {i.state:10s} qps={i.qps:6.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default="localhost:16379")
    ap.add_argument("--watch", type=float, default=0.0)
    args = ap.parse_args()
    host, _, port = args.redis.rpartition(":")
    md = RedisMetadata(connect(host, int(port)))
    while True:
        dump(md)
        if not args.watch:
            return 0
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    sys.exit(main())
