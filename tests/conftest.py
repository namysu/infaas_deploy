import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from infaas.policy import states  # noqa: E402
from infaas.policy.types import (InstanceView, Snapshot, VariantProfile,  # noqa: E402
                                 WorkerView)

GB = 1 << 30


def prof(model, hw, lat, load=2000.0, sat=None, mem=GB // 2):
    return VariantProfile(variant=f"{model}__{hw}", model=model, hw=hw, lat_ms=lat,
                          load_ms=load, sat_qps=sat if sat is not None else 1000.0 / lat,
                          mem_bytes=mem)


def worker(name, hw, util=10.0, free=10 * GB, total=11 * GB, bl=False):
    return WorkerView(name=name, hw=hw, addr=f"{name}:9000", util=util, mem_free=free,
                      mem_total=total, blacklisted=bl)


def snap(workers, instances=()):
    s = Snapshot(workers={w.name: w for w in workers})
    for v, w, st, qps in instances:
        s.instances.setdefault(v, []).append(InstanceView(v, w, st, qps))
    return s


@pytest.fixture
def resnet():
    # 2080ti slowest/cheapest, a30 fastest/priciest
    return [prof("resnet-50", "2080ti", 40.0, load=1500, sat=30),
            prof("resnet-50", "a5000", 30.0, load=1500, sat=40),
            prof("resnet-50", "a30", 25.0, load=1200, sat=50)]


@pytest.fixture
def workers6():
    return [worker("t0", "2080ti"), worker("t1", "2080ti"),
            worker("f0", "a5000", total=24 * GB, free=23 * GB),
            worker("f1", "a5000", total=24 * GB, free=23 * GB),
            worker("x0", "a30", total=24 * GB, free=23 * GB),
            worker("x1", "a30", total=24 * GB, free=23 * GB)]


A, O, I, L = states.ACTIVE, states.OVERLOADED, states.INTERFERED, states.LOADING
