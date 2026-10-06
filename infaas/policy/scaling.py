"""Case II — Model-Autoscaler decisions (paper §4.2.1-4.2.2).

The paper's greedy heuristic approximates its ILP as:
  (a) identify whether the constraints are in danger of being violated,
  (b) consider two strategies, replicate or upgrade/downgrade,
  (c) compute the objective for each and pick the cheapest,
  (d) coordinate with the controller for VM-level scaling if nothing fits.

With batch 1 fixed [U C3], "upgrade" means a variant of the same model with a
higher saturation throughput, i.e. a faster GPU type, and "downgrade" a cheaper
one (paper: "downgrading to a cheaper variant (optimized for a lower batch size
or running on different hardware)"). One GPU per worker and GPU_MAX_REPLICAS = 1
put every new instance on another worker, so these functions only decide; the
controller places (placement.proto).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from infaas.common import config, hardware
from infaas.policy.costs import action_cost, instance_cost, instances_needed
from infaas.policy.types import VariantProfile

REPLICATE = "replicate"
UPGRADE = "upgrade"
DOWNGRADE = "downgrade"
REMOVE = "remove"
MIGRATE = "migrate"


@dataclass
class LocalStat:
    """This worker's instance of a variant, from the monitoring daemon."""
    qps: float
    min_slo_ms: Optional[float]   # smallest SLO among its recent requests


@dataclass
class ClusterStat:
    """The variant across all workers, from the Metadata Store."""
    n_running: int
    total_qps: float


@dataclass
class ScaleOption:
    kind: str
    src_variant: str
    dst_variant: str
    count: int
    cost: float


def _replacement_count(load: float, n_running: int, src: VariantProfile,
                       dst: VariantProfile) -> int:
    """Instances of `dst` so that the other src instances plus them carry the load."""
    rest = max(0, n_running - 1) * src.sat_qps
    need = load * config.SLACK_THRESHOLD - rest
    if need <= 0 or dst.sat_qps <= 0:
        return 1
    return max(1, int(math.ceil(need / dst.sat_qps - 1e-9)))


def headroom(local: Dict[str, LocalStat], profiles: Dict[str, VariantProfile]) -> float:
    """[P §4.2.2] combined saturation throughput / combined current load."""
    load = sum(s.qps for s in local.values())
    if load <= 0:
        return float("inf")
    cap = sum(profiles[v].sat_qps for v in local)
    return cap / load


def scale_up_options(local: Dict[str, LocalStat], profiles: Dict[str, VariantProfile],
                     by_model: Dict[str, Sequence[VariantProfile]],
                     cluster: Dict[str, ClusterStat]) -> Dict[str, List[ScaleOption]]:
    """Per variant that needs more capacity, its options, cheapest first."""
    if headroom(local, profiles) >= config.SLACK_THRESHOLD:
        return {}
    out: Dict[str, List[ScaleOption]] = {}
    order = sorted(local, key=lambda v: local[v].qps / max(profiles[v].sat_qps, 1e-9),
                   reverse=True)
    for v in order:
        p, s = profiles[v], local[v]
        if s.qps * config.SLACK_THRESHOLD <= p.sat_qps:
            continue                     # this instance is not the one short of room
        cl = cluster.get(v, ClusterStat(1, s.qps))
        load = max(cl.total_qps, s.qps)
        n = instances_needed(load, p.sat_qps) - max(cl.n_running, 1)
        if n <= 0:
            # the other instances have room; the Dispatcher spills over to them
            continue
        opts = [ScaleOption(REPLICATE, v, v, n, action_cost(p, n))]
        for u in by_model.get(p.model, ()):
            if u.variant == v or u.sat_qps <= p.sat_qps:
                continue                 # "support a higher throughput"
            if s.min_slo_ms is not None and u.lat_ms > s.min_slo_ms:
                continue                 # constraint (2): T_inf <= S
            m = _replacement_count(load, cl.n_running, p, u)
            opts.append(ScaleOption(UPGRADE, v, u.variant, m,
                                    action_cost(u, m) + action_cost(p, -1)))
        opts.sort(key=lambda o: (o.cost, o.dst_variant))
        out[v] = opts
    return out


def scale_down_option(v: str, local: LocalStat, profile: VariantProfile,
                      by_model: Dict[str, Sequence[VariantProfile]],
                      cluster: ClusterStat) -> Optional[ScaleOption]:
    """The cheapest cost-reducing action for this instance, or None.

    [P §4.2.2] "checks if the incoming query load can be supported by removing
    an instance of the running variant, or downgrading to a cheaper variant".
    """
    load, n = cluster.total_qps, max(cluster.n_running, 1)
    opts: List[ScaleOption] = []
    # remove this instance
    if n >= 2 and instances_needed(load, profile.sat_qps) <= n - 1:
        opts.append(ScaleOption(REMOVE, v, "", 0, action_cost(profile, -1)))
    elif n == 1 and load <= 0:
        # [C autoscaler.cc:410-415] "If the model has 0 QPS, then scale down."
        opts.append(ScaleOption(REMOVE, v, "", 0, action_cost(profile, -1)))
    # downgrade: a cheaper variant of the same model that still meets the SLO
    if load > 0 and local.min_slo_ms is not None:
        for d in by_model.get(profile.model, ()):
            if d.variant == v or instance_cost(d) >= instance_cost(profile):
                continue
            if d.lat_ms > local.min_slo_ms:
                continue                 # constraint (2)
            m = _replacement_count(load, n, profile, d)
            cost = action_cost(d, m) + action_cost(profile, -1)
            opts.append(ScaleOption(DOWNGRADE, v, d.variant, m, cost))
    opts = [o for o in opts if o.cost < 0]
    if not opts:
        return None
    return min(opts, key=lambda o: (o.cost, o.dst_variant))


class ScaleDownTimer:
    """[P §4.2.2] "waits for Tv time slots before executing the chosen strategy
    for a variant v ... Tv is set equal to the loading latency of variant v."
    """

    def __init__(self) -> None:
        self._since: Dict[str, float] = {}

    def observe(self, v: str, wanted: bool, now: float) -> None:
        if wanted:
            self._since.setdefault(v, now)
        else:
            self._since.pop(v, None)

    def ready(self, v: str, now: float, t_v_s: float) -> bool:
        start = self._since.get(v)
        slots = max(1, int(math.ceil(t_v_s / config.MODEL_AUTOSCALER_INTERVAL_S)))
        return start is not None and now - start >= slots * config.MODEL_AUTOSCALER_INTERVAL_S - 1e-6

    def reset(self, v: str) -> None:
        self._since.pop(v, None)


def migrate_option(v: str, profile: VariantProfile) -> ScaleOption:
    """[P §4.1] interfered variant -> place it on the least-loaded worker."""
    return ScaleOption(MIGRATE, v, v, 1, action_cost(profile, 1) + action_cost(profile, -1))


# ------------------------------------------------------------ VM-Autoscaler rules
@dataclass
class WorkerStat:
    name: str
    hw: str
    util: float
    cpu_util: float
    has_interfered: bool
    has_overloaded: bool
    managed: bool          # created by the autoscaler (dynamic mode), may be deleted


def vm_scale_up(workers: Sequence[WorkerStat], flags: Sequence[str],
                max_by_hw: Dict[str, int]) -> Optional[tuple]:
    """(hw, reason) of the worker to add, or None. [P §4.2.3] rules 1-3."""
    counts: Dict[str, int] = {}
    for w in workers:
        counts[w.hw] = counts.get(w.hw, 0) + 1

    def room(hw: str) -> bool:
        return counts.get(hw, 0) < max_by_hw.get(hw, 0)

    for hw in hardware.types():
        ws = [w for w in workers if w.hw == hw]
        if not ws or not room(hw):
            continue
        # rule 1: "utilization ... exceeds a configurable threshold across all workers"
        if all(w.util > config.VM_UTIL_THRESHOLD for w in ws):
            return hw, "rule1_util"
        # rule 2: "variants on a particular hardware platform ... are in the
        # Interfered state across all workers"
        if all(w.has_interfered for w in ws):
            return hw, "rule2_interfered"
    # rule 3: "more than 80% of workers have Overloaded variants" -> the type is
    # not given; the most-overloaded type, else a faster one [PLAN G5]
    if workers:
        over = [w for w in workers if w.has_overloaded]
        if len(over) / len(workers) > config.VM_OVERLOADED_FRACTION:
            tally: Dict[str, int] = {}
            for w in over:
                tally[w.hw] = tally.get(w.hw, 0) + 1
            ranked = sorted(tally, key=lambda h: (-tally[h], hardware.cost(h)))
            for hw in ranked:
                if room(hw):
                    return hw, "rule3_overloaded"
            for hw in hardware.pricier_than(ranked[0]):
                if room(hw):
                    return hw, "rule3_overloaded_upgrade"
    # [C master_vm_daemon.cc:336-343] explicit VM-scale requests (vm_scale flag),
    # raised by the Dispatcher (mode 6) or by a placement that found no room
    for hw in sorted(set(flags), key=hardware.cost):
        if hardware.is_known(hw) and room(hw):
            return hw, "vm_scale_flag"
    return None


def vm_scale_down_ok(ws: Sequence[WorkerStat]) -> bool:
    """[C master_vm_daemon.cc:523-528] low-utilization condition, per GPU type [N]."""
    if not ws:
        return False
    utils = [w.util for w in ws]
    cpus = [w.cpu_util for w in ws]
    avg_u, avg_c = sum(utils) / len(utils), sum(cpus) / len(cpus)
    return ((min(utils) <= config.VM_SHUTDOWN_GPU_UTIL and min(cpus) <= config.VM_SHUTDOWN_CPU_UTIL)
            or (avg_u <= config.VM_SHUTDOWN_AVG_GPU_UTIL and avg_c <= config.VM_SHUTDOWN_AVG_CPU_UTIL))
