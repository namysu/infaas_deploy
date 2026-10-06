"""Case I — model-variant selection on arrival of a query (paper §4.1, Algorithm 1).

    GETVARIANT(appID, accuracy, latency)
      L2  if searchActiveVariants(...)        -> L3 pick worker running it -> L4 return
      L5  if searchInactiveVariants(...)      -> L6 pick worker with its HW -> L7 return
      L8  return suggestVariant(...)

Requests here carry a model name and a latency SLO only [U C1], so "requirements
match" means: a variant of that model whose profiled latency is within the SLO.

Worker choice is online bin packing [U G4]: paper §4.2.3 "To improve
utilization, INFaaS dispatches requests to workers using an online bin packing
algorithm [64]", and §6.3 credits the cost saving to "bin-packing requests
across models to one GPU at low load". Algorithm 1 names least-loaded /
lowest-utilization workers; with the user's decision, bin packing (Best-Fit)
decides instead, and least-loaded remains only the fallback when no bin has room.

`get_variant` is the replaceable policy entry point (paper §5: "getVariant is a
virtual method, and can be overridden to add new algorithms").
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence

from infaas.common import config, hardware
from infaas.policy import states
from infaas.policy.types import (Decision, InstanceView, Snapshot, VariantProfile,
                                 WorkerView)

MODE_INFAAS_ALL = 0
MODE_NOQPSLAT = 1
MODE_GPUSHARETRIGGER_SKIPBLIST = 6
SUPPORTED_MODES = (MODE_INFAAS_ALL, MODE_NOQPSLAT, MODE_GPUSHARETRIGGER_SKIPBLIST)


# ------------------------------------------------------------ bin packing
def pack_request(insts: Sequence[InstanceView], sat_qps: float,
                 rng: Optional[random.Random] = None) -> InstanceView:
    """Pick the instance a request goes to (Algorithm 1 L3 under bin packing).

    Best-Fit on throughput: the fullest instance that still has headroom below
    1/slack-threshold of its saturation throughput. If every instance is past
    that, fall back to the least-loaded one (Algorithm 1's own rule).
    """
    cap = sat_qps if sat_qps > 0 else float("inf")
    fill = {i.worker: i.qps / cap for i in insts}
    room = [i for i in insts if fill[i.worker] < 1.0 / config.SLACK_THRESHOLD]
    pool = list(room or insts)
    if rng is not None:
        rng.shuffle(pool)   # mode 6 shuffles equal candidates [C queryfe_server.cc:1263-1274]
    if room:
        return max(pool, key=lambda i: fill[i.worker])
    return min(pool, key=lambda i: fill[i.worker])


def fits(w: WorkerView, mem_bytes: int) -> bool:
    """Room for one more instance: memory (constraint 3, 1 GB slack) and compute."""
    return (not w.blacklisted
            and w.mem_free - config.MEMORY_SLACK_BYTES >= mem_bytes
            and w.util < config.VM_UTIL_THRESHOLD)


def pack_placement(workers: Iterable[WorkerView], mem_bytes: int,
                   exclude: Iterable[str] = ()) -> Optional[WorkerView]:
    """Pick the worker a new instance is loaded on (Algorithm 1 L6 under bin packing).

    Best-Fit on GPU memory: of the workers the instance fits on, the one left
    with the least free memory afterwards, so load collects on few GPUs.
    """
    skip = set(exclude)
    cands = [w for w in workers if w.name not in skip and fits(w, mem_bytes)]
    if not cands:
        return None
    return min(cands, key=lambda w: (w.mem_free - mem_bytes, -w.util, w.name))


def least_loaded_placement(workers: Iterable[WorkerView], mem_bytes: int,
                           exclude: Iterable[str] = ()) -> Optional[WorkerView]:
    """Lowest-utilization worker with room (paper §4.1 mitigation: "least-loaded worker")."""
    skip = set(exclude)
    cands = [w for w in workers if w.name not in skip and fits(w, mem_bytes)]
    if not cands:
        return None
    return min(cands, key=lambda w: (w.util, -w.mem_free, w.name))


# ------------------------------------------------------------ Algorithm 1
def get_variant(model: str, slo_ms: float, profiles: Sequence[VariantProfile],
                snap: Snapshot, mode: int = config.DECISION_MODE,
                rng: Optional[random.Random] = None) -> Decision:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"decision mode {mode} not supported ({SUPPORTED_MODES})")
    rng = rng if mode == MODE_GPUSHARETRIGGER_SKIPBLIST else None
    by_name: Dict[str, VariantProfile] = {p.variant: p for p in profiles}

    # requirements: profiled latency within the SLO
    cands = [p for p in profiles if p.lat_ms <= slo_ms]
    if not cands:
        # L8: nothing can meet it; suggest the closest one [P §4.1]
        best = min(profiles, key=lambda p: p.lat_ms) if profiles else None
        return Decision(None, None, "reject", reject_kind="no_variant",
                        suggestion=best.variant if best else "",
                        note=f"min profiled {best.lat_ms:.1f}ms > slo" if best else "no variants")

    vm_hw: Optional[str] = None
    running: List[InstanceView] = []      # candidates' loaded instances, any running state
    for p in cands:
        for inst in snap.of(p.variant):
            if inst.state in states.RUNNING:
                running.append(inst)
                if (mode == MODE_GPUSHARETRIGGER_SKIPBLIST and inst.state == states.INTERFERED
                        and vm_hw is None):
                    # [C queryfe_server.cc:894-901] blacklisted GPU variant -> ask for a VM
                    vm_hw = p.hw

    # ---- L2-L4: Active variants
    def usable(inst: InstanceView) -> bool:
        if snap.workers[inst.worker].blacklisted:
            return False
        if mode == MODE_NOQPSLAT:   # [C :243-247] "Just pick the model"
            return inst.state in states.RUNNING
        # [P §4.1] "INFaaS avoids selecting variants that are in the Interfered or
        # Overloaded state" (kept in mode 6 as well: PLAN G7)
        return inst.state == states.ACTIVE

    active: Dict[str, List[InstanceView]] = {}
    for inst in running:
        if usable(inst):
            active.setdefault(inst.variant, []).append(inst)
    if active:
        # Several matching Active variants: paper silent -> [C queryfe_server.cc:870-877]
        # the one closest to the SLO (min SLO - latency); ties to the cheaper GPU.
        v = min(active, key=lambda n: (slo_ms - by_name[n].lat_ms,
                                       hardware.cost(by_name[n].hw), n))
        inst = pack_request(active[v], by_name[v].sat_qps, rng)
        return Decision(v, inst.worker, "active", vm_scale_hw=vm_hw)

    # ---- a load already under way: join it rather than start another [N]
    loading = [(by_name[i.variant], i) for p in cands for i in snap.of(p.variant)
               if i.state == states.LOADING and not snap.workers[i.worker].blacklisted]
    if loading:
        p, inst = min(loading, key=lambda t: (t[0].tot_ms, t[1].worker))
        return Decision(p.variant, inst.worker, "loading", vm_scale_hw=vm_hw)

    # ---- L5-L7: Inactive variants (not loaded on any worker)
    inactive = sorted((p for p in cands if not snap.of(p.variant)),
                      key=lambda p: (p.tot_ms, hardware.cost(p.hw), p.variant))
    for p in inactive:
        w = pack_placement(snap.workers_of(p.hw), p.mem_bytes)
        if w is not None:
            return Decision(p.variant, w.name, "inactive", vm_scale_hw=vm_hw)
        # §4 case (d): "may not be loaded due to lack of resources" -> next one
        if vm_hw is None:
            vm_hw = p.hw

    # ---- matching variants exist but none usable: send to the least-loaded running
    # instance anyway [C queryfe_server.cc:1000-1020 picks the blacklisted one]
    if running:
        inst = min(running, key=lambda i: (i.qps / max(by_name[i.variant].sat_qps, 1e-9),
                                           i.worker))
        return Decision(inst.variant, inst.worker, "fallback",
                        vm_scale_hw=vm_hw or by_name[inst.variant].hw,
                        note=f"state={inst.state}")

    best = inactive[0] if inactive else cands[0]
    return Decision(None, None, "reject", reject_kind="no_capacity",
                    suggestion=best.variant, vm_scale_hw=vm_hw or best.hw,
                    note="no worker has room for any matching variant")


def direct(variant: str, profile: VariantProfile, snap: Snapshot) -> Decision:
    """A query naming a model-variant explicitly (Table 4, third form)."""
    insts = [i for i in snap.of(variant) if i.state in states.RUNNING
             and not snap.workers[i.worker].blacklisted]
    if insts:
        return Decision(variant, pack_request(insts, profile.sat_qps).worker, "direct")
    loading = [i for i in snap.of(variant) if i.state == states.LOADING]
    if loading:
        return Decision(variant, loading[0].worker, "direct")
    w = pack_placement(snap.workers_of(profile.hw), profile.mem_bytes)
    if w is None:
        return Decision(None, None, "reject", reject_kind="no_capacity",
                        vm_scale_hw=profile.hw)
    return Decision(variant, w.name, "direct")
