"""Model-variant instance state machine (paper §4, Fig. 4).

    Inactive --Loaded--> Active --QPS >= peak--> Overloaded --QPS < peak--> Active
    Active --Contended--> Interfered --Mitigated--> Active
    Overloaded --Contended; QPS < peak--> Interfered
    any --Unloaded--> Inactive

LOADING and UNLOADING are the transitions themselves made visible, so the
Dispatcher never picks an instance that is not there yet or about to go away
(the code's `load_unload` flag, redis_metadata.h LOADUNL_SUFF).
"""
from __future__ import annotations

from infaas.common import config

INACTIVE = "INACTIVE"      # not loaded (no entry in the store)
LOADING = "LOADING"
ACTIVE = "ACTIVE"
OVERLOADED = "OVERLOADED"
INTERFERED = "INTERFERED"
UNLOADING = "UNLOADING"

RUNNING = frozenset({ACTIVE, OVERLOADED, INTERFERED})
PRESENT = frozenset({LOADING, ACTIVE, OVERLOADED, INTERFERED, UNLOADING})


def interfered_factor(prof_lat_ms: float) -> float:
    """How far above the profiled latency counts as contended.

    Paper: "experiencing higher inference latencies than the profiled values",
    no factor given -> [C query_executor.cc:690-696] GPU branch.
    """
    if prof_lat_ms < config.INTERFERED_SMALL_LAT_MS:
        return min(config.INTERFERED_SMALL_CAP,
                   config.INTERFERED_SMALL_NUM / max(prof_lat_ms, 1e-6))
    return config.INTERFERED_LAT_FACTOR


def next_state(cur: str, qps: float, avg_lat_ms: float,
               prof_lat_ms: float, sat_qps: float) -> str:
    """One monitoring-window transition for a loaded instance (cur in RUNNING)."""
    # [P Fig.4] QPS >= peak -> Overloaded; QPS < peak leaves it.
    if sat_qps > 0 and qps >= sat_qps:
        return OVERLOADED
    # [P §4] Interfered: not overloaded, latency above profiled.
    # [C query_executor.cc:711-712] also requires qps > 0.3 x capacity, so a
    # single slow request at trivial load does not count as contention.
    contended = (avg_lat_ms > prof_lat_ms * interfered_factor(prof_lat_ms)
                 and qps > config.INTERFERED_QPS_FRACTION * sat_qps)
    if contended:
        return INTERFERED
    # [C query_executor.cc:723-726] leave Interfered only below 1.25x profiled
    if cur == INTERFERED and avg_lat_ms >= prof_lat_ms * config.INTERFERED_UNSET_FACTOR:
        return INTERFERED
    return ACTIVE
