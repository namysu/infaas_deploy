"""The ILP objective (paper §4.2.1), used as the greedy heuristic's cost function.

    Cost(δ_ij) = C_ij · (δ_ij + λ · T^load_ij · max(δ_ij, 0))

C_ij, the $/s of running one instance of variant v_ij: paper §6.2 estimates "the
cost for a running variant instance ... based on AWS EC2 pricing, proportional
to its memory footprint". Here the price is the per-GPU ratio [U G1] and the
footprint the profiled peak memory, so C_ij = HW_COST[hw] × GB.
"""
from __future__ import annotations

import math

from infaas.common import config
from infaas.policy.types import VariantProfile


def instance_cost(p: VariantProfile) -> float:
    gb = max(p.mem_bytes / 1e9, config.COST_MEMORY_FLOOR_GB)
    return config.HW_COST.get(p.hw, float("inf")) * gb


def action_cost(p: VariantProfile, delta: int) -> float:
    """Cost of changing the instance count of `p` by `delta` (negative = unload)."""
    return instance_cost(p) * (delta + config.LAMBDA * p.load_s * max(delta, 0))


def instances_needed(load_qps: float, sat_qps: float) -> int:
    """Instances of capacity `sat_qps` to carry `load_qps` with slack (constraint 1)."""
    if load_qps <= 0:
        return 0
    if sat_qps <= 0:
        return 10 ** 6
    return int(math.ceil(load_qps * config.SLACK_THRESHOLD / sat_qps - 1e-9))
