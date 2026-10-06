"""GPU types as INFaaS hardware platforms.

The paper's platforms are CPU / GPU / Inferentia; here they are the three GPU
types of the cluster (PLAN §0.3). Cost order is the user's [U C5]; the ratio is
[U G1]. Speed is never inferred from the order; it comes from each variant's
profile, because it differs by model (PLAN §2.5 of v1).
"""
from __future__ import annotations

from typing import List, Optional

from infaas.common import config


def types() -> List[str]:
    """GPU types, cheapest first."""
    return sorted(config.HW_TYPES, key=cost)


def cost(hw: str) -> float:
    return float(config.HW_COST.get(hw, float("inf")))


def tier(hw: str) -> int:
    """0 for the cheapest type."""
    return types().index(hw) if hw in config.HW_TYPES else len(config.HW_TYPES)


def cheaper_than(hw: str) -> List[str]:
    return [h for h in types() if cost(h) < cost(hw)]


def pricier_than(hw: str) -> List[str]:
    return [h for h in types() if cost(h) > cost(hw)]


def is_known(hw: Optional[str]) -> bool:
    return hw in config.HW_TYPES
