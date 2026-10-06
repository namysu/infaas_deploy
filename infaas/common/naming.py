"""Model-variant names.

A variant here is (model, GPU type): batch 1 [U C3], FP32 PyTorch, GPU only [U C4].
The name keeps the parts readable in logs and in Redis: "resnet-50__a30".
"""
from __future__ import annotations

from typing import Tuple

SEP = "__"


def variant_name(model: str, hw: str) -> str:
    return f"{model}{SEP}{hw}"


def parse_variant(variant: str) -> Tuple[str, str]:
    """Return (model, hw)."""
    model, sep, hw = variant.rpartition(SEP)
    if not sep:
        raise ValueError(f"not a variant name: {variant!r}")
    return model, hw
