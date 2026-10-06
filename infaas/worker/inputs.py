"""A model's nominal input size, for profiling without an image (PROFILE_MODE=original).

The original profiler never sees a user image: GPU variants are timed by
trtis_perf_client on a randomly filled input buffer of the model's input shape
[C trtis_perf_client.cc:709-742], CPU variants on the bundled data/mug.jpg resized
to the model's input dimension [C profile_model.sh:50,197]. The input dimension
is a property of the model (`inputdim` in its .config). Here it is read from the
model's HuggingFace image processor: the size it crops or resizes to.

Torch-free so it can be tested without a GPU stack.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

DEFAULT_SIDE = 224


def _mapping(v) -> Optional[dict]:
    if v is None:
        return None
    if isinstance(v, dict):
        items = v.items()
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        items = dataclasses.asdict(v).items()
    elif hasattr(v, "keys"):
        items = ((k, v[k]) for k in v.keys())
    elif isinstance(v, int):
        return {"height": v, "width": v}
    else:
        return None
    return {k: val for k, val in items if val is not None}


def nominal_size(processor) -> Tuple[int, int]:
    """(width, height) of the image the model is fed: crop size, else resize size."""
    for attr in ("crop_size", "size"):
        if attr == "crop_size" and not getattr(processor, "do_center_crop", True):
            continue
        m = _mapping(getattr(processor, attr, None))
        if not m:
            continue
        if "height" in m and "width" in m:
            return int(m["width"]), int(m["height"])
        if "shortest_edge" in m:
            side = int(m["shortest_edge"])
            return side, side
    return DEFAULT_SIDE, DEFAULT_SIDE
