"""Variant profiles on disk: the persistent half of the registry.

The original keeps profiles as `.config` files in an S3 config bucket and loads
them at registration [C modelreg_server.cc]. Here each profile is one JSON file,
`<PROFILE_DIR>/<variant>.json`, on the control-plane's hostPath. The controller
re-imports them at startup so the registry survives restarts, and `profiles/` in
the repository can hold a copy for reproducibility.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from infaas.common import config

REQUIRED = ("variant", "model", "hardware", "inf_latency_ms", "load_latency_ms",
            "sat_qps", "peak_memory_bytes")


def validate(p: dict) -> dict:
    missing = [k for k in REQUIRED if k not in p]
    if missing:
        raise ValueError(f"profile missing {missing}")
    return p


def path_for(variant: str, root: Optional[str] = None) -> Path:
    return Path(root or config.PROFILE_DIR) / f"{variant}.json"


def save(p: dict, root: Optional[str] = None) -> Path:
    validate(p)
    path = path_for(p["variant"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def load(variant: str, root: Optional[str] = None) -> Optional[dict]:
    path = path_for(variant, root)
    if not path.exists():
        return None
    return validate(json.loads(path.read_text()))


def load_all(root: Optional[str] = None) -> List[dict]:
    d = Path(root or config.PROFILE_DIR)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            out.append(validate(json.loads(f.read_text())))
        except Exception:  # noqa: BLE001 — a broken file must not stop the rest
            continue
    return out


def by_variant(items: List[dict]) -> Dict[str, dict]:
    return {p["variant"]: p for p in items}
