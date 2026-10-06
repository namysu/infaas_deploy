"""Dispatcher (paper §3.2): Case I variant selection for every query.

The policy is infaas.policy.selection.get_variant (Algorithm 1). This class
reads the snapshot it needs in one Redis round trip, applies the mode-specific
side effects of the original, and returns the decision.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Dict, List, Optional, Tuple

from infaas.common import config, events
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.policy import selection
from infaas.policy.types import Decision, Snapshot, VariantProfile
from infaas.controller.state import Executors, Registry


class Dispatcher:
    def __init__(self, md: RedisMetadata, registry: Registry, executors: Executors,
                 mode: int = config.DECISION_MODE) -> None:
        if mode not in selection.SUPPORTED_MODES:
            raise ValueError(f"DECISION_MODE={mode} not supported {selection.SUPPORTED_MODES}")
        self.md = md
        self.registry = registry
        self.executors = executors
        self.mode = mode
        self._rng = random.Random()
        self._lock = threading.Lock()
        self._vm_flag_ts: Dict[str, float] = {}
        self._qps_win: Dict[str, Tuple[float, int]] = {}

    def decide(self, model: str, slo_ms: float,
               variant: Optional[str] = None) -> Tuple[Decision, Snapshot]:
        if variant:
            prof = self.registry.profile(variant)
            if prof is None:
                return Decision(None, None, "reject", reject_kind="bad_request",
                                note=f"variant {variant} not registered"), Snapshot()
            snap = self.md.snapshot([variant], self.executors.names())
            d = selection.direct(variant, prof, snap)
        else:
            profiles = self.registry.variants_of(model)
            if not profiles:
                return Decision(None, None, "reject", reject_kind="bad_request",
                                note=f"model {model} not registered"), Snapshot()
            snap = self.md.snapshot([p.variant for p in profiles], self.executors.names())
            with self._lock:
                d = selection.get_variant(model, slo_ms, profiles, snap, self.mode, self._rng)
        self._side_effects(d, snap)
        return d, snap

    def _side_effects(self, d: Decision, snap: Snapshot) -> None:
        now = time.time()
        # [C queryfe_server.cc:894-901] modes 4-6 ask the VM-Autoscaler for a worker
        if d.vm_scale_hw and self.mode == selection.MODE_GPUSHARETRIGGER_SKIPBLIST:
            if now - self._vm_flag_ts.get(d.vm_scale_hw, 0.0) > 1.0:
                self._vm_flag_ts[d.vm_scale_hw] = now
                self.md.set_vm_scale(d.vm_scale_hw)
                events.event("vm_scale_flag", hw=d.vm_scale_hw, path=d.path, note=d.note)
        # [C queryfe_server.cc:1606-1657] >= 200 requests to one worker within 1 s
        # -> blacklist it for 2 s (code-only, off by default: DEVIATIONS D-22)
        if d.worker and config.EXEC_BLACKLIST_QPS_ENABLED:
            with self._lock:
                start, n = self._qps_win.get(d.worker, (now, 0))
                if now - start < 1.0:
                    n += 1
                    if n >= config.EXEC_BLACKLIST_QPS_LIMIT:
                        if len(snap.workers) > 1:
                            self.md.blacklist_executor(d.worker)
                            events.event("exec_blacklist", worker=d.worker, cause="qps")
                        start, n = now, 0
                else:
                    start, n = now, 0
                self._qps_win[d.worker] = (start, n)
