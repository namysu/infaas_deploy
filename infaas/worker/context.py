"""State shared by the worker's executor, monitoring daemon and Model-Autoscaler.

In the original these live in one process too: qpsMonitor/resourceMonitor and
the autoscaler threads run inside query_executor [C query_executor.cc:142-166],
reading the executor's atomic counters.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import grpc

from infaas.common import config, events
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.policy import states
from infaas.policy.scaling import ScaleOption
from infaas.policy.types import VariantProfile
from infaas.proto.internal import placement_pb2, placement_pb2_grpc

if TYPE_CHECKING:   # torch-free import for the controller-side tests
    from infaas.worker.runtime import Instance, Runtime

log = logging.getLogger("worker.context")


class Counters:
    """Per-variant window counters (model_total_reqs/comp/lat/slo in the original)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reqs = 0
        self.comp = 0
        self.lat_sum = 0.0
        self.min_slo: Optional[float] = None

    def begin(self, slo_ms: float) -> None:
        with self._lock:
            self.reqs += 1
            if slo_ms > 0 and (self.min_slo is None or slo_ms < self.min_slo):
                self.min_slo = slo_ms

    def end(self, lat_ms: float) -> None:
        with self._lock:
            self.comp += 1
            self.lat_sum += lat_ms

    def take(self) -> Tuple[int, int, float, Optional[float]]:
        with self._lock:
            out = (self.reqs, self.comp, self.lat_sum, self.min_slo)
            self.reqs = self.comp = 0
            self.lat_sum = 0.0
            self.min_slo = None
            return out


class WorkerContext:
    def __init__(self, name: str, hw: str, md: RedisMetadata, runtime: "Runtime") -> None:
        self.name = name
        self.hw = hw
        self.md = md
        self.runtime = runtime
        self.counters: Dict[str, Counters] = {}
        self.states: Dict[str, str] = {}          # authoritative local state per instance
        self._lock = threading.Lock()
        self._profiles: Dict[str, VariantProfile] = {}
        self._registry: Tuple[float, int, Dict[str, List[VariantProfile]]] = (0.0, -1, {})
        self._placement = None
        self._placement_lock = threading.Lock()

    # ------------------------------------------------------------ registry
    def profile(self, variant: str) -> Optional[VariantProfile]:
        p = self._profiles.get(variant)
        if p is None:
            p = self.md.get_profile(variant)
            if p is not None:
                self._profiles[variant] = p
        return p

    def by_model(self) -> Dict[str, List[VariantProfile]]:
        ts, version, reg = self._registry
        if time.time() - ts > config.REGISTRY_CACHE_S:
            v = self.md.registry_version()
            if v != version:
                version, reg = self.md.load_registry()
                for ps in reg.values():
                    for p in ps:
                        self._profiles[p.variant] = p
            self._registry = (time.time(), version, reg)
        return self._registry[2]

    # ------------------------------------------------------------ instances
    def counters_for(self, variant: str) -> Counters:
        with self._lock:
            c = self.counters.get(variant)
            if c is None:
                c = self.counters[variant] = Counters()
            return c

    def running(self) -> List[str]:
        with self._lock:
            return [v for v, s in self.states.items() if s in states.RUNNING]

    def ensure_loaded(self, variant: str) -> Tuple["Instance", float, bool]:
        """Load on demand (the original loads inside QueryModelOnline, request path)."""
        inst = self.runtime.get(variant)
        if inst is not None:
            return inst, 0.0, False
        if not self.runtime.is_loading(variant):
            self.md.set_instance_state(self.name, variant, states.LOADING)
            with self._lock:
                self.states[variant] = states.LOADING
        try:
            inst, load_ms, did = self.runtime.load(variant)
        except Exception:
            with self._lock:
                if self.states.get(variant) == states.LOADING:
                    self.states.pop(variant, None)
            if self.runtime.get(variant) is None:
                self.md.remove_running_model(self.name, variant)
            raise
        if did:
            with self._lock:
                self.states[variant] = states.ACTIVE
                self.counters[variant] = Counters()
            self.md.update_instance(self.name, variant, states.ACTIVE, 0.0, 0.0)
            events.event("load", worker=self.name, variant=variant, load_ms=round(load_ms, 1))
        return inst, load_ms, did

    def unload(self, variant: str, reason: str = "") -> bool:
        with self._lock:
            if variant not in self.states:
                return False
            self.states[variant] = states.UNLOADING
        self.md.set_instance_state(self.name, variant, states.UNLOADING)
        ok = self.runtime.unload(variant)
        with self._lock:
            # a request may have started loading it again meanwhile; leave that alone
            mine = self.states.get(variant) == states.UNLOADING
            if mine:
                self.states.pop(variant, None)
                self.counters.pop(variant, None)
        if mine:
            self.md.remove_running_model(self.name, variant)
        events.event("unload", worker=self.name, variant=variant, reason=reason)
        return ok

    def clear_stale(self) -> None:
        """At start this worker holds nothing; drop anything the store still lists."""
        for v in self.md.get_variants_on_executor(self.name):
            self.md.remove_running_model(self.name, v)

    # ------------------------------------------------------------ placement
    def _stub(self):
        with self._placement_lock:
            if self._placement is None:
                ch = grpc.insecure_channel(config.CONTROLLER_ADDR)
                self._placement = placement_pb2_grpc.PlacementStub(ch)
            return self._placement

    def request_placement(self, options: Sequence[ScaleOption], reason: str) -> bool:
        """Ask the controller to carry out one of `options` (cheapest first)."""
        req = placement_pb2.ScaleActionRequest(src_worker=self.name, reason=reason)
        for o in options:
            req.options.add(kind=o.kind, src_variant=o.src_variant, dst_variant=o.dst_variant,
                            count=o.count, cost=o.cost)
        try:
            resp = self._stub().RequestScale(req, timeout=5.0)
        except grpc.RpcError as e:
            log.warning("placement request failed: %s", e.code())
            return False
        events.event("scale_request", worker=self.name, reason=reason,
                     options=[vars(o) for o in options],
                     accepted=resp.status.status == 1, msg=resp.status.msg)
        return resp.status.status == 1
