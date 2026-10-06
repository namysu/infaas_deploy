"""Controller-side caches over the Metadata Store, and worker gRPC clients.

Variant profiles are static after registration, so they are cached and refreshed
when `registry_version` moves (paper §3.2: "strategically uses data structures to
ensure low access latencies"). Dynamic state is always read fresh (snapshot()).
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import grpc

from infaas.common import config
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.policy.types import VariantProfile
from infaas.proto.internal import query_pb2_grpc, sys_monitor_pb2_grpc

_CH_OPTS = [("grpc.max_receive_message_length", 64 << 20),
            ("grpc.max_send_message_length", 64 << 20)]


class Registry:
    def __init__(self, md: RedisMetadata) -> None:
        self.md = md
        self._lock = threading.Lock()
        self._ts = 0.0
        self._version = -1
        self._by_model: Dict[str, List[VariantProfile]] = {}
        self._by_variant: Dict[str, VariantProfile] = {}

    def _refresh(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._ts < config.REGISTRY_CACHE_S:
            return
        with self._lock:
            if not force and now - self._ts < config.REGISTRY_CACHE_S:
                return
            v = self.md.registry_version()
            if force or v != self._version:
                self._version, by_model = self.md.load_registry()
                self._by_model = by_model
                self._by_variant = {p.variant: p for ps in by_model.values() for p in ps}
            self._ts = now

    def invalidate(self) -> None:
        self._refresh(force=True)

    def variants_of(self, model: str) -> List[VariantProfile]:
        self._refresh()
        return self._by_model.get(model, [])

    def profile(self, variant: str) -> Optional[VariantProfile]:
        self._refresh()
        return self._by_variant.get(variant)

    def models(self) -> List[str]:
        self._refresh()
        return sorted(self._by_model)


class Executors:
    """Cached executor list (membership changes on the VM-Autoscaler's 2 s cycle)."""

    def __init__(self, md: RedisMetadata) -> None:
        self.md = md
        self._ts = 0.0
        self._names: List[str] = []

    def names(self) -> List[str]:
        now = time.time()
        if now - self._ts > config.EXECUTOR_CACHE_S:
            self._names = self.md.get_all_executors()
            self._ts = now
        return self._names


class WorkerClients:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ch: Dict[str, grpc.Channel] = {}

    def _channel(self, addr: str) -> grpc.Channel:
        with self._lock:
            ch = self._ch.get(addr)
            if ch is None:
                ch = self._ch[addr] = grpc.insecure_channel(addr, options=_CH_OPTS)
            return ch

    def query(self, addr: str) -> query_pb2_grpc.QueryStub:
        return query_pb2_grpc.QueryStub(self._channel(addr))

    def sys(self, addr: str) -> sys_monitor_pb2_grpc.SysStatusStub:
        return sys_monitor_pb2_grpc.SysStatusStub(self._channel(addr))

    def drop(self, addr: str) -> None:
        with self._lock:
            ch = self._ch.pop(addr, None)
        if ch is not None:
            ch.close()
