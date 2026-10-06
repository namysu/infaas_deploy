"""Plain data the policies decide on. No I/O here, so every policy is testable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class VariantProfile:
    """Static, profiled facts about one model-variant (paper §3.2 Variant-Profiler)."""
    variant: str
    model: str            # parent model (Lumina short name)
    hw: str               # GPU type
    lat_ms: float         # profiled latency: worker service time incl. preprocessing [U C6]
    load_ms: float        # profiled loading latency (to Active, incl. warmup)
    sat_qps: float        # saturation throughput Q_ij
    mem_bytes: int        # peak GPU memory of one instance

    @property
    def load_s(self) -> float:
        return self.load_ms / 1000.0

    @property
    def tot_ms(self) -> float:
        """Combined loading + inference latency (Algorithm 1 L5)."""
        return self.load_ms + self.lat_ms


@dataclass
class WorkerView:
    name: str
    hw: str
    addr: str
    util: float = 0.0          # SM utilization %, window average [U G3]
    mem_free: int = 0          # bytes
    mem_total: int = 0
    blacklisted: bool = False  # code-only executor blacklist (off by default)


@dataclass
class InstanceView:
    """One {variant, worker} pair (paper §5)."""
    variant: str
    worker: str
    state: str
    qps: float = 0.0


@dataclass
class Snapshot:
    """Dynamic state read from the Metadata Store for one decision."""
    workers: Dict[str, WorkerView] = field(default_factory=dict)
    instances: Dict[str, List[InstanceView]] = field(default_factory=dict)  # by variant

    def of(self, variant: str) -> List[InstanceView]:
        return [i for i in self.instances.get(variant, []) if i.worker in self.workers]

    def workers_of(self, hw: str) -> List[WorkerView]:
        return [w for w in self.workers.values() if w.hw == hw]

    def variants_on(self, worker: str) -> List[str]:
        return [v for v, insts in self.instances.items()
                if any(i.worker == worker for i in insts)]


@dataclass
class Decision:
    variant: Optional[str]
    worker: Optional[str]
    path: str                       # active | loading | inactive | fallback | direct | reject
    reject_kind: str = ""           # no_variant | no_capacity
    suggestion: str = ""            # Algorithm 1 L8
    vm_scale_hw: Optional[str] = None
    note: str = ""
