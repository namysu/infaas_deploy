"""A fixed worker list in place of the Kubernetes API (ORCHESTRATOR=static).

Used for docker deployments without a cluster manager (deploy/docker/): each
worker is a container bound to one GPU, started by docker compose. The list
comes from STATIC_WORKERS, "name=gpu_type@host:port,...", written by
deploy/docker/gen_compose.py from the GPU settings.

It offers the same three calls the VM-Autoscaler makes on the K8s adapter.
`list_workers` reports a worker ready while its gRPC heartbeat answers; it goes
down only after STATIC_FAIL_THRESHOLD misses in a row, so one slow answer does
not count as a failure (a down worker is a failure in the VM-Autoscaler's sense,
and its variants are restored when it comes back, paper §7). Workers cannot be
created or deleted here, so WORKER_MODE=dynamic is refused (vm_autoscaler.main).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import grpc

from infaas.common import config
from infaas.controller.k8s_adapter import PodInfo
from infaas.controller.state import WorkerClients
from infaas.proto.internal import infaas_request_status_pb2 as ist, query_pb2

log = logging.getLogger("static-workers")


@dataclass(frozen=True)
class StaticWorker:
    name: str
    hw: str
    host: str
    port: int

    @property
    def addr(self) -> str:
        return f"{self.host}:{self.port}"


def parse(spec: str) -> List[StaticWorker]:
    """Parse "name=hw@host:port,..." (whitespace and empty items ignored)."""
    out: List[StaticWorker] = []
    seen = set()
    for item in spec.replace("\n", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, eq, rest = item.partition("=")
        hw, at, hostport = rest.partition("@")
        host, colon, port = hostport.rpartition(":")
        if not (eq and at and colon and name and hw and host and port.isdigit()):
            raise ValueError(f"STATIC_WORKERS item {item!r}: expected name=gpu_type@host:port")
        if name in seen:
            raise ValueError(f"STATIC_WORKERS: duplicate worker name {name!r}")
        seen.add(name)
        out.append(StaticWorker(name.strip(), hw.strip(), host.strip(), int(port)))
    return out


class StaticWorkers:
    def __init__(self, workers: List[StaticWorker],
                 fail_threshold: int = config.STATIC_FAIL_THRESHOLD,
                 timeout_s: float = 1.0) -> None:
        if not workers:
            raise ValueError("ORCHESTRATOR=static needs STATIC_WORKERS")
        self.workers = workers
        self.fail_threshold = max(1, fail_threshold)
        self.timeout_s = timeout_s
        self.clients = WorkerClients()
        self._ever_up: Dict[str, bool] = {w.name: False for w in workers}
        self._misses: Dict[str, int] = {w.name: 0 for w in workers}
        self._pool = ThreadPoolExecutor(max_workers=min(16, len(workers)))

    @classmethod
    def from_config(cls) -> "StaticWorkers":
        return cls(parse(config.STATIC_WORKERS))

    def _alive(self, w: StaticWorker) -> bool:
        try:
            r = self.clients.query(w.addr).Heartbeat(query_pb2.HeartbeatRequest(),
                                                     timeout=self.timeout_s)
            return r.status.status == ist.SUCCESS
        except grpc.RpcError:
            return False

    def list_workers(self) -> Dict[str, PodInfo]:
        out: Dict[str, PodInfo] = {}
        for w, ok in zip(self.workers, self._pool.map(self._alive, self.workers)):
            if ok:
                self._ever_up[w.name], self._misses[w.name] = True, 0
            else:
                self._misses[w.name] += 1
            ready = self._ever_up[w.name] and self._misses[w.name] < self.fail_threshold
            out[w.name] = PodInfo(name=w.name, hw=w.hw, ip=w.host, ready=ready,
                                  phase="Running" if ready else "Unreachable",
                                  managed=False, deleting=False, addr=w.addr)
        return out

    def create_worker(self, hw: str) -> Optional[str]:
        log.error("static workers cannot be created (asked for %s); start a container instead", hw)
        return None

    def delete_pod(self, name: str) -> None:
        log.error("static workers cannot be deleted (asked for %s); stop the container instead", name)
