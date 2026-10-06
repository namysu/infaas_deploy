"""Placement: carries out Model-Autoscaler decisions on other workers.

[P §4.2.2] "if the strategy requires more resources than are available on the
current worker (e.g., hardware accelerator), the worker coordinates with the
controller to load the variant on a capable worker" and, when nothing fits,
"(d) Coordinate with the controller to invoke VM-level autoscaling".

Options arrive cheapest first; the first feasible one is executed:
  replicate  load `count` more instances of the variant
  upgrade / downgrade  load the other variant, then unload the requester's instance
  migrate    [P §4.1] interfered: load on the least-loaded worker, then unload
Target workers are chosen by bin packing [U G4], except migrate, which the paper
sends to "the least-loaded worker".
The requester holds `<variant>-pending`; it is cleared here when done.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import grpc

from infaas.common import config, events
from infaas.controller.state import Registry, WorkerClients
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.policy import scaling, selection, states
from infaas.policy.types import Snapshot, VariantProfile, WorkerView
from infaas.proto.internal import (infaas_request_status_pb2 as ist, placement_pb2,
                                   placement_pb2_grpc, sys_monitor_pb2)

log = logging.getLogger("placement")


def choose_targets(snap: Snapshot, dst: VariantProfile, count: int, kind: str,
                   src_worker: str) -> List[WorkerView]:
    """Workers to load `dst` on, updating the snapshot's free memory as it goes."""
    present = {i.worker for i in snap.of(dst.variant) if i.state in states.PRESENT}
    exclude = set(present)
    if kind == scaling.MIGRATE:
        exclude.add(src_worker)
    pick = (selection.least_loaded_placement if kind == scaling.MIGRATE
            else selection.pack_placement)
    out: List[WorkerView] = []
    for _ in range(max(1, count)):
        w = pick(snap.workers_of(dst.hw), dst.mem_bytes, exclude)
        if w is None:
            break
        out.append(w)
        exclude.add(w.name)
        w.mem_free -= dst.mem_bytes
    return out


class Placement(placement_pb2_grpc.PlacementServicer):
    def __init__(self, md: RedisMetadata, registry: Registry, clients: WorkerClients) -> None:
        self.md = md
        self.registry = registry
        self.clients = clients
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="placement")
        self._loads = ThreadPoolExecutor(max_workers=16, thread_name_prefix="placement-load")
        self._lock = threading.Lock()   # one placement decision at a time (consistent snapshot)

    def RequestScale(self, request, context):
        if not request.options:
            return placement_pb2.ScaleActionResponse(
                status=ist.InfaasRequestStatus(status=ist.INVALID, msg="no options"))
        self._pool.submit(self._execute, request)
        return placement_pb2.ScaleActionResponse(
            status=ist.InfaasRequestStatus(status=ist.SUCCESS, msg="accepted"))

    def _execute(self, request) -> None:
        src_variant = request.options[0].src_variant
        try:
            done = False
            for opt in request.options:
                dst = self.registry.profile(opt.dst_variant)
                if dst is None:
                    continue
                with self._lock:
                    snap = self.md.worker_snapshot()
                    targets = choose_targets(snap, dst, opt.count, opt.kind, request.src_worker)
                    for w in targets:   # visible to the Dispatcher at once
                        self.md.set_instance_state(w.name, dst.variant, states.LOADING)
                if not targets:
                    events.event("placement_infeasible", option=opt.kind, dst=opt.dst_variant,
                                 src_worker=request.src_worker)
                    continue
                ok = self._load_all(targets, dst.variant)
                events.event("placement", option=opt.kind, src_variant=opt.src_variant,
                             dst=opt.dst_variant, src_worker=request.src_worker,
                             targets=[w.name for w in targets], loaded=ok,
                             cost=round(opt.cost, 6), reason=request.reason)
                if ok and opt.kind in (scaling.UPGRADE, scaling.DOWNGRADE, scaling.MIGRATE):
                    self._unload(request.src_worker, opt.src_variant)
                done = bool(ok)
                if done:
                    break
            if not done:
                # (d) nothing fits: VM-level autoscaling for the cheapest option's GPU type
                first = self.registry.profile(request.options[0].dst_variant)
                if first is not None:
                    self.md.set_vm_scale(first.hw)
                    events.event("vm_scale_flag", hw=first.hw, path="placement",
                                 note=request.reason)
        except Exception:  # noqa: BLE001
            log.exception("placement failed")
        finally:
            self.md.clear_pending(src_variant)

    def _load_all(self, targets: List[WorkerView], variant: str) -> List[str]:
        def one(w: WorkerView) -> Optional[str]:
            try:
                r = self.clients.sys(w.addr).CreateModel(
                    sys_monitor_pb2.CreateMigrateRequest(model=variant, hardware=w.hw, numreplicas=1),
                    timeout=config.LOAD_TIMEOUT_S)
                if r.status.status == ist.SUCCESS:
                    return w.name
                log.warning("CreateModel %s on %s: %s", variant, w.name, r.status.msg)
            except grpc.RpcError as e:
                log.warning("CreateModel %s on %s: %s", variant, w.name, e.code())
            # the worker owns the state once it answers; clear only what we set
            if self.md.get_instance_states(variant).get(w.name) == states.LOADING:
                self.md.remove_running_model(w.name, variant)
            return None
        return [n for n in self._loads.map(one, targets) if n]

    def _unload(self, worker: str, variant: str) -> None:
        addr = self.md.get_executor_addr(worker)
        if not addr:
            return
        try:
            self.clients.sys(addr).ScaleDownModel(
                sys_monitor_pb2.ScaleRequest(model=variant, numreplicas=1), timeout=60.0)
        except grpc.RpcError as e:
            log.warning("ScaleDownModel %s on %s: %s", variant, worker, e.code())
