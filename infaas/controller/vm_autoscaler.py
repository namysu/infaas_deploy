"""VM-Autoscaler (paper §4.2.3) — here a worker-pod autoscaler. Separate process [P §5].

Every VM_AUTOSCALER_INTERVAL_S (2 s [P §5]):
  1. discover worker pods and keep the executor list in the Metadata Store in step
     (the original registers a VM once its heartbeat answers,
     [C master_vm_daemon.cc:430-447]); a worker that disappears without being
     retired is a failure, and its variants are restored on the next worker of
     the same type [P §7];
  2. (optional, code-only) blacklist workers above the utilization threshold;
  3. WORKER_MODE=dynamic only: apply rules 1-3 and the VM-scale flags to add a
     worker, or the original's low-utilization rule to remove one, with the
     original's backoff [C master_vm_daemon.cc:312-668].

WORKER_MODE=static (default) keeps every GPU's worker up, as Lumina does, so
step 3 is skipped; steps 1-2 run in both modes.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Set

import grpc

from infaas.common import config, events, hardware
from infaas.controller.k8s_adapter import K8s, PodInfo
from infaas.controller.state import WorkerClients
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.policy import scaling, states
from infaas.proto.internal import infaas_request_status_pb2 as ist, query_pb2, sys_monitor_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [vm-autoscaler] %(name)s %(message)s")
log = logging.getLogger("vm-autoscaler")


class VMAutoscaler:
    def __init__(self, md: RedisMetadata, k8s: K8s, mode: str = config.WORKER_MODE) -> None:
        if mode not in ("static", "dynamic"):
            raise ValueError(f"WORKER_MODE={mode!r}: use static or dynamic")
        self.md = md
        self.k8s = k8s
        self.mode = mode
        self.clients = WorkerClients()
        self.retiring: Set[str] = set()
        self.backoff = config.VM_BACKOFF_ITERS      # == limit: free to decide
        self.shutdown_count: Dict[str, int] = {hw: 0 for hw in config.HW_TYPES}

    # ------------------------------------------------------------ loop
    def run(self) -> None:
        log.info("VM-Autoscaler: mode=%s interval=%.1fs", self.mode, config.VM_AUTOSCALER_INTERVAL_S)
        if self.mode == "dynamic":
            self.ensure_initial()
        while True:
            t0 = time.monotonic()
            try:
                self.step()
            except Exception:  # noqa: BLE001
                log.exception("step failed")
            time.sleep(max(0.0, config.VM_AUTOSCALER_INTERVAL_S - (time.monotonic() - t0)))

    def step(self) -> None:
        pods = self.k8s.list_workers()
        self.sync(pods)
        if config.EXEC_BLACKLIST_UTIL_ENABLED:
            self.blacklist_hot()
        if self.mode == "dynamic":
            self.scale(pods)

    # ------------------------------------------------------------ membership
    def _heartbeat(self, addr: str) -> bool:
        try:
            r = self.clients.query(addr).Heartbeat(query_pb2.HeartbeatRequest(), timeout=2.0)
            return r.status.status == ist.SUCCESS
        except grpc.RpcError:
            return False

    def sync(self, pods: Dict[str, PodInfo]) -> None:
        registered = set(self.md.get_all_executors())
        live = {n: p for n, p in pods.items() if p.ready and p.ip and not p.deleting
                and hardware.is_known(p.hw)}
        for n, p in live.items():
            if n in registered or n in self.retiring:
                continue
            addr = f"{p.ip}:{config.WORKER_PORT}"
            if not self._heartbeat(addr):
                continue
            self.md.add_executor(n, addr, p.hw, managed=p.managed)
            events.event("worker_added", worker=n, hw=p.hw, managed=p.managed)
            restore = self.md.pop_recover(p.hw)
            if restore:
                threading.Thread(target=self._restore, args=(n, addr, restore), daemon=True).start()
        for n in registered - set(live):
            info = self.md.get_executor_info(n)
            variants = self.md.delete_executor(n)
            self.clients.drop(info.get("addr", ""))
            if n in self.retiring:
                self.retiring.discard(n)
                events.event("worker_removed", worker=n, cause="scale_down")
                continue
            hw = info.get("hw", "")
            self.md.push_recover(hw, variants)
            events.event("worker_failed", worker=n, hw=hw, variants=variants)
            pod = pods.get(n)
            if self.mode == "dynamic" and info.get("managed") == "1" and pod is None:
                # [P §7] "starts a new worker with the state of the failed worker"
                self.k8s.create_worker(hw)
        self.retiring &= set(pods)

    def _restore(self, worker: str, addr: str, variants: List[str]) -> None:
        for v in variants:
            try:
                self.clients.sys(addr).CreateModel(
                    sys_monitor_pb2.CreateMigrateRequest(model=v, numreplicas=1),
                    timeout=config.LOAD_TIMEOUT_S)
                events.event("restored", worker=worker, variant=v)
            except grpc.RpcError as e:
                log.warning("restore %s on %s: %s", v, worker, e.code())

    def blacklist_hot(self) -> None:
        # [C master_vm_daemon.cc:224-235] (off by default; DEVIATIONS D-22)
        snap = self.md.worker_snapshot()
        for w in snap.workers.values():
            if config.EXEC_BLACKLIST_UTIL < w.util < 100.0 and len(snap.workers) > 1:
                self.md.blacklist_executor(w.name)

    # ------------------------------------------------------------ dynamic mode
    def ensure_initial(self) -> None:
        pods = self.k8s.list_workers()
        for hw, n in config.DYN_INIT_WORKERS.items():
            have = sum(1 for p in pods.values() if p.hw == hw and not p.deleting)
            for _ in range(max(0, n - have)):
                self.k8s.create_worker(hw)

    def _stats(self) -> List[scaling.WorkerStat]:
        snap = self.md.worker_snapshot()
        out = []
        for w in snap.workers.values():
            insts = [i for insts in snap.instances.values() for i in insts if i.worker == w.name]
            info = self.md.get_executor_info(w.name)
            out.append(scaling.WorkerStat(
                name=w.name, hw=w.hw, util=w.util, cpu_util=float(info.get("cpu_util", 0.0)),
                has_interfered=any(i.state == states.INTERFERED for i in insts),
                has_overloaded=any(i.state == states.OVERLOADED for i in insts),
                managed=info.get("managed") == "1"))
        return out

    def scale(self, pods: Dict[str, PodInfo]) -> None:
        if self.backoff != config.VM_BACKOFF_ITERS:
            self.backoff += 1
            if self.backoff == config.VM_BACKOFF_ITERS:
                self.md.unset_vm_scale()           # [C master_vm_daemon.cc:655-668]
            return
        stats = self._stats()
        flags = self.md.vm_scale_flags()
        # pods still starting count against the per-type maximum
        starting = [scaling.WorkerStat(n, p.hw, 0.0, 0.0, False, False, p.managed)
                    for n, p in pods.items()
                    if not p.deleting and n not in {s.name for s in stats} and hardware.is_known(p.hw)]
        up = scaling.vm_scale_up(stats + starting, flags, config.MAX_WORKERS) \
            if not starting else None
        if up is not None:
            hw, reason = up
            name = self.k8s.create_worker(hw)
            events.event("vm_scale_up", hw=hw, reason=reason, pod=name, flags=flags)
            self.backoff = 0
            for h in config.HW_TYPES:
                self.shutdown_count[h] = 0
            return
        if flags:
            return
        for hw in config.HW_TYPES:
            ws = [s for s in stats if s.hw == hw]
            floor = config.DYN_INIT_WORKERS.get(hw, 0)
            if len(ws) <= floor or not scaling.vm_scale_down_ok(ws):
                self.shutdown_count[hw] = 0
                continue
            self.shutdown_count[hw] += 1
            if self.shutdown_count[hw] < config.VM_SHUTDOWN_ITERS:
                continue
            self.shutdown_count[hw] = 0
            victims = sorted((s for s in ws if s.managed), key=lambda s: (s.util, s.name))
            if not victims:
                continue
            v = victims[0]
            self.retiring.add(v.name)
            self.md.delete_executor(v.name)       # stop dispatching first
            self.k8s.delete_pod(v.name)
            events.event("vm_scale_down", hw=hw, pod=v.name, util=v.util)
            self.backoff = 0
            return


def main() -> None:
    md = RedisMetadata()
    while True:
        try:
            md.r.ping()
            break
        except Exception:  # noqa: BLE001
            log.info("waiting for redis ...")
            time.sleep(1.0)
    VMAutoscaler(md, K8s()).run()


if __name__ == "__main__":
    main()
