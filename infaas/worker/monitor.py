"""Monitoring Daemon (paper §3.2, §4, §5).

> "Every 2 seconds, the monitoring daemon updates compute and memory utilization
> of the worker, loading, and average inference latencies, along with the
> current state (as noted in Figure 4) for each variant running on that worker,
> to the Metadata Store."

Ports qpsMonitor / resourceMonitor [C query_executor.cc:522-852] and gpuMonitor
[C gpu_daemon.cc:80-140] into one thread. Differences, both logged in DEVIATIONS:
  * GPU "utilization" is NVML SM utilization averaged over the window [U G3];
    the original used memory occupancy (cudaMemGetInfo), kept here as mem_free.
  * CPU utilization is the pod's cgroup, not the whole host: two worker pods
    share each GPU node [N].
"""
from __future__ import annotations

import collections
import logging
import os
import threading
import time
from typing import Deque, Dict, Optional, Tuple

from infaas.common import config, events
from infaas.policy import states
from infaas.policy.scaling import migrate_option
from infaas.worker.context import WorkerContext

log = logging.getLogger("worker.monitor")


class GpuSampler:
    """SM utilization samples from NVML, averaged per monitoring window."""

    def __init__(self) -> None:
        self._h = None
        self._sum = 0.0
        self._n = 0
        self._lock = threading.Lock()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self._h)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
        except Exception as e:  # noqa: BLE001
            log.warning("NVML unavailable (%s); GPU utilization reads 0", e)
            self._nv = None
            self.name = "unknown"

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="gpu-sampler").start()

    def _loop(self) -> None:
        while True:
            if self._h is not None:
                try:
                    u = float(self._nv.nvmlDeviceGetUtilizationRates(self._h).gpu)
                    with self._lock:
                        self._sum += u
                        self._n += 1
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(config.UTIL_SAMPLE_S)

    def take_util(self) -> float:
        with self._lock:
            u = self._sum / self._n if self._n else 0.0
            self._sum, self._n = 0.0, 0
        return u

    def memory(self) -> Optional[Tuple[int, int, int]]:
        """(used, free, total) bytes of this pod's GPU; None if it cannot be read."""
        if self._h is None:
            return None
        m = self._nv.nvmlDeviceGetMemoryInfo(self._h)
        return int(m.used), int(m.free), int(m.total)


class CpuMeter:
    """CPU utilization of this pod from cgroup v2 (v1 fallback), 0-100."""

    def __init__(self) -> None:
        self._last: Optional[Tuple[float, float]] = None
        self.ncpu = self._ncpu()

    @staticmethod
    def _usage_us() -> Optional[float]:
        try:
            with open("/sys/fs/cgroup/cpu.stat") as f:
                for line in f:
                    k, v = line.split()
                    if k == "usage_usec":
                        return float(v)
        except OSError:
            pass
        try:
            with open("/sys/fs/cgroup/cpuacct/cpuacct.usage") as f:
                return float(f.read()) / 1000.0
        except OSError:
            return None

    @staticmethod
    def _ncpu() -> float:
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                quota, period = f.read().split()
                if quota != "max":
                    return max(float(quota) / float(period), 0.01)
        except (OSError, ValueError):
            pass
        return float(len(os.sched_getaffinity(0)))

    def take(self) -> float:
        now, usage = time.monotonic(), self._usage_us()
        if usage is None:
            return 0.0
        last, self._last = self._last, (now, usage)
        if last is None:
            return 0.0
        dt_us = (now - last[0]) * 1e6
        return max(0.0, min(100.0, (usage - last[1]) / (dt_us * self.ncpu) * 100.0)) if dt_us > 0 else 0.0


class Monitor:
    def __init__(self, ctx: WorkerContext) -> None:
        self.ctx = ctx
        self.gpu = GpuSampler()
        self.cpu = CpuMeter()
        self._last: Dict[str, Tuple[float, float, str]] = {}   # v -> (qps, avg_lat, state)
        self._windows: Dict[str, int] = {}                     # full windows seen since load
        self._slo: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    # read by the Model-Autoscaler
    def last_qps(self, v: str) -> float:
        with self._lock:
            return self._last.get(v, (0.0, 0.0, ""))[0]

    def observed(self, v: str) -> bool:
        """At least one monitoring window has measured this instance since its load."""
        with self._lock:
            return self._windows.get(v, 0) >= 1

    def min_slo(self, v: str) -> Optional[float]:
        with self._lock:
            d = self._slo.get(v)
            return min(d) if d else None

    def start(self) -> None:
        self.gpu.start()
        threading.Thread(target=self._loop, daemon=True, name="monitor").start()

    def _loop(self) -> None:
        self.cpu.take()
        prev = time.monotonic()
        while True:
            time.sleep(max(0.0, config.MONITOR_INTERVAL_S - (time.monotonic() - prev)))
            now = time.monotonic()
            interval, prev = now - prev, now
            try:
                self.window(interval)
            except Exception:  # noqa: BLE001 — the daemon must keep running
                log.exception("monitor window failed")

    def window(self, interval_s: float) -> None:
        ctx = self.ctx
        util = self.gpu.take_util()
        mem = self.gpu.memory()
        cpu = self.cpu.take()
        ctx.md.update_worker_stats(ctx.name, util, cpu, *(mem[1:] if mem else (None, None)))

        for v in ctx.running():
            reqs, comp, lat_sum, min_slo = ctx.counters_for(v).take()
            # an instance loaded inside this window has been up for less than it
            inst = ctx.runtime.get(v)
            up_s = time.time() - getattr(inst, "loaded_at", 0.0) if inst is not None else interval_s
            span = max(0.1, min(interval_s, up_s))
            qps = reqs / span
            with self._lock:
                prev_qps, prev_lat, _ = self._last.get(v, (0.0, 0.0, ""))
            if comp:
                avg_lat = lat_sum / comp
            else:
                avg_lat = prev_lat / config.AVGLAT_DECAY   # [C query_executor.cc:621-628]
            cur = ctx.states.get(v, states.ACTIVE)
            prof = ctx.profile(v)
            new = (states.next_state(cur, qps, avg_lat, prof.lat_ms, prof.sat_qps)
                   if prof is not None else states.ACTIVE)
            if ctx.states.get(v) in states.RUNNING:    # not unloaded meanwhile
                ctx.states[v] = new
                ctx.md.update_instance(ctx.name, v, new, qps, avg_lat)
            with self._lock:
                self._last[v] = (qps, avg_lat, new)
                self._windows[v] = self._windows.get(v, 0) + 1
                d = self._slo.setdefault(v, collections.deque(maxlen=config.MIN_SLO_WINDOWS))
                if min_slo is not None:
                    d.append(min_slo)
            if new != cur:
                events.event("state", worker=ctx.name, variant=v, old=cur, new=new,
                             qps=round(qps, 2), avg_lat=round(avg_lat, 2),
                             prof_lat=prof.lat_ms if prof else None,
                             sat_qps=prof.sat_qps if prof else None)
                if new == states.INTERFERED and prof is not None and ctx.md.set_pending(v):
                    # [P §4.1] "the worker asks the controller's Dispatcher to place
                    # the variant on the least-loaded worker"
                    threading.Thread(target=self._mitigate, args=(v, prof), daemon=True).start()
        with self._lock:
            for v in list(self._last):
                if v not in ctx.states:
                    self._last.pop(v, None)
                    self._slo.pop(v, None)
                    self._windows.pop(v, None)

    def _mitigate(self, v: str, prof) -> None:
        ok = self.ctx.request_placement([migrate_option(v, prof)], reason="interfered")
        if not ok:
            self.ctx.md.clear_pending(v)
