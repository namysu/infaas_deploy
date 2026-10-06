"""Model-Autoscaler at each worker (paper §4.2.1-4.2.2), 1 s period [P §5].

Decisions are in infaas.policy.scaling; this thread gathers the inputs and acts:
  scale up   -> placement request to the controller (cheapest option first)
  remove     -> unload locally, after T_v
  downgrade  -> placement request, after T_v

AUTOSCALER_TYPE mirrors the original's AutoscalerType [C autoscaler.h:53-58]:
  infaas      replicate or upgrade/downgrade (default)
  individual  replicate / remove only, no model-vertical scaling (≈ SM+ baseline)
  none        no model-level scaling (≈ Clipper+ baseline: fixed instances)
"""
from __future__ import annotations

import logging
import os
import threading
import time

from infaas.common import config
from infaas.policy import scaling
from infaas.policy.scaling import ClusterStat, LocalStat
from infaas.worker.context import WorkerContext
from infaas.worker.monitor import Monitor

log = logging.getLogger("worker.autoscaler")

AUTOSCALER_TYPE = os.environ.get("AUTOSCALER_TYPE", "infaas").lower()


class ModelAutoscaler:
    def __init__(self, ctx: WorkerContext, monitor: Monitor) -> None:
        self.ctx = ctx
        self.monitor = monitor
        self.timer = scaling.ScaleDownTimer()
        self.kind = AUTOSCALER_TYPE

    def start(self) -> None:
        if self.kind == "none":
            log.info("model autoscaler disabled (AUTOSCALER_TYPE=none)")
            return
        threading.Thread(target=self._loop, daemon=True, name="model-autoscaler").start()

    def _loop(self) -> None:
        while True:
            t0 = time.monotonic()
            try:
                self.tick(time.time())
            except Exception:  # noqa: BLE001
                log.exception("autoscaler tick failed")
            time.sleep(max(0.0, config.MODEL_AUTOSCALER_INTERVAL_S - (time.monotonic() - t0)))

    def tick(self, now: float) -> None:
        ctx = self.ctx
        # an instance is judged only on a measured window: right after a load its
        # QPS still reads 0 (the original waits for a non-zero average batch and
        # 20 consecutive scale-down requests, [C autoscaler.cc:410-415, 1181])
        running = [v for v in ctx.running() if self.monitor.observed(v)]
        profiles = {v: p for v in running if (p := ctx.profile(v)) is not None}
        if not profiles:
            return
        local = {v: LocalStat(self.monitor.last_qps(v), self.monitor.min_slo(v)) for v in profiles}
        by_model = ctx.by_model()
        cluster = {v: ClusterStat(n, q) for v, (n, q) in ctx.md.cluster_stats(list(profiles)).items()}

        ups = scaling.scale_up_options(local, profiles, by_model, cluster)
        for v, opts in ups.items():
            self.timer.reset(v)
            if self.kind == "individual":
                opts = [o for o in opts if o.kind == scaling.REPLICATE]
            if not opts:
                continue
            if ctx.md.set_pending(v):
                if not ctx.request_placement(opts, reason="scale_up"):
                    ctx.md.clear_pending(v)

        for v, p in profiles.items():
            if v in ups:
                continue
            opt = scaling.scale_down_option(v, local[v], p, by_model,
                                            cluster.get(v, ClusterStat(1, local[v].qps)))
            if opt is not None and self.kind == "individual" and opt.kind != scaling.REMOVE:
                opt = None
            self.timer.observe(v, opt is not None, now)
            if opt is None or not self.timer.ready(v, now, p.load_s):
                continue
            self.timer.reset(v)
            if opt.kind == scaling.REMOVE:
                # a short lock so two workers do not both drop the last spare instance
                if ctx.md.set_pending(v, ttl_s=2 * config.MONITOR_INTERVAL_S):
                    threading.Thread(target=ctx.unload, args=(v, "scale_down_remove"),
                                     daemon=True).start()
            elif ctx.md.set_pending(v):
                if not ctx.request_placement([opt], reason="scale_down"):
                    ctx.md.clear_pending(v)
