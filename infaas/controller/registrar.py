"""Model Registrar (paper §3.1-3.2): register_model -> Variant-Generator/Profiler.

For each GPU type, the variant (model, GPU type) is generated and profiled once,
on a worker of that type (paper §3.2: "a few minutes on a single VM with the
variant's target hardware"), then recorded in the Metadata Store. Profiles are
also written to PROFILE_DIR so a restart re-imports them instead of profiling
again. `submitter == "tester"` registers a stored profile directly, the way the
original reads a pre-made `.config` [C modelreg_server.cc: submitter == "tester"].

Profiling occupies the worker's GPU; register models before sending traffic.

PROFILE_MODE=original (default) needs no image, as the original profiler;
PROFILE_MODE=service needs the request image. A stored profile is reused only
if it was measured in the current mode.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import grpc

from infaas.common import config, events, profiles
from infaas.common.naming import variant_name
from infaas.controller.state import Registry, WorkerClients
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.proto import modelreg_pb2, modelreg_pb2_grpc, request_reply_pb2 as rr
from infaas.proto.internal import infaas_request_status_pb2 as ist, sys_monitor_pb2
from infaas.vendor.lumina import models

log = logging.getLogger("registrar")


def import_profiles(md: RedisMetadata, root: Optional[str] = None) -> int:
    n = skipped = 0
    for p in profiles.load_all(root):
        if not profiles.matches_mode(p):
            skipped += 1
            continue
        md.add_model(p)
        n += 1
    if skipped:
        log.warning("skipped %d stored profiles measured with another PROFILE_MODE "
                    "(current: %s); register those models again", skipped, config.PROFILE_MODE)
    return n


class Registrar(modelreg_pb2_grpc.ModelRegServicer):
    def __init__(self, md: RedisMetadata, registry: Registry, clients: WorkerClients) -> None:
        self.md = md
        self.registry = registry
        self.clients = clients
        # one profiling job per GPU type at a time: each needs a worker to itself
        self._hw_locks = {hw: threading.Lock() for hw in config.HW_TYPES}
        self._pool = ThreadPoolExecutor(max_workers=len(config.HW_TYPES) * 2)

    def Heartbeat(self, request, context):
        return modelreg_pb2.HeartbeatResponse(status=rr.RequestReply(status=rr.SUCCESS))

    def RegisterModel(self, request, context):
        if request.submitter == "tester" and request.profile_json:
            try:
                p = profiles.validate(json.loads(request.profile_json))
            except Exception as e:  # noqa: BLE001
                return self._reply(False, f"bad profile: {e}")
            if not profiles.matches_mode(p):
                return self._reply(False, f"profile was measured with PROFILE_MODE="
                                          f"{profiles.mode_of(p)}, this deployment uses "
                                          f"{config.PROFILE_MODE}")
            self.md.add_model(p)
            profiles.save(p)
            self.registry.invalidate()
            return self._reply(True, "registered from profile", [p["variant"]])

        model = request.parent_model
        if models.resolve(model) is None:
            return self._reply(False, f"unsupported model: {model}")
        hws = list(request.hardware) or list(config.HW_TYPES)
        unknown = [h for h in hws if h not in config.HW_TYPES]
        if unknown:
            return self._reply(False, f"unknown hardware {unknown}")
        self.md.add_parent_model(model)
        futs = [self._pool.submit(self._one, model, hw, request.profile_image,
                                  request.reprofile) for hw in hws]
        done, errors = [], []
        for f in futs:
            v, err = f.result()
            (errors.append(err) if err else done.append(v))
        self.registry.invalidate()
        return self._reply(not errors, "; ".join(errors) or "Successfully registered model", done)

    def _one(self, model: str, hw: str, image: bytes, reprofile: bool) -> Tuple[str, str]:
        v = variant_name(model, hw)
        stored = None if reprofile else profiles.load(v)
        if stored is not None and profiles.matches_mode(stored):
            self.md.add_model(stored)
            return v, ""
        if not image and config.PROFILE_MODE == "service":
            return v, f"{v}: PROFILE_MODE=service needs a profile_image to measure with"
        with self._hw_locks[hw]:
            worker, addr = self._pick_worker(hw)
            if not worker:
                return v, f"{v}: no {hw} worker registered"
            try:
                r = self.clients.sys(addr).ProfileVariant(
                    sys_monitor_pb2.ProfileRequest(model=model, image=image,
                                                   runs=config.PROFILE_RUNS,
                                                   sat_seconds=config.PROFILE_SAT_SECONDS,
                                                   sat_concurrency=config.PROFILE_SAT_CONCURRENCY),
                    timeout=config.LOAD_TIMEOUT_S + 120)
            except grpc.RpcError as e:
                return v, f"{v}: profiling on {worker} failed: {e.code().name}"
        if r.status.status != ist.SUCCESS:
            return v, f"{v}: {r.status.msg}"
        p = profiles.validate(json.loads(r.profile_json))
        p["profiled_on"] = worker
        profiles.save(p)
        self.md.add_model(p)
        events.event("profiled", variant=v, worker=worker,
                     lat_ms=p["inf_latency_ms"], load_ms=p["load_latency_ms"],
                     sat_qps=p["sat_qps"], mem=p["peak_memory_bytes"])
        return v, ""

    def _pick_worker(self, hw: str) -> Tuple[str, str]:
        """A worker of this type, preferring one with nothing loaded."""
        best: Optional[Tuple[int, str, str]] = None
        for w in self.md.get_all_executors():
            info = self.md.get_executor_info(w)
            if info.get("hw") != hw:
                continue
            n = len(self.md.get_variants_on_executor(w))
            cand = (n, w, info.get("addr", ""))
            if best is None or cand < best:
                best = cand
        return (best[1], best[2]) if best else ("", "")

    @staticmethod
    def _reply(ok: bool, msg: str, variants: List[str] = ()) -> modelreg_pb2.ModelRegResponse:
        return modelreg_pb2.ModelRegResponse(
            status=rr.RequestReply(status=rr.SUCCESS if ok else rr.INVALID, msg=msg),
            variants=list(variants))
