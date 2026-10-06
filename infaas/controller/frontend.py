"""Front-End (paper §3.2, Fig. 3 step 1-4): accept a query, select, dispatch, reply.

Two wire APIs over the same path:
  * native  infaaspublic.infaasqueryfe.Query  (the original public API)
  * compat  podexec.Executor/Infer            (Lumina's API, PLAN §3.4) so Lumina's
            benchmark scripts run unchanged against INFaaS
Requests carry image + model + latency SLO [U C1].
"""
from __future__ import annotations

import io
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import grpc

from infaas.common import config, events
from infaas.controller.dispatcher import Dispatcher
from infaas.controller.state import Registry, WorkerClients
from infaas.proto import (podexec_pb2, podexec_pb2_grpc, queryfe_pb2, queryfe_pb2_grpc,
                          request_reply_pb2 as rr)
from infaas.proto.internal import infaas_request_status_pb2 as ist
from infaas.proto.internal import query_pb2

log = logging.getLogger("frontend")


def gen_request_name() -> str:
    # same format as Lumina podexec/common/schemas.py
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)[:5]}"


@dataclass
class Result:
    request_name: str
    ok: bool
    reject_kind: str = ""
    reason: str = ""
    variant: str = ""
    worker: str = ""
    hardware: str = ""
    path: str = ""
    suggestion: str = ""
    label: str = ""
    boxes: list = field(default_factory=list)
    img_width: int = 0
    img_height: int = 0
    latency_ms: float = 0.0
    worker_ms: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)
    arrive_epoch_ms: float = 0.0
    reply_epoch_ms: float = 0.0


class Frontend:
    def __init__(self, dispatcher: Dispatcher, registry: Registry, clients: WorkerClients) -> None:
        self.dispatcher = dispatcher
        self.registry = registry
        self.clients = clients

    def serve(self, model: str, image: bytes, slo_ms: float,
              variant: Optional[str] = None) -> Result:
        name = gen_request_name()
        arrive = time.time() * 1000.0
        t0 = time.perf_counter()
        res = Result(request_name=name, ok=False, arrive_epoch_ms=arrive)
        if not image:
            res.reject_kind, res.reason = "bad_request", "empty image"
            return self._finish(res, model, slo_ms, t0)
        d, snap = self.dispatcher.decide(model, slo_ms, variant)
        t_dec = time.perf_counter()
        res.path, res.suggestion = d.path, d.suggestion
        res.timings["srv.decide"] = (t_dec - t0) * 1000.0
        if d.variant is None:
            res.reject_kind = d.reject_kind
            res.reason = d.note or d.reject_kind
            return self._finish(res, model, slo_ms, t0)
        w = snap.workers[d.worker]
        res.variant, res.worker, res.hardware = d.variant, d.worker, w.hw
        req = query_pb2.QueryOnlineRequest(raw_input=[image], model=[d.variant],
                                           slo=query_pb2.QuerySLO(LatencyInUSec=int(slo_ms * 1000)))
        t_rpc = time.perf_counter()
        try:
            r = self.clients.query(w.addr).QueryOnline(req, timeout=config.QUERY_TIMEOUT_S)
        except grpc.RpcError as e:
            res.reject_kind, res.reason = "worker_error", f"{d.worker}: {e.code().name}"
            return self._finish(res, model, slo_ms, t0)
        t_done = time.perf_counter()
        if r.status.status != ist.SUCCESS:
            res.reject_kind, res.reason = "worker_error", r.status.msg
            return self._finish(res, model, slo_ms, t0)
        rpc_ms = (t_done - t_rpc) * 1000.0
        wt = dict(r.timings)
        res.ok = True
        res.label = r.label
        res.boxes = [(b.x1, b.y1, b.x2, b.y2, b.label, b.score) for b in r.boxes]
        res.img_width, res.img_height = r.img_width, r.img_height
        res.worker_ms = wt.get("wall", r.latency_ms)
        res.timings["srv.dispatch_rpc"] = rpc_ms
        res.timings["srv.worker_transit"] = rpc_ms - res.worker_ms
        for k, v in wt.items():
            res.timings[f"worker.{k}"] = v
        prof = self.registry.profile(d.variant)
        if prof is not None:
            res.timings["prof.latency"] = prof.lat_ms
        return self._finish(res, model, slo_ms, t0)

    def _finish(self, res: Result, model: str, slo_ms: float, t0: float) -> Result:
        res.latency_ms = (time.perf_counter() - t0) * 1000.0
        res.timings["srv.total"] = res.latency_ms
        res.reply_epoch_ms = time.time() * 1000.0
        events.request(name=res.request_name, model=model, slo_ms=slo_ms, ok=res.ok,
                       reject_kind=res.reject_kind, variant=res.variant, worker=res.worker,
                       hw=res.hardware, path=res.path, latency_ms=round(res.latency_ms, 3),
                       load_ms=round(res.timings.get("worker.load", 0.0), 3),
                       decide_ms=round(res.timings.get("srv.decide", 0.0), 3))
        return res


# ------------------------------------------------------------ native API
class QueryService(queryfe_pb2_grpc.QueryServicer):
    def __init__(self, fe: Frontend) -> None:
        self.fe = fe

    def QueryOnline(self, request, context):
        out = queryfe_pb2.QueryOnlineResponse()
        if request.grandparent_model and not request.parent_model and not request.model_variant:
            out.status.CopyFrom(rr.RequestReply(
                status=rr.UNAVAILABLE, msg="grandparent (accuracy) queries are out of scope"))
            return out
        if len(request.raw_input) != 1:
            out.status.CopyFrom(rr.RequestReply(status=rr.INVALID, msg="batch 1: one input"))
            return out
        slo_ms = request.slo.LatencyInUSec / 1000.0
        res = self.fe.serve(request.parent_model, request.raw_input[0], slo_ms,
                            variant=request.model_variant or None)
        if res.ok:
            out.status.CopyFrom(rr.RequestReply(status=rr.SUCCESS, msg="Successfully executed query"))
            out.raw_output.append(res.label.encode())
        else:
            out.status.CopyFrom(rr.RequestReply(status=rr.UNAVAILABLE, msg=res.reason))
        out.variant, out.worker, out.hardware = res.variant, res.worker, res.hardware
        out.latency_ms, out.label, out.path = res.latency_ms, res.label, res.path
        out.suggested_variant, out.reject_kind = res.suggestion, res.reject_kind
        out.img_width, out.img_height = res.img_width, res.img_height
        for k, v in res.timings.items():
            out.timings[k] = v
        for x1, y1, x2, y2, lb, sc in res.boxes:
            out.detections.add(x1=x1, y1=y1, x2=x2, y2=y2, label=lb, score=sc)
        return out

    def QueryOffline(self, request, context):
        return queryfe_pb2.QueryOfflineResponse(
            status=rr.RequestReply(status=rr.UNAVAILABLE, msg="offline queries are out of scope"))

    def AllParentInfo(self, request, context):
        return queryfe_pb2.AllParResponse(reply=rr.AllParReply(
            all_models=self.fe.registry.models(), status=rr.RequestReply(status=rr.SUCCESS)))

    def QueryModelInfo(self, request, context):
        vs = self.fe.registry.variants_of(request.model)
        st = rr.RequestReply(status=rr.SUCCESS if vs else rr.UNAVAILABLE)
        return queryfe_pb2.QueryModelInfoResponse(reply=rr.QueryModelReply(
            all_models=[p.variant for p in vs], status=st))

    def Heartbeat(self, request, context):
        return queryfe_pb2.HeartbeatResponse(status=rr.RequestReply(status=rr.SUCCESS))


# ------------------------------------------------------------ Lumina-compatible API
class ExecutorCompat(podexec_pb2_grpc.ExecutorServicer):
    """podexec.Executor/Infer on the INFaaS path. `dispatch` is ignored except
    "predict", which INFaaS has no equivalent of: it is forwarded to a Lumina
    server when PREDICT_PROXY is set (the MINHIT ground truth stays Lumina's
    predictor, PLAN A-10), and rejected otherwise."""

    def __init__(self, fe: Frontend, predict_proxy: str = "") -> None:
        self.fe = fe
        self._proxy = (podexec_pb2_grpc.ExecutorStub(grpc.insecure_channel(predict_proxy))
                       if predict_proxy else None)

    def Infer(self, request, context):
        if request.dispatch == "predict":
            if self._proxy is not None:
                return self._proxy.Infer(request, timeout=30.0)
            return podexec_pb2.InferReply(rejected=True, reject_kind="bad_request",
                                          reason="dispatch=predict: set PREDICT_PROXY")
        res = self.fe.serve(request.model, request.image, request.slo)
        reply = podexec_pb2.InferReply(
            request_name=res.request_name, inference=res.label,
            latency=round(res.latency_ms, 2), gpu=res.hardware if res.ok else "",
            rejected=not res.ok, reason=res.reason, reject_kind=res.reject_kind,
            worker_ms=res.worker_ms, img_width=res.img_width, img_height=res.img_height,
            arrive_epoch_ms=res.arrive_epoch_ms, reply_epoch_ms=res.reply_epoch_ms)
        for k, v in res.timings.items():
            reply.timings[k] = v
        for x1, y1, x2, y2, lb, sc in res.boxes:
            reply.detections.add(x1=x1, y1=y1, x2=x2, y2=y2, label=lb, score=sc)
        for p in self.fe.registry.variants_of(request.model):
            reply.estimates[p.hw] = p.lat_ms
        return reply
