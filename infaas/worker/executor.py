"""Worker gRPC services: internal Query (master -> worker) and SysStatus.

Query.QueryOnline follows the original executor [C query_executor.cc:255-424]:
one model-variant per request, loaded on demand when it is not resident
[C common_model_util.cc:901-930], request counters updated around the call.
The load, when it happens, is reported separately and kept out of the latency
statistics: the paper's Interfered state compares *inference* latency with the
profile (§4), and a cold load is not that.
"""
from __future__ import annotations

import json
import logging
import time

import grpc

from infaas.common.naming import parse_variant
from infaas.proto.internal import (infaas_request_status_pb2 as st, query_pb2, query_pb2_grpc,
                                   sys_monitor_pb2, sys_monitor_pb2_grpc)
from infaas.worker.context import WorkerContext

log = logging.getLogger("worker.executor")


def _status(ok: bool, msg: str = "") -> st.InfaasRequestStatus:
    return st.InfaasRequestStatus(status=st.SUCCESS if ok else st.UNAVAILABLE, msg=msg)


class QueryService(query_pb2_grpc.QueryServicer):
    def __init__(self, ctx: WorkerContext) -> None:
        self.ctx = ctx

    def Heartbeat(self, request, context):
        return query_pb2.HeartbeatResponse(status=_status(True, self.ctx.name))

    def QueryOnline(self, request, context):
        resp = query_pb2.QueryOnlineResponse()
        if len(request.model) != 1 or len(request.raw_input) != 1:
            resp.status.CopyFrom(st.InfaasRequestStatus(
                status=st.INVALID, msg="exactly one model-variant and one input"))
            return resp
        variant = request.model[0]
        try:
            _, hw = parse_variant(variant)
        except ValueError as e:
            resp.status.CopyFrom(st.InfaasRequestStatus(status=st.INVALID, msg=str(e)))
            return resp
        if hw != self.ctx.hw:
            resp.status.CopyFrom(st.InfaasRequestStatus(
                status=st.INVALID, msg=f"{variant} needs {hw}, worker is {self.ctx.hw}"))
            return resp
        slo_ms = request.slo.LatencyInUSec / 1000.0
        t0 = time.perf_counter()
        try:
            inst, load_ms, loaded = self.ctx.ensure_loaded(variant)
        except Exception as e:  # noqa: BLE001
            log.warning("load %s failed: %s", variant, e)
            resp.status.CopyFrom(_status(False, f"load failed: {e}"))
            return resp
        counters = self.ctx.counters_for(variant)
        counters.begin(slo_ms)
        try:
            label, boxes, (w, h), tm = inst.infer(request.raw_input[0])
        except Exception as e:  # noqa: BLE001
            counters.end((time.perf_counter() - t0) * 1000.0 - load_ms)
            resp.status.CopyFrom(_status(False, f"inference failed: {e}"))
            return resp
        lat = tm["total"]
        counters.end(lat)
        tm["load"] = load_ms
        tm["swap"] = load_ms           # name Lumina's scripts read for model movement
        tm["wall"] = (time.perf_counter() - t0) * 1000.0
        resp.status.CopyFrom(_status(True))
        resp.raw_output.append(label.encode())
        resp.label = label
        resp.img_width, resp.img_height = w, h
        resp.loaded = loaded
        resp.variant = variant
        resp.latency_ms = lat
        for k, v in tm.items():
            resp.timings[k] = float(v)
        for x1, y1, x2, y2, lb, sc in boxes:
            resp.boxes.add(x1=x1, y1=y1, x2=x2, y2=y2, label=lb, score=sc)
        return resp

    def QueryOffline(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "offline queries are out of scope")


class SysStatusService(sys_monitor_pb2_grpc.SysStatusServicer):
    def __init__(self, ctx: WorkerContext, monitor) -> None:
        self.ctx = ctx
        self.monitor = monitor

    def CreateModel(self, request, context):
        try:
            _, load_ms, _ = self.ctx.ensure_loaded(request.model)
        except Exception as e:  # noqa: BLE001
            return sys_monitor_pb2.CreateMigrateResponse(status=_status(False, str(e)))
        return sys_monitor_pb2.CreateMigrateResponse(status=_status(True), load_ms=load_ms)

    def ScaleUpModel(self, request, context):
        # GPU_MAX_REPLICAS = 1 per worker: a second instance is never on the same GPU
        if self.ctx.runtime.get(request.model) is not None:
            return sys_monitor_pb2.ScaleResponse(status=_status(False, "GPU_MAX_REPLICAS reached"))
        try:
            self.ctx.ensure_loaded(request.model)
        except Exception as e:  # noqa: BLE001
            return sys_monitor_pb2.ScaleResponse(status=_status(False, str(e)))
        return sys_monitor_pb2.ScaleResponse(status=_status(True))

    def ScaleDownModel(self, request, context):
        ok = self.ctx.unload(request.model, reason="controller")
        return sys_monitor_pb2.ScaleResponse(status=_status(ok, "" if ok else "not loaded"))

    def MigrateModel(self, request, context):
        return sys_monitor_pb2.CreateMigrateResponse(
            status=_status(False, "migration is orchestrated by the controller's Placement"))

    def ProfileVariant(self, request, context):
        from infaas.worker import profiler   # needs torch; imported where used
        try:
            prof = profiler.profile(self.ctx.runtime, self.monitor.gpu, request.model,
                                    request.image, request.runs, request.sat_seconds,
                                    request.sat_concurrency)
        except Exception as e:  # noqa: BLE001
            log.exception("profiling %s failed", request.model)
            return sys_monitor_pb2.ProfileResponse(status=_status(False, str(e)))
        return sys_monitor_pb2.ProfileResponse(status=_status(True), profile_json=json.dumps(prof))
