"""INFaaS worker: one pod, one GPU (the paper's worker VM, PLAN §2.2).

Startup mirrors Lumina's worker minus the model preload [PLAN A-05]: library
preload and CUDA init are engineering, shared by both systems; which models are
resident is policy, and in INFaaS they are loaded on demand.
"""
from __future__ import annotations

import logging
import os
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from infaas.common import config
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.proto.internal import query_pb2_grpc, sys_monitor_pb2_grpc
from infaas.vendor.lumina import dali_preprocess, preload_libs
from infaas.worker.autoscaler import ModelAutoscaler
from infaas.worker.context import WorkerContext
from infaas.worker.executor import QueryService, SysStatusService
from infaas.worker.monitor import Monitor
from infaas.worker.runtime import Runtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(name)s %(message)s")
log = logging.getLogger("worker")


def serve() -> None:
    name = os.environ.get("POD_NAME") or os.uname().nodename
    hw = os.environ["GPU_TYPE"]
    log.info("worker %s gpu=%s preprocessing=%s", name, hw, dali_preprocess.status())
    log.info("preload: %s", preload_libs.preload())

    runtime = Runtime(hw)
    runtime.warm_cuda()
    md = RedisMetadata()
    ctx = WorkerContext(name, hw, md, runtime)
    ctx.clear_stale()
    monitor = Monitor(ctx)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.WORKER_THREADS),
                         options=[("grpc.max_receive_message_length", 64 << 20),
                                  ("grpc.max_send_message_length", 64 << 20)])
    query_pb2_grpc.add_QueryServicer_to_server(QueryService(ctx), server)
    sys_monitor_pb2_grpc.add_SysStatusServicer_to_server(SysStatusService(ctx, monitor), server)
    hs = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(hs, server)
    server.add_insecure_port(f"[::]:{config.WORKER_PORT}")
    server.start()

    monitor.start()
    ModelAutoscaler(ctx, monitor).start()
    for svc in ("", "infaas.internal.Query"):
        hs.set(svc, health_pb2.HealthCheckResponse.SERVING)
    log.info("worker READY on :%d (gpu %s)", config.WORKER_PORT, monitor.gpu.name)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
