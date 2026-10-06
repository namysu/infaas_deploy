"""INFaaS controller process: Front-End + Dispatcher + Model Registrar (+ Placement).

[P §5] "On the controller machine, the Front-End, Dispatcher, and Model Registrar
are threads of the same process for efficient query dispatch." The VM-Autoscaler
is a separate process (infaas.controller.vm_autoscaler), as in the paper.
"""
from __future__ import annotations

import logging
import os
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from infaas.common import config
from infaas.controller.dispatcher import Dispatcher
from infaas.controller.frontend import ExecutorCompat, Frontend, QueryService
from infaas.controller.placement import Placement
from infaas.controller.registrar import Registrar, import_profiles
from infaas.controller.state import Executors, Registry, WorkerClients
from infaas.metadata.redis_metadata import RedisMetadata
from infaas.proto import modelreg_pb2_grpc, podexec_pb2_grpc, queryfe_pb2_grpc
from infaas.proto.internal import placement_pb2_grpc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [controller] %(name)s %(message)s")
log = logging.getLogger("controller")


def serve() -> None:
    md = RedisMetadata()
    md.r.ping()
    # like start_infaas.sh: start from a clean dynamic state, keep the registry
    md.flush_dynamic()
    n = import_profiles(md)
    log.info("imported %d variant profiles from %s", n, config.PROFILE_DIR)

    registry = Registry(md)
    executors = Executors(md)
    clients = WorkerClients()
    dispatcher = Dispatcher(md, registry, executors, config.DECISION_MODE)
    fe = Frontend(dispatcher, registry, clients)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.FRONTEND_THREADS),
                         options=[("grpc.max_receive_message_length", 64 << 20),
                                  ("grpc.max_send_message_length", 64 << 20)])
    queryfe_pb2_grpc.add_QueryServicer_to_server(QueryService(fe), server)
    podexec_pb2_grpc.add_ExecutorServicer_to_server(
        ExecutorCompat(fe, os.environ.get("PREDICT_PROXY", "")), server)
    modelreg_pb2_grpc.add_ModelRegServicer_to_server(Registrar(md, registry, clients), server)
    placement_pb2_grpc.add_PlacementServicer_to_server(Placement(md, registry, clients), server)
    hs = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(hs, server)
    for port in (config.QUERYFE_PORT, config.MODELREG_PORT, config.PLACEMENT_PORT,
                 config.COMPAT_PORT):
        server.add_insecure_port(f"[::]:{port}")
    server.start()
    for svc in ("", "infaaspublic.infaasqueryfe.Query", "podexec.Executor"):
        hs.set(svc, health_pb2.HealthCheckResponse.SERVING)
    log.info("controller up: queryfe :%d, modelreg :%d, placement :%d, lumina-compat :%d; "
             "decision mode %d, worker mode %s", config.QUERYFE_PORT, config.MODELREG_PORT,
             config.PLACEMENT_PORT, config.COMPAT_PORT, config.DECISION_MODE, config.WORKER_MODE)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
