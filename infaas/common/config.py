"""Every tunable constant, in one place, with where its value comes from.

Source tags (PLAN §0.2):
  [P §x]  the paper states it
  [C f:l] the paper is silent or unrealistic here, so the released code decides
  [U Gx]  confirmed by the user (PLAN §5)
  [N]     new, forced by the Kubernetes port; logged in docs/DEVIATIONS.md

Every value can be overridden by an environment variable of the same name (the
`infaas-config` ConfigMap sets them in the cluster).
"""
from __future__ import annotations

import os
from typing import Dict


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _s(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _hw_map(name: str, default: str, cast=float) -> Dict[str, float]:
    """Parse "2080ti:1,a5000:2,a30:4" into a dict."""
    out = {}
    for item in _s(name, default).split(","):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.partition(":")
        out[k.strip()] = cast(v)
    return out


# ---------------------------------------------------------------- hardware
# GPU types in increasing cost order. [U C5] 2080ti < a5000 < a30
HW_TYPES = [h.strip() for h in _s("HW_TYPES", "2080ti,a5000,a30").split(",") if h.strip()]
# Relative price per GPU. [U G1] 1:2:4
HW_COST: Dict[str, float] = _hw_map("HW_COST", "2080ti:1,a5000:2,a30:4")

# ---------------------------------------------------------------- periods
# [P §5] "arrived at a 1 second polling interval" (Model-Autoscaler)
MODEL_AUTOSCALER_INTERVAL_S = _f("MODEL_AUTOSCALER_INTERVAL_S", 1.0)
# [P §5] "arrived at a 2 seconds polling interval" (VM-Autoscaler)
VM_AUTOSCALER_INTERVAL_S = _f("VM_AUTOSCALER_INTERVAL_S", 2.0)
# [P §5] "Every 2 seconds, the monitoring daemon updates ..." (code: 1 s / 2.5 s)
MONITOR_INTERVAL_S = _f("MONITOR_INTERVAL_S", 2.0)
# [N] SM utilization samples per monitor window (NVML reports a short moving average)
UTIL_SAMPLE_S = _f("UTIL_SAMPLE_S", 0.25)

# ---------------------------------------------------------------- scaling
# [P §5] "To tune slack-threshold ... and set it to 1.05."
SLACK_THRESHOLD = _f("SLACK_THRESHOLD", 1.05)
# [P §4.2.1] Cost(δ) = C(δ + λ·T_load·max(δ,0)); λ is "a tunable parameter". [U G2] 1.0 (1/s)
LAMBDA = _f("LAMBDA", 1.0)
# [P §6.2] instance cost "proportional to its memory footprint" -> C_ij = HW_COST * GB
COST_MEMORY_FLOOR_GB = _f("COST_MEMORY_FLOOR_GB", 0.001)
# [C autoscaler.cc:49] GPU_MAX_REPLICAS = 1 per worker (paper gives no number)
GPU_MAX_REPLICAS = _i("GPU_MAX_REPLICAS", 1)
# [C autoscaler.cc:59] memorySlack: keep at least 1 GB free
MEMORY_SLACK_BYTES = _i("MEMORY_SLACK_BYTES", 1 << 30)
# [N] how many recent monitor windows the min-SLO for a downgrade check covers
MIN_SLO_WINDOWS = _i("MIN_SLO_WINDOWS", 10)
# [N] a pending placement for a variant blocks new requests for it this long
PENDING_TTL_S = _f("PENDING_TTL_S", 30.0)

# ---------------------------------------------------------------- states
# [C query_executor.cc:685-702] Interfered heuristic coefficients for GPU variants.
# Paper §4 defines Interfered ("higher inference latencies than the profiled
# values") without a factor, so the code's factor fills the gap.
INTERFERED_LAT_FACTOR = _f("INTERFERED_LAT_FACTOR", 1.5)       # profiled >= 10 ms
INTERFERED_SMALL_LAT_MS = _f("INTERFERED_SMALL_LAT_MS", 10.0)   # below this ...
INTERFERED_SMALL_NUM = _f("INTERFERED_SMALL_NUM", 15.0)         # ... factor = min(5, 15/lat)
INTERFERED_SMALL_CAP = _f("INTERFERED_SMALL_CAP", 5.0)
INTERFERED_QPS_FRACTION = _f("INTERFERED_QPS_FRACTION", 0.3)    # blist_qps_heuristic (GPU)
INTERFERED_UNSET_FACTOR = _f("INTERFERED_UNSET_FACTOR", 1.25)   # blist_lat_unset_heuristic (GPU)
# [C query_executor.cc:627] with no completions in a window, avg latency /= 1.5 (GPU)
AVGLAT_DECAY = _f("AVGLAT_DECAY", 1.5)

# ---------------------------------------------------------------- VM(worker) autoscaler
# [P §4.2.3] "We empirically set the threshold to 80%"; utilization = SM util [U G3]
VM_UTIL_THRESHOLD = _f("VM_UTIL_THRESHOLD", 80.0)
# [P §4.2.3] rule 3: "more than 80% of workers have Overloaded variants"
VM_OVERLOADED_FRACTION = _f("VM_OVERLOADED_FRACTION", 0.8)
# [C master_vm_daemon.cc:62,312] backoff after a scale action: 15 iterations
VM_BACKOFF_ITERS = _i("VM_BACKOFF_ITERS", 15)
# [C master_vm_daemon.cc:71,523-545] scale-down after 15 low-utilization iterations
VM_SHUTDOWN_ITERS = _i("VM_SHUTDOWN_ITERS", 15)
VM_SHUTDOWN_GPU_UTIL = _f("VM_SHUTDOWN_GPU_UTIL", 8.0)          # gpu_shutdown_min
VM_SHUTDOWN_CPU_UTIL = _f("VM_SHUTDOWN_CPU_UTIL", 5.0)          # cpu_shutdown_min
VM_SHUTDOWN_AVG_GPU_UTIL = _f("VM_SHUTDOWN_AVG_GPU_UTIL", 8.0)  # avg_gpu_shutdown_min
VM_SHUTDOWN_AVG_CPU_UTIL = _f("VM_SHUTDOWN_AVG_CPU_UTIL", 30.0) # avg_cpu_shutdown_min

# Code-only executor blacklists (no counterpart in the paper, which handles
# overload through the Overloaded state). Off by default; see DEVIATIONS D-22.
# [C master_vm_daemon.cc:228] util > 80 -> blacklist the worker for 2 s
EXEC_BLACKLIST_UTIL_ENABLED = _b("EXEC_BLACKLIST_UTIL_ENABLED", False)
EXEC_BLACKLIST_UTIL = _f("EXEC_BLACKLIST_UTIL", 80.0)
# [C queryfe_server.cc:62-63,1606-1657] >= 200 requests within 1 s -> blacklist 2 s
EXEC_BLACKLIST_QPS_ENABLED = _b("EXEC_BLACKLIST_QPS_ENABLED", False)
EXEC_BLACKLIST_QPS_LIMIT = _i("EXEC_BLACKLIST_QPS_LIMIT", 200)
EXEC_BLACKLIST_EXPIRE_S = _i("EXEC_BLACKLIST_EXPIRE_S", 2)

# ---------------------------------------------------------------- worker modes
# [U] "static" (default): every GPU has a worker, like Lumina (worker Deployments).
#     "dynamic": the VM-Autoscaler creates/deletes worker pods (paper §4.2.3).
WORKER_MODE = _s("WORKER_MODE", "static").lower()
# dynamic mode: workers created at start (also the floor for scale-down) and ceiling
DYN_INIT_WORKERS: Dict[str, int] = {k: int(v) for k, v in
                                    _hw_map("DYN_INIT_WORKERS", "2080ti:1,a5000:1,a30:1").items()}
MAX_WORKERS: Dict[str, int] = {k: int(v) for k, v in
                               _hw_map("MAX_WORKERS", "2080ti:2,a5000:2,a30:2").items()}

# ---------------------------------------------------------------- dispatcher
# [C queryfe_server.cc:71-80, start_infaas.sh:21] decision mode; default 6 [U C7]
#   0 INFAAS_ALL, 1 INFAAS_NOQPSLAT, 2 ROUNDROBIN, 6 GPUSHARETRIGGER_SKIPBLIST
DECISION_MODE = _i("DECISION_MODE", 6)
# [C] warmup queries after a load (common_model_util.cc WARMUP_QUERIES)
WARMUP_QUERIES = _i("WARMUP_QUERIES", 10)
# [N] seconds the static variant registry is cached in the controller
REGISTRY_CACHE_S = _f("REGISTRY_CACHE_S", 5.0)
# [N] seconds the executor list is cached in the controller
EXECUTOR_CACHE_S = _f("EXECUTOR_CACHE_S", 1.0)

# ---------------------------------------------------------------- profiler
# [C profile_model.sh] latency = average of repeated single-concurrency runs
PROFILE_RUNS = _i("PROFILE_RUNS", 30)
# [N] saturation throughput Q_ij (paper Table 2 / §4.2.1) is measured closed-loop
PROFILE_SAT_SECONDS = _f("PROFILE_SAT_SECONDS", 5.0)
PROFILE_SAT_CONCURRENCY = _i("PROFILE_SAT_CONCURRENCY", 4)
PROFILE_DIR = _s("PROFILE_DIR", "/data/infaas-profiles")

# ---------------------------------------------------------------- endpoints
NAMESPACE = _s("INFAAS_NAMESPACE", "infaas")
REDIS_HOST = _s("REDIS_HOST", "localhost")
REDIS_PORT = _i("REDIS_PORT", 16379)
QUERYFE_PORT = _i("QUERYFE_PORT", 50052)      # [C constants.h queryfe_port]
MODELREG_PORT = _i("MODELREG_PORT", 50053)    # [C constants.h modelreg_port]
PLACEMENT_PORT = _i("PLACEMENT_PORT", 50054)  # [N]
COMPAT_PORT = _i("COMPAT_PORT", 8081)         # [N] Lumina-compatible podexec.Executor
WORKER_PORT = _i("WORKER_PORT", 9000)
CONTROLLER_ADDR = _s("CONTROLLER_ADDR", f"localhost:{PLACEMENT_PORT}")
FRONTEND_THREADS = _i("FRONTEND_THREADS", 64)  # same as Lumina's SERVER_GRPC_THREADS
WORKER_THREADS = _i("WORKER_THREADS", 32)
QUERY_TIMEOUT_S = _f("QUERY_TIMEOUT_S", 120.0)  # Lumina WORKER_RUN_TIMEOUT
LOAD_TIMEOUT_S = _f("LOAD_TIMEOUT_S", 300.0)
REQUEST_LOG = _s("REQUEST_LOG", "")            # JSON-lines per request, "" = off
EVENT_LOG = _s("EVENT_LOG", "")                # JSON-lines scaling/state events

WORKER_LABEL = "app=infaas-worker"

# ---------------------------------------------------------------- orchestrator
# [N] Where workers come from (the original: EC2 VMs).
#   k8s     worker pods found through the Kubernetes API (k8s/*.yaml)
#   static  a fixed list in STATIC_WORKERS, checked by gRPC heartbeat — docker on
#           one or more hosts without a cluster manager (deploy/docker/)
ORCHESTRATOR = _s("ORCHESTRATOR", "k8s").lower()
# "name=gpu_type@host:port,..." — written by deploy/docker/gen_compose.py
STATIC_WORKERS = _s("STATIC_WORKERS", "")
# consecutive failed heartbeats before a static worker counts as down
STATIC_FAIL_THRESHOLD = _i("STATIC_FAIL_THRESHOLD", 3)
