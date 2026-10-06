"""Variant-Profiler (paper §3.2), run on a worker of the variant's GPU type.

> "the Variant-Profiler conducts one-time profiling for each variant where it
> measures statistics, such as the loading and inference latencies, and peak
> memory utilization."

Measured the way the original profile_model.sh does, adapted to this runtime:
  load latency     start of the load until the instance can serve
                   ([C profile_model.sh:254-266] copy -> MODEL_READY; here the
                   warmup queries are included, as the worker does them before
                   serving)
  inference lat.   average of PROFILE_RUNS sequential requests through the full
                   worker path, preprocessing included [U C6]
                   ([C profile_model.sh] perf client "Avg latency", concurrency 1)
  peak memory      GPU memory in use after load and the runs, minus before
                   ([C profile_model.sh:257-273] nvidia-smi delta)
  saturation QPS   closed-loop, PROFILE_SAT_CONCURRENCY clients for
                   PROFILE_SAT_SECONDS — the Q_ij of paper §4.2.1 / Table 2 [N]

The variant is loaded outside the Metadata Store, so the Dispatcher never sees it.
"""
from __future__ import annotations

import io
import statistics
import threading
import time
from typing import Dict, List

import torch
from PIL import Image

from infaas.common import config
from infaas.common.naming import variant_name
from infaas.vendor.lumina import models
from infaas.worker.monitor import GpuSampler
from infaas.worker.runtime import Runtime


def _used(gpu: GpuSampler) -> int:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem = gpu.memory()
    if mem is not None:
        return mem[0]
    return int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0


def profile(runtime: Runtime, gpu: GpuSampler, model: str, image: bytes,
            runs: int = 0, sat_seconds: float = 0.0, sat_concurrency: int = 0) -> dict:
    runs = runs or config.PROFILE_RUNS
    sat_seconds = sat_seconds or config.PROFILE_SAT_SECONDS
    sat_concurrency = sat_concurrency or config.PROFILE_SAT_CONCURRENCY
    model_id = models.resolve(model)
    if model_id is None:
        raise ValueError(f"unsupported model: {model}")
    v = variant_name(model, runtime.hw)
    if runtime.get(v) is not None or runtime.is_loading(v):
        raise RuntimeError(f"{v} is loaded on this worker; unload it before profiling")

    mem0 = _used(gpu)
    inst, load_ms, _ = runtime.load(v)
    try:
        lats: List[float] = []
        stages: Dict[str, List[float]] = {}
        for _ in range(runs):
            t0 = time.perf_counter()
            _, _, _, tm = inst.infer(image)
            lats.append((time.perf_counter() - t0) * 1000.0)
            for k, val in tm.items():
                stages.setdefault(k, []).append(val)

        stop = time.perf_counter() + sat_seconds
        done = [0] * sat_concurrency

        def client(i: int) -> None:
            while time.perf_counter() < stop:
                inst.infer(image)
                done[i] += 1

        t_sat = time.perf_counter()
        threads = [threading.Thread(target=client, args=(i,)) for i in range(sat_concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        sat_qps = sum(done) / (time.perf_counter() - t_sat)
        mem1 = _used(gpu)
        dali = bool(inst.dali_cfg is not None)
        dali_note = inst.dali_note
    finally:
        runtime.unload(v)

    with Image.open(io.BytesIO(image)) as im:
        w, h = im.size
    lats_sorted = sorted(lats)
    return {
        "variant": v,
        "model": model,
        "hf_id": model_id,
        "hardware": runtime.hw,
        "gpu_name": gpu.name,
        "framework": "pytorch",
        "precision": "fp32",
        "max_batch": 1,
        "task": models.category_of(model_id) or "",
        "load_latency_ms": round(load_ms, 3),
        "inf_latency_ms": round(statistics.fmean(lats), 3),
        "inf_latency_p50_ms": round(lats_sorted[len(lats_sorted) // 2], 3),
        "inf_latency_p95_ms": round(lats_sorted[min(len(lats_sorted) - 1,
                                                    int(0.95 * len(lats_sorted)))], 3),
        "stage_ms": {k: round(statistics.fmean(vals), 3) for k, vals in stages.items()},
        "sat_qps": round(sat_qps, 3),
        "sat_concurrency": sat_concurrency,
        "sat_seconds": sat_seconds,
        "peak_memory_bytes": max(0, mem1 - mem0),
        "runs": runs,
        "warmup_queries": config.WARMUP_QUERIES,
        "image_size": [w, h],
        "image_bytes": len(image),
        "dali": dali,
        "dali_note": dali_note,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
