"""Variant-Profiler (paper §3.2), run on a worker of the variant's GPU type.

> "the Variant-Profiler conducts one-time profiling for each variant where it
> measures statistics, such as the loading and inference latencies, and peak
> memory utilization."

Two modes (PROFILE_MODE, infaas/common/config.py):

original — as the released profile_model.sh, no image involved:
  load latency     start of the load until the model is ready, warmup excluded
                   ([C profile_model.sh:254-266] copy -> MODEL_READY)
  inference lat.   average of PROFILE_RUNS forward passes on an input tensor of
                   the model's input size filled with random values
                   ([C trtis_perf_client.cc:709-742] random input, "Avg latency")
  saturation QPS   1000 / latency × batch ([C autoscaler.cc:206] single_throughput)

service — the worker's whole request path on the given JPEG [U C6]:
  load latency     until the instance can serve, warmup included
  inference lat.   average of PROFILE_RUNS requests, decode + preprocess included
  saturation QPS   closed-loop, PROFILE_SAT_CONCURRENCY clients for PROFILE_SAT_SECONDS

Both: peak memory = GPU memory in use after load and runs, minus before
([C profile_model.sh:257-273] nvidia-smi delta). The variant is loaded outside the
Metadata Store, so the Dispatcher never sees it.
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
from infaas.worker.inputs import nominal_size
from infaas.worker.monitor import GpuSampler
from infaas.worker.runtime import Runtime


def _used(gpu: GpuSampler) -> int:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem = gpu.memory()
    if mem is not None:
        return mem[0]
    return int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0


def synthetic_inputs(processor) -> Dict[str, torch.Tensor]:
    """The model's input at its nominal size, pixel values random (like perf_client)."""
    w, h = nominal_size(processor)
    dummy = Image.new("RGB", (w, h), color=(127, 127, 127))
    inputs = dict(processor(images=dummy, return_tensors="pt"))
    inputs["pixel_values"] = torch.randn_like(inputs["pixel_values"])
    return inputs


def _summary(lats: List[float]) -> dict:
    s = sorted(lats)
    return {"inf_latency_ms": round(statistics.fmean(lats), 3),
            "inf_latency_p50_ms": round(s[len(s) // 2], 3),
            "inf_latency_p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3)}


def profile(runtime: Runtime, gpu: GpuSampler, model: str, image: bytes,
            runs: int = 0, sat_seconds: float = 0.0, sat_concurrency: int = 0,
            mode: str = "") -> dict:
    mode = (mode or config.PROFILE_MODE).lower()
    if mode not in ("original", "service"):
        raise ValueError(f"PROFILE_MODE={mode!r}: original | service")
    if mode == "service" and not image:
        raise ValueError("PROFILE_MODE=service measures the request path and needs an image")
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
    extra: dict = {}
    try:
        if mode == "original":
            inputs = synthetic_inputs(inst.processor)
            lats = [inst.forward_only(inputs) for _ in range(runs)]
            summary = _summary(lats)
            load_latency = inst.ready_ms or load_ms
            sat_qps = 1000.0 / summary["inf_latency_ms"] * 1     # batch 1
            extra = {"input_shape": list(inputs["pixel_values"].shape),
                     "load_with_warmup_ms": round(load_ms, 3)}
        else:
            lats, stages = [], {}
            for _ in range(runs):
                t0 = time.perf_counter()
                _, _, _, tm = inst.infer(image)
                lats.append((time.perf_counter() - t0) * 1000.0)
                for k, val in tm.items():
                    stages.setdefault(k, []).append(val)
            summary = _summary(lats)
            load_latency = load_ms
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
            with Image.open(io.BytesIO(image)) as im:
                iw, ih = im.size
            extra = {"stage_ms": {k: round(statistics.fmean(x), 3) for k, x in stages.items()},
                     "sat_concurrency": sat_concurrency, "sat_seconds": sat_seconds,
                     "image_size": [iw, ih], "image_bytes": len(image)}
        mem1 = _used(gpu)
        dali, dali_note = bool(inst.dali_cfg is not None), inst.dali_note
    finally:
        runtime.unload(v)

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
        "profile_mode": mode,
        "load_latency_ms": round(load_latency, 3),
        **summary,
        "sat_qps": round(sat_qps, 3),
        "peak_memory_bytes": max(0, mem1 - mem0),
        "runs": runs,
        "warmup_queries": config.WARMUP_QUERIES,
        "dali": dali,
        "dali_note": dali_note,
        **extra,
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
