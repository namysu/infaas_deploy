"""GPU Hardware Executor (paper §3.2) — stands in for the original's Triton 19.03.

One `Instance` per loaded model-variant. Like a Triton model instance it has its
own execution thread and its own CUDA stream, so different variants on the
same GPU run concurrently and can interfere with each other (paper §2.3, Fig. 1)
— the situation INFaaS' Interfered state exists to detect. GPU_MAX_REPLICAS = 1
per worker [C autoscaler.cc:49], so there is at most one instance per variant.

Loading is from disk, not from a RAM cache: the original copies the variant from
its local model directory into Triton's repository and waits for MODEL_READY
[C common_model_util.cc:632-809], and unloading drops it again. Here: weights
from the node's HF cache (same files as Lumina) -> GPU, then WARMUP_QUERIES
warmup queries [C common_model_util.cc WARMUP_QUERIES]; unloading frees the GPU
memory.

Preprocessing is Lumina's path, vendored unchanged (JPEG in, DALI or PIL), so
both systems execute the same work per request [PLAN A-04].
"""
from __future__ import annotations

import gc
import io
import logging
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from infaas.common import config
from infaas.common.naming import parse_variant
from infaas.vendor.lumina import dali_preprocess, models

log = logging.getLogger("worker.runtime")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_THRESHOLD = float(os.environ.get("DET_SCORE_THRESHOLD", "0.5"))  # Lumina default

Box = Tuple[float, float, float, float, str, float]


def _autoclass(name: str):
    import transformers
    return getattr(transformers, name)


def _get_processor(model_id: str):
    # same as Lumina podexec/worker/server.py _get_processor
    from transformers import AutoImageProcessor, AutoProcessor
    try:
        return AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        return AutoProcessor.from_pretrained(model_id)


def _decode(out, model, processor, image_size: Tuple[int, int]) -> Tuple[str, List[Box]]:
    """Summary label + boxes. Same logic as Lumina podexec/worker/server.py _decode."""
    id2label = getattr(model.config, "id2label", None) or {}
    if hasattr(out, "pred_boxes") and hasattr(processor, "post_process_object_detection"):
        w, h = image_size
        res = processor.post_process_object_detection(
            out, target_sizes=torch.tensor([[h, w]]), threshold=DET_THRESHOLD)[0]
        boxes: List[Box] = []
        for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            boxes.append((x1, y1, x2, y2, str(id2label.get(int(label_id), int(label_id))),
                          round(float(score), 4)))
        return f"{len(boxes)} objects", boxes
    logits = getattr(out, "logits", None)
    if logits is not None and logits.ndim == 2:
        idx = int(logits.argmax(-1).item())
        return str(id2label.get(idx, idx)), []
    if logits is not None and logits.ndim == 4:
        return "<segmentation>", []
    return "<features>", []


class Instance:
    """A loaded model-variant on this worker's GPU."""

    def __init__(self, variant: str, model_id: str, model, processor,
                 dali_cfg, dali_note: str) -> None:
        self.variant = variant
        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.dali_cfg = dali_cfg
        self.dali_note = dali_note
        self.lock = threading.Lock()          # held while a forward pass runs
        self.stream = torch.cuda.Stream() if DEVICE == "cuda" else None
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self.loaded_at = time.time()
        self.ready_ms = 0.0                   # load time up to "ready", set by Runtime._build
        # Every forward pass runs on this one thread (a Triton model instance has
        # its own execution thread too). cuDNN keeps handles and execution-plan
        # caches per calling thread; forwards arriving from different gRPC
        # threads made ConvNeXt rebuild its plans on nearly every call (7 ms ->
        # 2.2 s per forward, measured on the 2080 Ti). One thread, warmed at load,
        # pays that once.
        self._jobs: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._exec_loop, daemon=True, name=f"exec:{variant}").start()

    def _exec_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            inputs, box = job
            try:
                with self.lock:
                    t_locked = time.perf_counter()
                    with torch.inference_mode():
                        if self.stream is not None:
                            with torch.cuda.stream(self.stream):
                                out = self.model(**{k: v.to(DEVICE) for k, v in inputs.items()})
                            self.stream.synchronize()
                        else:
                            out = self.model(**{k: v.to(DEVICE) for k, v in inputs.items()})
                    box["out"], box["t_locked"], box["t_fwd"] = out, t_locked, time.perf_counter()
            except BaseException as e:  # noqa: BLE001 — hand it to the caller
                box["err"] = e
            finally:
                box["done"].set()

    def close(self) -> None:
        self._jobs.put(None)

    @property
    def inflight(self) -> int:
        return self._inflight

    def infer(self, image_bytes: bytes) -> Tuple[str, List[Box], Tuple[int, int], Dict[str, float]]:
        with self._inflight_lock:
            self._inflight += 1
        try:
            return self._infer(image_bytes)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _forward(self, inputs: Dict[str, torch.Tensor]):
        """Run one forward pass on the instance's execution thread."""
        box = {"done": threading.Event()}
        self._jobs.put((inputs, box))
        box["done"].wait()
        if "err" in box:
            raise box["err"]
        return box["out"], box["t_locked"], box["t_fwd"]

    def forward_only(self, inputs: Dict[str, torch.Tensor]) -> float:
        """Forward pass on an already-preprocessed input; returns its ms.

        What the original profiler times: the model on a ready input tensor,
        with no image decode or preprocessing (PROFILE_MODE=original).
        """
        with self._inflight_lock:
            self._inflight += 1
        try:
            _, t_locked, t_fwd = self._forward(inputs)
            return (t_fwd - t_locked) * 1000.0
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _infer(self, image_bytes: bytes):
        t0 = time.perf_counter()
        if self.dali_cfg is not None:
            with Image.open(io.BytesIO(image_bytes)) as im:
                img_size = im.size
            t_decode = time.perf_counter()
            pixel_values, pre_wait = dali_preprocess.run_timed(self.dali_cfg, image_bytes)
            inputs = {"pixel_values": pixel_values}
        else:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_size = img.size
            t_decode = time.perf_counter()
            inputs = dict(self.processor(images=img, return_tensors="pt"))
            pre_wait = 0.0
        t_pre = time.perf_counter()
        out, t_locked, t_fwd = self._forward(inputs)
        label, boxes = _decode(out, self.model, self.processor, img_size)
        t_end = time.perf_counter()
        ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
        timings = {
            "decode": ms(t0, t_decode),
            "preprocess": ms(t_decode, t_pre),
            "pre_wait": pre_wait,
            "lock_wait": ms(t_pre, t_locked),
            "forward": ms(t_locked, t_fwd),
            "postprocess": ms(t_fwd, t_end),
            "total": ms(t0, t_end),
            "dali": 1.0 if self.dali_cfg is not None else 0.0,
        }
        return label, boxes, img_size, timings


class Runtime:
    """The set of instances on this worker, with load/unload."""

    def __init__(self, hw: str) -> None:
        self.hw = hw
        self._instances: Dict[str, Instance] = {}
        self._loading: Dict[str, threading.Event] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._probe = dali_preprocess.textured_jpeg() if dali_preprocess.available() else None
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), color=(127, 127, 127)).save(buf, format="JPEG")
        self._fallback_jpeg = buf.getvalue()

    def warm_cuda(self) -> None:
        if DEVICE == "cuda":
            x = torch.randn(1, 3, 224, 224, device=DEVICE)
            torch.cuda.synchronize()
            del x

    def get(self, variant: str) -> Optional[Instance]:
        return self._instances.get(variant)

    def loaded(self) -> List[str]:
        with self._lock:
            return sorted(self._instances)

    def is_loading(self, variant: str) -> bool:
        with self._lock:
            return variant in self._loading

    def load(self, variant: str) -> Tuple[Instance, float, bool]:
        """Make `variant` resident. Returns (instance, load_ms, this call loaded it).

        Concurrent callers for the same variant wait for the one load.
        """
        model_name, hw = parse_variant(variant)
        if hw != self.hw:
            raise ValueError(f"{variant} is for {hw}, this worker is {self.hw}")
        model_id = models.resolve(model_name)
        if model_id is None:
            raise ValueError(f"unsupported model: {model_name}")
        with self._lock:
            inst = self._instances.get(variant)
            if inst is not None:
                return inst, 0.0, False
            ev = self._loading.get(variant)
            owner = ev is None
            if owner:
                ev = self._loading[variant] = threading.Event()
        if not owner:
            ev.wait(config.LOAD_TIMEOUT_S)
            inst = self._instances.get(variant)
            if inst is None:
                raise RuntimeError(f"load of {variant} failed: {self._errors.get(variant, 'timeout')}")
            return inst, 0.0, False
        t0 = time.perf_counter()
        try:
            inst = self._build(variant, model_id)
            load_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._instances[variant] = inst
            self._errors.pop(variant, None)
            log.info("loaded %s in %.0f ms [%s]", variant, load_ms, inst.dali_note)
            return inst, load_ms, True
        except Exception as e:
            self._errors[variant] = str(e)
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            raise
        finally:
            with self._lock:
                self._loading.pop(variant, None)
            ev.set()

    def _build(self, variant: str, model_id: str) -> Instance:
        t0 = time.perf_counter()
        cls = _autoclass(models.MODEL_AUTOCLASS[model_id])
        model = cls.from_pretrained(model_id).eval()
        processor = _get_processor(model_id)
        model.to(DEVICE)
        dali_cfg, note = None, "PIL"
        if self._probe is not None:
            dali_cfg, note = dali_preprocess.evaluate(model_id, processor, *self._probe)
            note = ("DALI " if dali_cfg is not None else "PIL: ") + note
        inst = Instance(variant, model_id, model, processor, dali_cfg, note)
        # ready to run, before the warmup queries: the original profiler's load
        # latency stops here (copy -> MODEL_READY, [C profile_model.sh:254-266])
        inst.ready_ms = (time.perf_counter() - t0) * 1000.0
        warm = self._probe[0] if self._probe is not None else self._fallback_jpeg
        for _ in range(config.WARMUP_QUERIES):
            inst.infer(warm)
        return inst

    def unload(self, variant: str, drain_timeout_s: float = 5.0) -> bool:
        with self._lock:
            inst = self._instances.pop(variant, None)
        if inst is None:
            return False
        deadline = time.time() + drain_timeout_s
        while inst.inflight > 0 and time.time() < deadline:
            time.sleep(0.01)
        with inst.lock:          # no forward pass is running past this point
            inst.model = None
        inst.close()
        del inst
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        log.info("unloaded %s", variant)
        return True
