"""DALI preprocessing, so the worker runs what the latency predictor models.

The preprocess predictor was trained on DALI pipelines — its modes are literally
named `dali_cpu`, `dali_cpu_decode_gpu`, `dali_mixed_decode_sw/hw`. The worker,
however, ran PIL decode + a HuggingFace image processor, which measured 2.4-3x
slower than the prediction. That gap is not predictor error; it is the worker
doing something the predictor was never asked about. Running DALI closes it.

Which mode: for a JPEG on a GPU without the fixed-function NVJPG engine the
predictor only offers `cpu_only` and `cpu_decode_gpu`, and picks the latter
(13.75 ms vs 17.24 ms on this cluster). 2080 Ti (Turing) and A5000 (GA102) are
both in that class, so `cpu_decode_gpu` is the mode to match: libjpeg decode on
the CPU, then resize and normalize on the GPU. The hardware-decode modes stay
out of reach until the cluster has an A30/L40S-class card.

A DALI pipeline is a graph built ahead of time, while a HuggingFace processor is
per-model Python. The bridge is one pipeline per *distinct preprocessing config*
(resize / crop / mean / std / interpolation) rather than per model — 43 models
collapse to a handful of configs, most of them 224x224 with ImageNet statistics.

Nothing here is trusted blindly. `evaluate()` runs both paths on the same
textured JPEG at startup and only clears a model for DALI if the tensors agree;
anything unexpected (extra processor outputs, an unmappable resize rule, a
resampling filter DALI does not have) falls back to the PIL path for that model
alone. Correct results matter more than matching the predictor.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("worker.dali")

# "auto" uses DALI when it imports, else PIL. "dali"/"pil" force one path, which
# is what an A/B run of the two backends needs.
BACKEND = os.environ.get("PREPROCESS_BACKEND", "auto").lower()
# Which predictor mode to reproduce, set per GPU type on the Deployment. The
# right mode is a property of the hardware, not of the request — the predictor
# picks it from the GPU and the (fixed) JPEG format alone — so pinning it per
# node is enough to keep prediction and execution describing the same thing,
# with no need to carry a mode through the request path.
#
#   cpu_only        decode on CPU, resize/normalize on CPU
#   cpu_decode_gpu  decode on CPU, resize/normalize on GPU
#   mixed_decode    nvJPEG on the GPU's shader cores (any CUDA GPU)
#   gpu_only        nvJPEG on the fixed-function NVJPG engine
#
# The last two are what the predictor offers only for A30 and L40S, the GPUs
# that carry that engine. On one without it, "gpu_only" still runs — DALI falls
# back to the shader path — but it then competes with inference for SMs and the
# predictor has no figure for it, so those nodes stay on cpu_decode_gpu.
MODE = os.environ.get("PREPROCESS_MODE", "cpu_decode_gpu").lower()
# (decode device, hw_decoder_load, run the resize/normalize stage on the GPU)
_MODE_SPEC = {
    "cpu_only":       ("cpu", None, False),
    "cpu_decode_gpu": ("cpu", None, True),
    "mixed_decode":   ("mixed", 0.0, True),
    "gpu_only":       ("mixed", 1.0, True),
}
if MODE not in _MODE_SPEC:
    log.warning("unknown PREPROCESS_MODE=%r; falling back to cpu_decode_gpu", MODE)
    MODE = "cpu_decode_gpu"
DECODE_DEVICE, HW_DECODER_LOAD, GPU_STAGE = _MODE_SPEC[MODE]
NUM_THREADS = int(os.environ.get("DALI_NUM_THREADS", "2"))
# Mean absolute difference (in normalized tensor units) tolerated between the
# DALI and PIL tensors. Interpolation differs slightly between libjpeg/DALI and
# PIL, so this is not zero; it is tight enough to catch a wrong size, a missing
# crop or mis-scaled normalization statistics, which shift the mean far more.
MAX_MEAN_DIFF = float(os.environ.get("DALI_MAX_MEAN_DIFF", "0.05"))

_dali = None          # module handle once imported
_import_error: Optional[str] = None


def _try_import() -> bool:
    """Import DALI once. Returns whether it is usable."""
    global _dali, _import_error
    if _dali is not None:
        return True
    if _import_error is not None or BACKEND == "pil":
        return False
    try:
        from nvidia.dali import fn, pipeline_def, types  # noqa: F401
        from nvidia.dali.plugin.pytorch import feed_ndarray  # noqa: F401
        import nvidia.dali as dali_mod
        _dali = {"fn": fn, "types": types, "pipeline_def": pipeline_def,
                 "feed_ndarray": feed_ndarray, "version": dali_mod.__version__}
        log.info("DALI %s available; mode=%s", dali_mod.__version__, MODE)
        return True
    except Exception as e:  # noqa: BLE001
        _import_error = str(e)
        if BACKEND == "dali":
            log.error("PREPROCESS_BACKEND=dali but DALI is unusable: %s", e)
        else:
            log.info("DALI unavailable (%s); using PIL preprocessing", e)
        return False


def available() -> bool:
    return _try_import()


def status() -> str:
    if BACKEND == "pil":
        return "pil (forced)"
    if not _try_import():
        return f"pil (DALI unavailable: {_import_error})"
    return (f"dali {_dali['version']} mode={MODE} "
            f"(decode={DECODE_DEVICE}"
            + (f" hw={HW_DECODER_LOAD:g}" if HW_DECODER_LOAD is not None else "")
            + f", gpu_stage={GPU_STAGE})")


# --- preprocessing config -------------------------------------------------
# PIL resampling filter -> the closest DALI interpolation. DALI antialiases when
# downscaling, which is what PIL does too, so the pair stays comparable.
_INTERP_NAMES = {0: "INTERP_NN", 1: "INTERP_LANCZOS3", 2: "INTERP_LINEAR",
                 3: "INTERP_CUBIC", 4: "INTERP_TRIANGULAR", 5: "INTERP_LINEAR"}


@dataclass(frozen=True)
class Config:
    """A resize/crop/normalize recipe, hashable so it can key a pipeline cache."""
    resize_shorter: Optional[float]   # resize by shorter side, then crop
    resize_x: Optional[int]           # or resize to exact dimensions
    resize_y: Optional[int]
    crop_h: Optional[int]
    crop_w: Optional[int]
    mean: Tuple[float, ...]           # in decoded 0-255 units
    std: Tuple[float, ...]
    interp: str
    gpu_stage: bool                   # resize/normalize (and output) on the GPU
    bgr: bool                         # decode straight to BGR (MobileViT)


def _as_mapping(v) -> Optional[dict]:
    """Normalize a processor's size/crop_size field to a plain dict.

    transformers wraps these in a SizeDict dataclass rather than a dict, and it
    carries every possible key with None for the unused ones — so the Nones are
    dropped here, otherwise "shortest_edge" would appear to be set on a
    height/width processor.
    """
    if v is None:
        return None
    if isinstance(v, dict):
        items = v.items()
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        items = dataclasses.asdict(v).items()
    elif hasattr(v, "keys"):
        items = ((k, v[k]) for k in v.keys())
    else:
        return None
    return {k: val for k, val in items if val is not None}


def config_from_processor(proc) -> Tuple[Optional[Config], str]:
    """Translate a HuggingFace image processor into a Config.

    Returns (config, reason). A None config means this model keeps the PIL path,
    and `reason` says why — logged once at startup so the fallback is visible
    rather than silent.
    """
    # Steps this translation does not model. They are named rather than left to
    # the numeric gate so the startup log says what is unsupported instead of
    # just reporting a large difference. MobileViT flips RGB to BGR; detection
    # processors pad to a common size; segmentation ones remap label ids.
    for flag in ("do_pad", "do_reduce_labels"):
        if getattr(proc, flag, False):
            return None, f"{flag}=True is not modelled"

    # MobileViT checkpoints expect BGR and the processor reverses the channels
    # as its final step. DALI can simply decode to BGR instead, which is the
    # same thing done earlier — as long as the normalization statistics are
    # reversed to match, which happens once they are computed below.
    bgr = bool(getattr(proc, "do_flip_channel_order", False))

    size = _as_mapping(getattr(proc, "size", None))
    if not size:
        raw = getattr(proc, "size", None)
        return None, f"size is {type(raw).__name__}, not a usable mapping"

    resample = getattr(proc, "resample", 2)
    try:
        interp = _INTERP_NAMES[int(resample)]
    except Exception:  # noqa: BLE001
        return None, f"unmappable resample={resample!r}"

    resize_shorter = resize_x = resize_y = None
    crop_h = crop_w = None

    if "height" in size and "width" in size:
        resize_y, resize_x = int(size["height"]), int(size["width"])
    elif "shortest_edge" in size:
        if "longest_edge" in size:
            return None, "size has longest_edge (aspect-bounded resize)"
        shortest = float(size["shortest_edge"])
        # ConvNeXt resizes to shortest_edge/crop_pct and then crops back down.
        crop_pct = getattr(proc, "crop_pct", None)
        if crop_pct:
            resize_shorter = float(int(shortest / float(crop_pct)))
            crop_h = crop_w = int(shortest)
        else:
            resize_shorter = shortest
    else:
        return None, f"unrecognized size keys {sorted(size)}"

    if getattr(proc, "do_center_crop", False):
        cs = _as_mapping(getattr(proc, "crop_size", None))
        if not (cs and "height" in cs and "width" in cs):
            return None, ("do_center_crop with crop_size="
                          f"{getattr(proc, 'crop_size', None)!r}")
        crop_h, crop_w = int(cs["height"]), int(cs["width"])

    # A shorter-side resize leaves the long side aspect dependent, so DALI needs
    # a crop to land on a fixed shape. Without one the output size would vary
    # per input image and the model would reject it.
    if resize_shorter is not None and crop_h is None:
        return None, "shortest_edge resize with no center crop"

    # HF's value pipeline is a chain of affine steps on the decoded pixels,
    # while DALI's crop_mirror_normalize applies exactly one: (x - M) / S. Every
    # step below is affine, so composing them into a single (a, b) per channel
    # and solving a*x + b == (x - M)/S reproduces the chain exactly. Doing it
    # this way covers the variants that a hand-written formula keeps missing:
    # EfficientNet's rescale_offset (x*s - 1) and its include_top second
    # normalization pass, arbitrary rescale factors, do_normalize=False.
    a = np.ones(3, dtype=np.float64)   # v = a*x + b, x being the raw 0-255 pixel
    b = np.zeros(3, dtype=np.float64)

    if getattr(proc, "do_rescale", True):
        scale = float(getattr(proc, "rescale_factor", 1.0 / 255.0))
        if scale <= 0:
            return None, f"non-positive rescale_factor={scale}"
        a *= scale
        if getattr(proc, "rescale_offset", False):
            b -= 1.0        # EfficientNet maps into [-1, 1]

    def _triple(v, what):
        try:
            arr = np.broadcast_to(np.asarray(v, dtype=np.float64), (3,)).copy()
        except Exception:  # noqa: BLE001
            return None, f"{what}={v!r} is not broadcastable to 3 channels"
        return arr, "ok"

    std_arr = None
    if getattr(proc, "do_normalize", True):
        mean_v, std_v = getattr(proc, "image_mean", None), getattr(proc, "image_std", None)
        if mean_v is None or std_v is None:
            return None, "do_normalize without image_mean/image_std"
        mean_arr, why_m = _triple(mean_v, "image_mean")
        if mean_arr is None:
            return None, why_m
        std_arr, why_s = _triple(std_v, "image_std")
        if std_arr is None:
            return None, why_s
        if np.any(std_arr == 0):
            return None, "image_std contains 0"
        a /= std_arr
        b = (b - mean_arr) / std_arr

    # EfficientNet normalizes a second time with mean=0 and the same std.
    if getattr(proc, "include_top", False):
        if std_arr is None:
            return None, "include_top without do_normalize"
        a /= std_arr
        b /= std_arr

    if np.any(a == 0):
        return None, "degenerate scaling (a == 0)"
    # (x - M)/S == a*x + b  =>  S = 1/a,  M = -b/a
    std_255 = 1.0 / a
    mean_255 = -b / a

    # HF reverses channels after normalizing; DALI reverses them at decode, so
    # channel c of its output is source channel 2-c. Reversing the statistics
    # makes the two orders produce identical tensors.
    if bgr:
        std_255 = std_255[::-1]
        mean_255 = mean_255[::-1]

    return Config(
        resize_shorter=resize_shorter, resize_x=resize_x, resize_y=resize_y,
        crop_h=crop_h, crop_w=crop_w,
        mean=tuple(float(v) for v in mean_255),
        std=tuple(float(v) for v in std_255),
        interp=interp,
        gpu_stage=GPU_STAGE,
        bgr=bgr,
    ), "ok"


# --- pipeline cache -------------------------------------------------------
# One pipeline per Config, each with its own lock: a DALI pipeline is not
# thread-safe, and this worker serves several gRPC threads. Separate locks let
# two different configs preprocess concurrently, and none of them touch the
# `_gpu_lock` that serializes model swap + forward.
_pipelines: Dict[Config, Tuple[object, threading.Lock]] = {}
_pipelines_lock = threading.Lock()


def _build(cfg: Config):
    fn, types = _dali["fn"], _dali["types"]
    interp = getattr(types, cfg.interp)

    @_dali["pipeline_def"](
        batch_size=1, num_threads=NUM_THREADS, device_id=0,
        # One request in, one tensor out. DALI's pipelined async executor exists
        # to hide latency behind prefetching across iterations, which a
        # request/response server has nothing to prefetch for; synchronous
        # execution keeps run() returning exactly the batch just fed.
        exec_pipelined=False, exec_async=False)
    def pipe():
        jpegs = fn.external_source(name="jpegs", dtype=types.UINT8, ndim=1)
        # `mixed` decodes with nvJPEG and hands back a GPU tensor; hw_decoder_load
        # is the share routed to the fixed-function engine, so 1.0 is the
        # predictor's gpu_only and 0.0 its mixed_decode (shader path).
        dec_kwargs = {}
        if HW_DECODER_LOAD is not None:
            dec_kwargs["hw_decoder_load"] = HW_DECODER_LOAD
        images = fn.decoders.image(
            jpegs, device=DECODE_DEVICE,
            output_type=(types.BGR if cfg.bgr else types.RGB),
            **dec_kwargs)
        if cfg.gpu_stage and DECODE_DEVICE == "cpu":
            images = images.gpu()      # mixed already produced a GPU tensor
        # DALI antialiases when downscaling, as PIL does — except for nearest
        # neighbour, where the option does not apply and PIL does not either.
        aa = cfg.interp != "INTERP_NN"
        if cfg.resize_shorter is not None:
            images = fn.resize(images, resize_shorter=cfg.resize_shorter,
                               interp_type=interp, antialias=aa)
        else:
            images = fn.resize(images, resize_x=cfg.resize_x,
                               resize_y=cfg.resize_y,
                               interp_type=interp, antialias=aa)
        # Center crop, scale, normalize and NHWC->NCHW in a single kernel.
        return fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, output_layout="CHW",
            crop=([cfg.crop_h, cfg.crop_w] if cfg.crop_h else None),
            mean=list(cfg.mean), std=list(cfg.std), mirror=0)

    p = pipe()
    p.build()
    return p


def pipeline_for(cfg: Config) -> Tuple[object, threading.Lock]:
    """Return the (pipeline, lock) for a config, building it on first use."""
    with _pipelines_lock:
        got = _pipelines.get(cfg)
        if got is not None:
            return got
    # Build outside the registry lock: it is slow, and two threads racing to
    # build the same config is wasteful but harmless.
    built = (_build(cfg), threading.Lock())
    with _pipelines_lock:
        return _pipelines.setdefault(cfg, built)


def run(cfg: Config, image_bytes: bytes) -> torch.Tensor:
    """Decode + preprocess one JPEG. Returns a (1,3,H,W) float32 tensor."""
    return run_timed(cfg, image_bytes)[0]


def run_timed(cfg: Config, image_bytes: bytes) -> Tuple[torch.Tensor, float]:
    """Same as `run`, plus how long this call waited for the pipeline lock.

    A DALI pipeline is not thread safe, so requests sharing a config serialize
    here. That wait is queueing, not work: it belongs with the scheduler's queue
    term, and being indistinguishable inside `preprocess` it was landing in the
    measured service time instead — so the same delay got counted twice, once
    there and once in the worker's predicted free time.

    With cpu_decode_gpu the tensor is already on the GPU, so the H2D copy that
    `inputs.to(DEVICE)` used to make disappears from the request path.
    """
    pipe, lock = pipeline_for(cfg)
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    t_want = time.perf_counter()
    with lock:
        wait_ms = (time.perf_counter() - t_want) * 1000.0
        pipe.feed_input("jpegs", [buf])
        (out,) = pipe.run()
        tensor = out.as_tensor()
        shape = tuple(tensor.shape())
        if cfg.gpu_stage:
            dst = torch.empty(shape, dtype=torch.float32, device="cuda")
            stream = torch.cuda.current_stream()
            _dali["feed_ndarray"](tensor, dst, cuda_stream=stream)
            stream.synchronize()
        else:
            dst = torch.empty(shape, dtype=torch.float32)
            _dali["feed_ndarray"](tensor, dst)
    return dst, wait_ms


# --- startup validation ---------------------------------------------------
def textured_jpeg(width: int = 1920, height: int = 1080) -> Tuple[bytes, object]:
    """A deterministic, detailed test image plus its PIL form.

    Detail is the point: a flat colour resizes identically under any filter, so
    it would validate nothing. This mixes gradients with fine checkerboard and
    pseudo-random noise so an interpolation or crop mismatch actually shows up.
    """
    import io
    from PIL import Image
    yy, xx = np.mgrid[0:height, 0:width]
    rng = np.random.default_rng(0)
    checker = (((xx // 8) + (yy // 8)) % 2) * 40
    img = np.stack([
        (xx * 255 // max(width - 1, 1)),
        (yy * 255 // max(height - 1, 1)),
        checker + rng.integers(0, 60, size=(height, width)),
    ], axis=-1).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    # Re-open the encoded bytes so both paths see identical (lossy) pixels.
    return data, Image.open(io.BytesIO(data)).convert("RGB")


def evaluate(model_id: str, processor, jpeg: bytes, pil_img) -> Tuple[Optional[Config], str]:
    """Decide whether `model_id` may use DALI, by comparing against PIL output.

    Returns (config or None, human-readable reason). Called once per model at
    startup, so the request path never pays for this and never guesses.
    """
    if not _try_import():
        return None, "DALI unavailable"

    ref = processor(images=pil_img, return_tensors="pt")
    keys = set(ref.keys())
    # DALI produces pixel_values only. A processor that also emits a pixel_mask
    # or similar (some detection models) carries information we would drop.
    if keys != {"pixel_values"}:
        return None, f"processor returns {sorted(keys)}, not just pixel_values"

    cfg, why = config_from_processor(processor)
    if cfg is None:
        return None, why

    try:
        got = run(cfg, jpeg)
    except Exception as e:  # noqa: BLE001
        return None, f"pipeline failed: {e}"

    want = ref["pixel_values"]
    if tuple(got.shape) != tuple(want.shape):
        return None, f"shape {tuple(got.shape)} != PIL {tuple(want.shape)}"

    diff = (got.detach().float().cpu() - want.float()).abs()
    mean_d, max_d = float(diff.mean()), float(diff.max())
    if mean_d > MAX_MEAN_DIFF:
        return None, f"mean|diff|={mean_d:.4f} > {MAX_MEAN_DIFF} (max {max_d:.3f})"
    return cfg, f"ok mean|diff|={mean_d:.4f} max={max_d:.3f}"
