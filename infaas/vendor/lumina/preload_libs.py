"""Base + cross-model common libraries to pre-import at worker startup.

The lists are derived from the dependency profiling in
`workload_dependency/run_third_05262046.log`:

  * BASE_LIBS  — "Base Common Libraries" (loaded by `import torch` +
    `import transformers`), i.e. paid once regardless of which model is served.
  * CROSS_LIBS — "Cross-Model Common Libraries" (loaded by `from_pretrained` in
    >=80% of the 42 models).

Pure-stdlib modules and the profiling harness's own imports (cProfile, pdb,
timeit, unittest, ...) are intentionally excluded; only torch/transformers and
their real third-party dependencies are kept.

Pre-importing these in the long-lived worker process moves their import cost out
of the request path — so a request only pays model load (first time per model) +
preprocess + inference, never library import time.
"""
from __future__ import annotations

import importlib
import logging
import time
from typing import List

log = logging.getLogger("worker.preload")

# "Base Common" — torch/transformers core + real third-party deps (68)
BASE_LIBS: List[str] = [
    "torch",
    "transformers",
    "transformers.utils",
    "transformers.models",
    "sympy",
    "numpy",
    "torchvision",
    "huggingface_hub",
    "triton",
    "transformers.processing_utils",
    "rich",
    "accelerate",
    "transformers.modeling_outputs",
    "torchgen",
    "mpmath",
    "anyio",
    "jinja2",
    "yaml",
    "PIL",
    "httpx",
    "transformers.integrations",
    "transformers.generation",
    "h11",
    "psutil",
    "transformers.dependency_versions_check",
    "pygments",
    "optree",
    "regex",
    "httpcore",
    "click",
    "transformers.quantizers",
    "tokenizers",
    "tqdm",
    "transformers.models.auto",
    "transformers.modeling_utils",
    "filelock",
    "transformers.loss",
    "packaging",
    "idna",
    "typing_extensions",
    "transformers.image_utils",
    "transformers.tokenization_utils_base",
    "transformers.configuration_utils",
    "transformers.modeling_gguf_pytorch_utils",
    "transformers.video_utils",
    "transformers.convert_slow_tokenizer",
    "safetensors",
    "transformers.masking_utils",
    "transformers.core_model_loading",
    "transformers.cache_utils",
    "transformers.models.encoder_decoder",
    "transformers.distributed",
    "transformers.image_transforms",
    "markupsafe",
    "transformers.audio_utils",
    "transformers.activations",
    "transformers.modeling_rope_utils",
    "transformers.feature_extraction_utils",
    "certifi",
    "transformers.modeling_flash_attention_utils",
    "transformers.pytorch_utils",
    "transformers._typing",
    "transformers.dynamic_module_utils",
    "transformers.conversion_mapping",
    "transformers.safetensors_conversion",
    "transformers.initialization",
    "transformers.monkey_patching",
    "transformers.dependency_versions_table",
]

# "Cross-Model Common" — loaded by from_pretrained across most models (11)
CROSS_LIBS: List[str] = [
    "transformers.models.auto",
    "torchvision",
    "transformers.generation",
    "transformers.models.encoder_decoder",
    "transformers.image_processing_utils",
    "transformers.tokenization_utils_tokenizers",
    "transformers.tokenization_python",
    "transformers.image_processing_backends",
    "transformers.video_processing_utils",
    "transformers.image_processing_base",
    "tokenizers",
]


def _dedup(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# Order matters: import the heavy roots (torch, transformers) first so the
# submodules that follow are cheap.
PRELOAD_LIBS: List[str] = _dedup(BASE_LIBS + CROSS_LIBS)


def preload() -> dict:
    """Import every preload library once. Returns timing/failure summary.

    Failures are logged but never fatal — a missing optional submodule across
    transformers versions must not stop the worker from coming up warm.
    """
    t0 = time.perf_counter()
    ok, failed = 0, []
    for name in PRELOAD_LIBS:
        try:
            importlib.import_module(name)
            ok += 1
        except Exception as e:  # noqa: BLE001 - intentionally tolerant
            failed.append(name)
            log.warning("preload skip %s (%s)", name, e)
    elapsed = (time.perf_counter() - t0) * 1000.0
    log.info("preloaded %d/%d libraries in %.0f ms (failed: %s)",
             ok, len(PRELOAD_LIBS), elapsed, failed or "none")
    return {"ok": ok, "total": len(PRELOAD_LIBS), "failed": failed,
            "elapsed_ms": round(elapsed, 1)}
