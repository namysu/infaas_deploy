"""Canonical model registry — the only models the executor serves.

Source of truth: the "Model list" section of CLAUDE.md. Each entry is a
HuggingFace repo id plus the category it was listed under. The category is kept
as metadata (logging / introspection); the actual pipeline task is auto-detected
from each model's own config at load time, which is more reliable than the
section headings (several entries are backbones rather than task-specific heads).

The server uses SUPPORTED_MODELS for admission: a request for a model not in
this list is rejected before any predictor/scheduler work.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# (repo_id, category) in the exact order of CLAUDE.md's Model list.
MODELS: List[tuple] = [
    # IMAGE CLASSIFICATION (22)
    ("microsoft/resnet-18", "image-classification"),
    ("microsoft/resnet-34", "image-classification"),
    ("microsoft/resnet-50", "image-classification"),
    ("google/vit-base-patch16-224", "image-classification"),
    ("google/vit-base-patch16-224-in21k", "image-classification"),
    ("facebook/deit-tiny-patch16-224", "image-classification"),
    ("facebook/deit-small-patch16-224", "image-classification"),
    ("facebook/deit-base-patch16-224", "image-classification"),
    ("facebook/convnext-tiny-224", "image-classification"),
    ("facebook/convnext-small-224", "image-classification"),
    ("google/efficientnet-b0", "image-classification"),
    ("google/efficientnet-b1", "image-classification"),
    ("google/efficientnet-b2", "image-classification"),
    ("google/mobilenet_v1_1.0_224", "image-classification"),
    ("google/mobilenet_v2_1.0_224", "image-classification"),
    ("google/mobilenet_v2_0.75_160", "image-classification"),
    ("apple/mobilevit-small", "image-classification"),
    ("apple/mobilevit-xx-small", "image-classification"),
    ("facebook/levit-128S", "image-classification"),
    ("facebook/levit-192", "image-classification"),
    ("facebook/levit-256", "image-classification"),
    ("sail/poolformer_s12", "image-classification"),
    # OBJECT DETECTION (11)
    ("hustvl/yolos-tiny", "object-detection"),
    ("hustvl/yolos-small", "object-detection"),
    ("hustvl/yolos-base", "object-detection"),
    ("facebook/dino-vits16", "object-detection"),
    ("facebook/dino-vits8", "object-detection"),
    ("facebook/dino-vitb16", "object-detection"),
    ("facebook/dinov2-small", "object-detection"),
    ("facebook/dinov2-base", "object-detection"),
    ("hustvl/yolos-small-300", "object-detection"),
    ("hustvl/yolos-small-dwr", "object-detection"),
    ("microsoft/beit-base-patch16-224-pt22k-ft22k", "object-detection"),
    # DEPTH ESTIMATION (3)
    ("apple/deeplabv3-mobilevit-small", "depth-estimation"),
    ("apple/deeplabv3-mobilevit-x-small", "depth-estimation"),
    ("apple/deeplabv3-mobilevit-xx-small", "depth-estimation"),
    # POSE ESTIMATION (6)
    ("nvidia/mit-b0", "pose-estimation"),
    ("nvidia/mit-b1", "pose-estimation"),
    ("nvidia/mit-b2", "pose-estimation"),
    ("nvidia/mit-b3", "pose-estimation"),
    ("nvidia/mit-b4", "pose-estimation"),
    ("nvidia/mit-b5", "pose-estimation"),
    # SEMANTIC SEGMENTATION (1)
    ("microsoft/beit-base-patch16-224", "semantic-segmentation"),
]

MODEL_CATEGORY: Dict[str, str] = {mid: cat for mid, cat in MODELS}
SUPPORTED_MODELS = set(MODEL_CATEGORY.keys())

# Which transformers AutoClass to use when loading each model. Backbone-only
# checkpoints (ViT-in21k, DINO/DINOv2, SegFormer encoders) have no task head, so
# we load them as plain AutoModel. Used by both the downloader and the worker's
# startup loader.
MODEL_AUTOCLASS: Dict[str, str] = {
    # IMAGE CLASSIFICATION (head)
    "microsoft/resnet-18": "AutoModelForImageClassification",
    "microsoft/resnet-34": "AutoModelForImageClassification",
    "microsoft/resnet-50": "AutoModelForImageClassification",
    "google/vit-base-patch16-224": "AutoModelForImageClassification",
    "google/vit-base-patch16-224-in21k": "AutoModel",
    "facebook/deit-tiny-patch16-224": "AutoModelForImageClassification",
    "facebook/deit-small-patch16-224": "AutoModelForImageClassification",
    "facebook/deit-base-patch16-224": "AutoModelForImageClassification",
    "facebook/convnext-tiny-224": "AutoModelForImageClassification",
    "facebook/convnext-small-224": "AutoModelForImageClassification",
    "google/efficientnet-b0": "AutoModelForImageClassification",
    "google/efficientnet-b1": "AutoModelForImageClassification",
    "google/efficientnet-b2": "AutoModelForImageClassification",
    "google/mobilenet_v1_1.0_224": "AutoModelForImageClassification",
    "google/mobilenet_v2_1.0_224": "AutoModelForImageClassification",
    "google/mobilenet_v2_0.75_160": "AutoModelForImageClassification",
    "apple/mobilevit-small": "AutoModelForImageClassification",
    "apple/mobilevit-xx-small": "AutoModelForImageClassification",
    "facebook/levit-128S": "AutoModelForImageClassification",
    "facebook/levit-192": "AutoModelForImageClassification",
    "facebook/levit-256": "AutoModelForImageClassification",
    "sail/poolformer_s12": "AutoModelForImageClassification",
    # OBJECT DETECTION
    "hustvl/yolos-tiny": "AutoModelForObjectDetection",
    "hustvl/yolos-small": "AutoModelForObjectDetection",
    "hustvl/yolos-base": "AutoModelForObjectDetection",
    "facebook/dino-vits16": "AutoModel",
    "facebook/dino-vits8": "AutoModel",
    "facebook/dino-vitb16": "AutoModel",
    "facebook/dinov2-small": "AutoModel",
    "facebook/dinov2-base": "AutoModel",
    "hustvl/yolos-small-300": "AutoModelForObjectDetection",
    "hustvl/yolos-small-dwr": "AutoModelForObjectDetection",
    "microsoft/beit-base-patch16-224-pt22k-ft22k": "AutoModelForImageClassification",
    # DEPTH ESTIMATION (deeplabv3 = semantic seg head)
    "apple/deeplabv3-mobilevit-small": "AutoModelForSemanticSegmentation",
    "apple/deeplabv3-mobilevit-x-small": "AutoModelForSemanticSegmentation",
    "apple/deeplabv3-mobilevit-xx-small": "AutoModelForSemanticSegmentation",
    # POSE ESTIMATION = SegFormer encoders (no head)
    "nvidia/mit-b0": "AutoModel",
    "nvidia/mit-b1": "AutoModel",
    "nvidia/mit-b2": "AutoModel",
    "nvidia/mit-b3": "AutoModel",
    "nvidia/mit-b4": "AutoModel",
    "nvidia/mit-b5": "AutoModel",
    # SEMANTIC SEGMENTATION (the BEiT-base here is actually a classifier)
    "microsoft/beit-base-patch16-224": "AutoModelForImageClassification",
}

# Users refer to models by short name (the repo suffix, e.g. "resnet-50"), per
# CLAUDE.md. Suffixes are unique across the 42-model list, so the mapping to the
# full HuggingFace id is unambiguous.
SHORT_TO_ID: Dict[str, str] = {mid.split("/")[-1]: mid for mid in SUPPORTED_MODELS}


def resolve(name: str) -> Optional[str]:
    """Return the full HF repo id for a short name or full id; None if unknown."""
    if name in SUPPORTED_MODELS:
        return name
    return SHORT_TO_ID.get(name)


def is_supported(name: str) -> bool:
    return resolve(name) is not None


def category_of(model_id: str) -> Optional[str]:
    return MODEL_CATEGORY.get(model_id)
