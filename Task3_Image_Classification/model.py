"""
Model loading and inference module for Project 1: AI Image Classification Studio.
Loads torchvision pretrained models (ImageNet-1K), executes evaluation-mode inference,
and computes Top-K class probabilities with latency benchmarking.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.models as models

from config_task3 import AVAILABLE_MODELS, DEFAULT_MODEL_NAME, ModelSpec


class ModelInferenceError(Exception):
    """Raised when model loading or inference fails."""
    pass


# Global cache for standalone scripts and test runners
_MODEL_CACHE: Dict[str, Tuple[torch.nn.Module, Any, List[str]]] = {}


def load_classifier(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[torch.nn.Module, Any, List[str]]:
    """
    Loads a pretrained PyTorch computer vision model and its associated ImageNet categories.

    Uses torchvision's modern `weights` enum API to guarantee authentic weights.
    Sets model to `eval()` mode to deactivate Dropout and freeze BatchNorm statistics.

    Args:
        model_name: Name of the supported model in AVAILABLE_MODELS.

    Returns:
        Tuple of (model, weights_transforms, categories_list)

    Raises:
        ModelInferenceError: If model name is unknown or loading fails.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if model_name not in AVAILABLE_MODELS:
        valid_options = ", ".join(AVAILABLE_MODELS.keys())
        raise ModelInferenceError(
            f"Unknown model '{model_name}'. Choose from: {valid_options}"
        )

    spec: ModelSpec = AVAILABLE_MODELS[model_name]

    try:
        if model_name == "MobileNetV3-Large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights=weights)
        elif model_name == "ResNet-50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        elif model_name == "EfficientNet-B0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        else:
            raise ModelInferenceError(f"Model loader not implemented for: {model_name}")

        # Set strictly to evaluation mode
        model.eval()

        # Retrieve official 1,000 ImageNet category labels
        categories = weights.meta["categories"]
        transforms_fn = weights.transforms()

        _MODEL_CACHE[model_name] = (model, transforms_fn, categories)
        return model, transforms_fn, categories

    except Exception as exc:
        raise ModelInferenceError(
            f"Failed to load pretrained model '{model_name}'. Details: {exc}"
        ) from exc


def clean_label_name(raw_label: str) -> str:
    """
    Formats raw ImageNet category names into clean, professional UI labels.
    Example: 'golden_retriever' -> 'Golden Retriever'
             'airliner' -> 'Airliner'
    """
    return raw_label.replace("_", " ").title()


def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    categories: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Executes model forward pass and computes Top-K class probabilities.

    Args:
        model: Pretrained PyTorch model in eval mode.
        input_tensor: 4D Tensor of shape (1, 3, 224, 224).
        categories: List of 1,000 ImageNet category string names.
        top_k: Number of highest-probability classes to return.

    Returns:
        Dictionary containing:
        - top_predictions: List of dicts with rank, class_id, label, probability, percentage
        - inference_time_ms: Wall-clock latency of forward pass in milliseconds
        - raw_top_probabilities: List of float confidence values
        - raw_top_indices: List of integer class IDs
    """
    if input_tensor.dim() != 4 or input_tensor.size(1) != 3:
        raise ModelInferenceError(
            f"Expected input tensor of shape (batch, 3, H, W), got {tuple(input_tensor.shape)}"
        )

    top_k = max(1, min(top_k, len(categories)))

    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    predictions: List[Dict[str, Any]] = []
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        raw_label = categories[idx]
        display_label = clean_label_name(raw_label)
        predictions.append({
            "rank": rank,
            "class_id": int(idx),
            "raw_label": raw_label,
            "label": display_label,
            "probability": float(prob),
            "percentage": f"{prob * 100:.2f}%",
        })

    return {
        "top_predictions": predictions,
        "inference_time_ms": round(elapsed_ms, 2),
        "raw_top_probabilities": [p["probability"] for p in predictions],
        "raw_top_indices": [p["class_id"] for p in predictions],
    }
