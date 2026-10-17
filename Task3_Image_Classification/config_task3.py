"""
Configuration and constants for Project 1: AI Image Classification Studio.
CodeOrbit Tech AI Internship - Task 3: Image Classification Using a Pretrained Model.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    architecture: str
    weights_name: str
    model_fn_name: str
    weights_fn_name: str
    approx_size_mb: float
    top1_accuracy: float
    top5_accuracy: float
    target_use_case: str
    description: str


AVAILABLE_MODELS: Dict[str, ModelSpec] = {
    "MobileNetV3-Large": ModelSpec(
        name="MobileNetV3-Large",
        architecture="Inverted Residuals + Squeeze-and-Excitation + Hard-Swish",
        weights_name="MobileNet_V3_Large_Weights.DEFAULT",
        model_fn_name="mobilenet_v3_large",
        weights_fn_name="MobileNet_V3_Large_Weights",
        approx_size_mb=21.1,
        top1_accuracy=75.2,
        top5_accuracy=92.5,
        target_use_case="Mobile / Edge devices, low latency, real-time web inference",
        description=(
            "Designed by Google researchers using Network Architecture Search (NAS) and NetAdapt. "
            "Combines depthwise separable convolutions, squeeze-and-excitation attention modules, "
            "and efficient Hard-Swish non-linearities for lightning-fast CPU inference."
        ),
    ),
    "ResNet-50": ModelSpec(
        name="ResNet-50",
        architecture="50-layer Residual Network with Identity Skip Connections",
        weights_name="ResNet50_Weights.DEFAULT",
        model_fn_name="resnet50",
        weights_fn_name="ResNet50_Weights",
        approx_size_mb=97.8,
        top1_accuracy=80.8,
        top5_accuracy=95.3,
        target_use_case="High-accuracy classification, complex visual feature extraction",
        description=(
            "Pioneered by Kaiming He et al. Introduced residual connections (skip connections) "
            "that solve the vanishing gradient problem, allowing deep 50-layer networks to "
            "train stably and learn rich hierarchical representations."
        ),
    ),
    "EfficientNet-B0": ModelSpec(
        name="EfficientNet-B0",
        architecture="Compound-scaled Mobile Inverted Bottleneck (MBConv)",
        weights_name="EfficientNet_B0_Weights.DEFAULT",
        model_fn_name="efficientnet_b0",
        weights_fn_name="EfficientNet_B0_Weights",
        approx_size_mb=20.5,
        top1_accuracy=77.1,
        top5_accuracy=93.4,
        target_use_case="Balanced efficiency and accuracy on consumer hardware",
        description=(
            "Introduced by Tan and Le (Google Research). Uses a compound scaling method that "
            "uniformly scales depth, width, and resolution with a fixed compound coefficient."
        ),
    ),
}

DEFAULT_MODEL_NAME = "MobileNetV3-Large"

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
INPUT_RESIZE_SIZE: int = 256
INPUT_CROP_SIZE: int = 224

ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
MAX_FILE_SIZE_MB: int = 10
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

DEFAULT_TOP_K: int = 5
MIN_TOP_K: int = 1
MAX_TOP_K: int = 10

INTERNSHIP_TASK_NAME = "Task 3: Image Classification Using a Pretrained Model"
INTERNSHIP_ORGANIZATION = "CodeOrbit Tech"
PROJECT_TITLE = "AI Image Classification Studio"
