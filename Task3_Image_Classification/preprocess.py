"""
Image preprocessing and validation pipeline for Project 1: AI Image Classification Studio.
Ensures rigorous input validation, color mode normalization, and ImageNet standardization.
"""

import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

from config_task3 import (
    ALLOWED_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_CROP_SIZE,
    INPUT_RESIZE_SIZE,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
)


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation (format, size, corruption)."""
    pass


def validate_file_metadata(filename: str, file_size_bytes: int) -> None:
    """
    Validates file extension and size constraints before parsing image content.

    Args:
        filename: Name of the uploaded file.
        file_size_bytes: Size of the raw file in bytes.

    Raises:
        ImageValidationError: If extension is unsupported or file size exceeds maximum.
    """
    if not filename:
        raise ImageValidationError("Invalid file: No filename provided.")

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(ALLOWED_EXTENSIONS)
        raise ImageValidationError(
            f"Unsupported file format '{ext}'. Allowed formats: {allowed_str}."
        )

    if file_size_bytes <= 0:
        raise ImageValidationError("The uploaded file is empty (0 bytes).")

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        raise ImageValidationError(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB."
        )


def load_and_validate_image(
    image_source: Union[bytes, io.BytesIO, str],
    filename: Optional[str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Safely loads, inspects, and normalizes an image to standard 3-channel RGB.

    Handles:
    - RGBA images (removes alpha channel on white background)
    - Grayscale (L) and Palette (P) images
    - CMYK / exotic modes
    - File corruption detection

    Args:
        image_source: Raw bytes, BytesIO buffer, or local file path.
        filename: Optional filename for validation hints.

    Returns:
        Tuple of (clean_pil_image_rgb, metadata_dict)

    Raises:
        ImageValidationError: If the image cannot be decoded or is invalid.
    """
    raw_bytes: bytes
    if isinstance(image_source, bytes):
        raw_bytes = image_source
    elif isinstance(image_source, io.BytesIO):
        raw_bytes = image_source.getvalue()
    elif isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise ImageValidationError(f"Image file not found: {image_source}")
        with open(image_source, "rb") as f:
            raw_bytes = f.read()
        if filename is None:
            filename = os.path.basename(image_source)
    else:
        raise ImageValidationError("Unsupported image source type.")

    if filename:
        validate_file_metadata(filename, len(raw_bytes))
    elif len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"Image size exceeds maximum limit of {MAX_FILE_SIZE_MB} MB."
        )

    try:
        raw_image = Image.open(io.BytesIO(raw_bytes))
        raw_image.verify()  # Verifies file integrity without decoding full raster
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(
            f"Unable to decode image file. File may be corrupted or not a valid image. Details: {exc}"
        ) from exc

    # Re-open after verify() because verify() invalidates the image instance in PIL
    image = Image.open(io.BytesIO(raw_bytes))

    orig_width, orig_height = image.size
    orig_mode = image.mode
    orig_format = image.format or "UNKNOWN"

    # Convert to 3-channel RGB safely
    if image.mode == "RGBA":
        # Composite against pure white background rather than dropping alpha arbitrarily
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        rgb_image = background
    elif image.mode != "RGB":
        rgb_image = image.convert("RGB")
    else:
        rgb_image = image.copy()

    metadata = {
        "filename": filename or "uploaded_image",
        "original_format": orig_format,
        "original_mode": orig_mode,
        "width": orig_width,
        "height": orig_height,
        "aspect_ratio": f"{orig_width / max(1, orig_height):.2f}:1",
        "size_bytes": len(raw_bytes),
        "size_kb": f"{len(raw_bytes) / 1024:.2f} KB",
        "channels": 3,
    }

    return rgb_image, metadata


def get_default_transform() -> transforms.Compose:
    """
    Returns the standard torchvision ImageNet evaluation transform pipeline.

    Steps:
    1. Resize shortest side to 256 using Bicubic/Bilinear interpolation.
    2. Center crop to 224x224.
    3. Convert to Float Tensor scaled to [0.0, 1.0].
    4. Normalize with ImageNet channel mean and standard deviation.
    """
    return transforms.Compose([
        transforms.Resize(INPUT_RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(INPUT_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image_tensor(
    pil_image: Image.Image,
    weights_transforms: Optional[Any] = None,
) -> torch.Tensor:
    """
    Transforms a PIL RGB image into a PyTorch batch tensor ready for model inference.

    Args:
        pil_image: PIL RGB image.
        weights_transforms: Optional torchvision weights.transforms() instance.

    Returns:
        Tensor of shape (1, 3, 224, 224) with dtype float32.
    """
    if weights_transforms is not None:
        tensor = weights_transforms(pil_image)
    else:
        transform = get_default_transform()
        tensor = transform(pil_image)

    # Add batch dimension: (3, H, W) -> (1, 3, H, W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    return tensor
