"""
Unit tests for preprocessing and validation pipeline.
"""

import io
import os
import sys
import pytest
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import (
    ImageValidationError,
    get_default_transform,
    load_and_validate_image,
    preprocess_image_tensor,
    validate_file_metadata,
)
from config_task3 import MAX_FILE_SIZE_BYTES


def test_validate_file_metadata_valid():
    """Tests valid file extensions and sizes pass without error."""
    validate_file_metadata("test_sample.jpg", 1024 * 500)
    validate_file_metadata("photo.PNG", 1024 * 1024)
    validate_file_metadata("graphic.webp", 2048)


def test_validate_file_metadata_invalid_extension():
    """Tests unsupported file extensions raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="Unsupported file format"):
        validate_file_metadata("document.pdf", 1024)

    with pytest.raises(ImageValidationError, match="Unsupported file format"):
        validate_file_metadata("script.py", 1024)


def test_validate_file_metadata_empty_file():
    """Tests zero-byte files raise ImageValidationError."""
    with pytest.raises(ImageValidationError, match="empty"):
        validate_file_metadata("empty.jpg", 0)


def test_validate_file_metadata_oversize():
    """Tests files exceeding max size limit raise ImageValidationError."""
    oversize = MAX_FILE_SIZE_BYTES + 1024
    with pytest.raises(ImageValidationError, match="exceeds maximum allowed limit"):
        validate_file_metadata("huge.jpg", oversize)


def test_load_and_validate_rgba_image():
    """Tests that an RGBA image with transparency is cleanly converted to 3-channel RGB."""
    # Create RGBA image with alpha transparency
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    clean_img, metadata = load_and_validate_image(raw_bytes, filename="test_rgba.png")

    assert clean_img.mode == "RGB"
    assert metadata["original_mode"] == "RGBA"
    assert metadata["width"] == 100
    assert metadata["height"] == 100
    assert metadata["channels"] == 3


def test_load_and_validate_grayscale_image():
    """Tests that a single-channel Grayscale (L) image is properly converted to RGB."""
    img = Image.new("L", (80, 80), color=150)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    clean_img, metadata = load_and_validate_image(raw_bytes, filename="grayscale.png")

    assert clean_img.mode == "RGB"
    assert metadata["original_mode"] == "L"
    assert metadata["width"] == 80
    assert metadata["height"] == 80


def test_load_and_validate_corrupted_file():
    """Tests that arbitrary corrupted bytes fail validation gracefully."""
    junk_bytes = b"NOT_A_VALID_IMAGE_HEADER_1234567890"
    with pytest.raises(ImageValidationError, match="Unable to decode image"):
        load_and_validate_image(junk_bytes, filename="corrupted.jpg")


def test_preprocess_image_tensor_shape():
    """Tests that preprocessed output tensor matches expected PyTorch ImageNet input format."""
    sample_img = Image.new("RGB", (300, 450), color=(100, 150, 200))
    tensor = preprocess_image_tensor(sample_img)

    # Must be 4-dimensional: (Batch, Channels, Height, Width)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
