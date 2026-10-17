"""
Unit tests for model loading and inference engine.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    ModelInferenceError,
    clean_label_name,
    load_classifier,
    run_inference,
)


def test_clean_label_name():
    """Tests label name formatting for UI display."""
    assert clean_label_name("golden_retriever") == "Golden Retriever"
    assert clean_label_name("sports_car") == "Sports Car"
    assert clean_label_name("airliner") == "Airliner"


def test_load_classifier_invalid_model():
    """Tests that requesting an unsupported model raises ModelInferenceError."""
    with pytest.raises(ModelInferenceError, match="Unknown model"):
        load_classifier("NonExistentSuperNet")


def test_load_classifier_mobilenet():
    """Tests that MobileNetV3-Large loads in eval mode with 1,000 ImageNet categories."""
    model, transforms_fn, categories = load_classifier("MobileNetV3-Large")

    assert model is not None
    assert not model.training  # Must be strictly in eval mode
    assert callable(transforms_fn)
    assert len(categories) == 1000
    assert "golden retriever" in categories or "airliner" in categories or "sports car" in categories


def test_run_inference_top_k():
    """Tests that run_inference returns valid probabilities and proper Top-K ranking."""
    model, _, categories = load_classifier("MobileNetV3-Large")

    # Generate dummy input tensor (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    result = run_inference(model, dummy_input, categories, top_k=5)

    assert "top_predictions" in result
    assert "inference_time_ms" in result
    assert len(result["top_predictions"]) == 5

    # Verify descending probability order
    probs = [p["probability"] for p in result["top_predictions"]]
    assert probs == sorted(probs, reverse=True)

    # Probabilities must be valid in [0.0, 1.0]
    for p in probs:
        assert 0.0 <= p <= 1.0

    # Ranks must be 1 to 5
    ranks = [p["rank"] for p in result["top_predictions"]]
    assert ranks == [1, 2, 3, 4, 5]


def test_run_inference_invalid_tensor():
    """Tests that malformed input tensor raises ModelInferenceError."""
    model, _, categories = load_classifier("MobileNetV3-Large")
    invalid_tensor = torch.randn(3, 224, 224)  # Missing batch dimension

    with pytest.raises(ModelInferenceError, match="Expected input tensor"):
        run_inference(model, invalid_tensor, categories, top_k=5)
