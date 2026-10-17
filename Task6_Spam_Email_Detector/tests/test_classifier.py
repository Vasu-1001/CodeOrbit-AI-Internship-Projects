"""
Unit tests for AI Spam Email Detector.
"""

import os
import sys
import pandas as pd
import pytest

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import SpamClassifier, SpamClassifierError
from config_task6 import LABEL_HAM, LABEL_SPAM
from train import train_spam_detector


@pytest.fixture(scope="module")
def classifier():
    return SpamClassifier()


def test_train_pipeline_execution():
    """Tests that train_spam_detector completes and produces valid metrics."""
    pipeline, metrics = train_spam_detector()
    assert pipeline is not None
    assert metrics["accuracy"] >= 0.85
    assert metrics["precision"] >= 0.80
    assert "confusion_matrix" in metrics


def test_predict_spam_message(classifier):
    """Tests that high-risk spam phrasing is classified as Spam."""
    text = "CONGRATULATIONS! You have won a guaranteed $1000 cash prize. Call now to claim."
    res = classifier.predict_single(text)
    assert res["prediction"] == LABEL_SPAM
    assert res["is_spam"] is True
    assert res["spam_probability"] > 0.50


def test_predict_legitimate_message(classifier):
    """Tests that standard workplace communications are classified as Legitimate."""
    text = "Hi team, please find the slide deck for tomorrow's 10 AM project status meeting."
    res = classifier.predict_single(text)
    assert res["prediction"] == LABEL_HAM
    assert res["is_spam"] is False
    assert res["legitimate_probability"] > 0.50


def test_empty_message_error(classifier):
    """Tests that empty input text raises SpamClassifierError."""
    with pytest.raises(SpamClassifierError, match="cannot be empty or whitespace only"):
        classifier.predict_single("   ")


def test_batch_classification(classifier):
    """Tests batch prediction across a DataFrame."""
    df = pd.DataFrame({
        "text": [
            "WINNER! Free tickets to Hawaii. Text CLAIM to 88000.",
            "Can we reschedule our coffee meeting to Friday afternoon?",
        ]
    })
    res_df, summary = classifier.predict_batch(df, text_column="text")

    assert "Predicted_Label" in res_df.columns
    assert summary["total_messages"] == 2
    assert summary["spam_count"] == 1
    assert summary["legitimate_count"] == 1
