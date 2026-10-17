"""
Unit tests for Customer Sentiment Analyzer.
"""

import os
import sys
import pandas as pd
import pytest

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import SentimentAnalysisError, SentimentAnalyzer
from config_task5 import LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_positive_sentiment(analyzer):
    """Tests that strong positive praise is classified as Positive."""
    res = analyzer.analyze_single_text("This laptop is absolutely amazing! Super fast and gorgeous screen.")
    assert res["sentiment"] == LABEL_POSITIVE
    assert res["compound_score"] > 0.05


def test_negative_sentiment(analyzer):
    """Tests that harsh criticism is classified as Negative."""
    res = analyzer.analyze_single_text("Horrible service. The package was completely ruined and customer support was rude.")
    assert res["sentiment"] == LABEL_NEGATIVE
    assert res["compound_score"] < -0.05


def test_neutral_sentiment(analyzer):
    """Tests that objective factual statements are classified as Neutral."""
    res = analyzer.analyze_single_text("The package arrived on Tuesday. The item is made of plastic and metal.")
    assert res["sentiment"] == LABEL_NEUTRAL
    assert -0.05 <= res["compound_score"] <= 0.05


def test_empty_text_error(analyzer):
    """Tests that empty or whitespace strings raise SentimentAnalysisError."""
    with pytest.raises(SentimentAnalysisError, match="empty or whitespace only"):
        analyzer.analyze_single_text("   \n\t  ")


def test_dataframe_batch_analysis(analyzer):
    """Tests batch processing of review dataframe."""
    df = pd.DataFrame({
        "review_text": [
            "Fantastic quality and great speed!",
            "Completely broken on arrival, total waste of money.",
            "The package weighs two kilograms.",
        ]
    })

    res_df, summary = analyzer.analyze_dataframe(df, text_column="review_text")

    assert "Predicted_Sentiment" in res_df.columns
    assert summary["total_reviews"] == 3
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert summary["neutral_count"] == 1
