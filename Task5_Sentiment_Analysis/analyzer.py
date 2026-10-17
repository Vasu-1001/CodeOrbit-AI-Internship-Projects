"""
Sentiment Analysis Engine using VADER (Valence Aware Dictionary and sEntiment Reasoner).
Genuinely supports three-class classification: Positive, Neutral, and Negative.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config_task5 import (
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    NEGATIVE_THRESHOLD,
    POSITIVE_THRESHOLD,
)


class SentimentAnalysisError(Exception):
    """Raised when text input or dataframe processing is invalid."""
    pass


class SentimentAnalyzer:
    """
    Robust Sentiment Analysis Engine powered by VADER.
    Specifically designed for customer reviews, social media texts, and e-commerce feedback.
    """

    def __init__(self):
        """Initializes the VADER sentiment intensity analyzer."""
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes text while preserving punctuation and capitalization
        which carry critical sentiment valence in VADER (e.g. 'EXCELLENT!!!').
        """
        if not isinstance(text, str):
            return ""
        # Remove null bytes or control characters, normalize whitespace
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return cleaned.strip()

    def analyze_single_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single customer review text.

        Args:
            text: Customer review string.

        Returns:
            Dictionary containing sentiment label, compound score, sub-scores, and interpretation.

        Raises:
            SentimentAnalysisError: If text is empty or whitespace-only.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            raise SentimentAnalysisError("Input text cannot be empty or whitespace only.")

        scores = self.sia.polarity_scores(cleaned)
        compound = scores["compound"]

        # Three-Class Sentiment Classification according to official VADER thresholds
        if compound >= POSITIVE_THRESHOLD:
            sentiment = LABEL_POSITIVE
            confidence = (compound - POSITIVE_THRESHOLD) / (1.0 - POSITIVE_THRESHOLD)
        elif compound <= NEGATIVE_THRESHOLD:
            sentiment = LABEL_NEGATIVE
            confidence = (abs(compound) - abs(NEGATIVE_THRESHOLD)) / (1.0 - abs(NEGATIVE_THRESHOLD))
        else:
            sentiment = LABEL_NEUTRAL
            confidence = 1.0 - (abs(compound) / POSITIVE_THRESHOLD)

        return {
            "text": cleaned,
            "sentiment": sentiment,
            "compound_score": round(float(compound), 4),
            "positive_score": round(float(scores["pos"]), 4),
            "neutral_score": round(float(scores["neu"]), 4),
            "negative_score": round(float(scores["neg"]), 4),
            "confidence_pct": f"{min(1.0, max(0.0, confidence)) * 100:.1f}%",
        }

    def analyze_dataframe(
        self, df: pd.DataFrame, text_column: str = "review_text"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes batch sentiment classification over a pandas DataFrame.

        Args:
            df: Input DataFrame containing review texts.
            text_column: Name of the column with text to analyze.

        Returns:
            Tuple of (augmented_df, summary_metrics_dict)

        Raises:
            SentimentAnalysisError: If text_column does not exist or DataFrame is empty.
        """
        if df.empty:
            raise SentimentAnalysisError("Uploaded CSV is empty (0 rows).")

        if text_column not in df.columns:
            available_cols = ", ".join(df.columns)
            raise SentimentAnalysisError(
                f"Column '{text_column}' not found. Available columns: {available_cols}"
            )

        result_df = df.copy()

        sentiments = []
        compounds = []
        pos_scores = []
        neu_scores = []
        neg_scores = []

        for val in result_df[text_column]:
            cleaned = self.clean_text(str(val)) if pd.notna(val) else ""
            if not cleaned:
                sentiments.append(LABEL_NEUTRAL)
                compounds.append(0.0)
                pos_scores.append(0.0)
                neu_scores.append(1.0)
                neg_scores.append(0.0)
                continue

            scores = self.sia.polarity_scores(cleaned)
            comp = scores["compound"]

            if comp >= POSITIVE_THRESHOLD:
                label = LABEL_POSITIVE
            elif comp <= NEGATIVE_THRESHOLD:
                label = LABEL_NEGATIVE
            else:
                label = LABEL_NEUTRAL

            sentiments.append(label)
            compounds.append(round(comp, 4))
            pos_scores.append(round(scores["pos"], 4))
            neu_scores.append(round(scores["neu"], 4))
            neg_scores.append(round(scores["neg"], 4))

        result_df["Predicted_Sentiment"] = sentiments
        result_df["Compound_Score"] = compounds
        result_df["Positive_Ratio"] = pos_scores
        result_df["Neutral_Ratio"] = neu_scores
        result_df["Negative_Ratio"] = neg_scores

        total = len(result_df)
        pos_count = sum(1 for s in sentiments if s == LABEL_POSITIVE)
        neu_count = sum(1 for s in sentiments if s == LABEL_NEUTRAL)
        neg_count = sum(1 for s in sentiments if s == LABEL_NEGATIVE)

        summary = {
            "total_reviews": total,
            "positive_count": pos_count,
            "neutral_count": neu_count,
            "negative_count": neg_count,
            "positive_pct": round((pos_count / max(1, total)) * 100, 1),
            "neutral_pct": round((neu_count / max(1, total)) * 100, 1),
            "negative_pct": round((neg_count / max(1, total)) * 100, 1),
            "avg_compound": round(float(pd.Series(compounds).mean()), 4),
        }

        return result_df, summary
