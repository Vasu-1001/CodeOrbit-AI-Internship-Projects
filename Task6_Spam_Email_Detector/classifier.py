"""
Inference Engine for AI Spam Email Detector.
Loads the trained pipeline, computes posterior probabilities, extracts informative tokens,
and performs batch CSV inference.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from config_task6 import (
    LABEL_HAM,
    LABEL_SPAM,
    METRICS_PATH,
    MODEL_PIPELINE_PATH,
)
from train import train_spam_detector


class SpamClassifierError(Exception):
    """Raised when text input is invalid or inference fails."""
    pass


class SpamClassifier:
    """
    Spam detection inference engine using serialized Naive Bayes pipeline.
    """

    def __init__(self, pipeline_path: Optional[str] = None):
        """Loads the serialized pipeline and metrics, auto-training if absent."""
        self.pipeline_path = pipeline_path or MODEL_PIPELINE_PATH
        self.metrics = None
        self._load_or_train_pipeline()

    def _load_or_train_pipeline(self) -> None:
        """Loads pipeline from disk or initiates training if model file is missing."""
        if not os.path.exists(self.pipeline_path):
            self.pipeline, self.metrics = train_spam_detector()
        else:
            self.pipeline = joblib.load(self.pipeline_path)
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    self.metrics = json.load(f)

    def clean_text(self, text: str) -> str:
        """Cleans input text for classification."""
        if not isinstance(text, str):
            return ""
        return text.strip().lower()

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Classifies an individual email or message as Spam or Legitimate.

        Args:
            text: Raw message content.

        Returns:
            Dictionary containing prediction label, confidence score, probabilities, and warnings.

        Raises:
            SpamClassifierError: If text is empty or whitespace only.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            raise SpamClassifierError("Input message text cannot be empty or whitespace only.")

        # Forward pass through pipeline
        probs = self.pipeline.predict_proba([cleaned])[0]
        classes = list(self.pipeline.classes_)

        # Extract probabilities
        spam_idx = classes.index(LABEL_SPAM) if LABEL_SPAM in classes else 1
        ham_idx = classes.index(LABEL_HAM) if LABEL_HAM in classes else 0

        spam_prob = float(probs[spam_idx])
        ham_prob = float(probs[ham_idx])

        prediction = LABEL_SPAM if spam_prob >= 0.50 else LABEL_HAM
        confidence = max(spam_prob, ham_prob)

        # Highlight detected trigger words
        tfidf = self.pipeline.named_steps["tfidf"]
        feature_names = tfidf.get_feature_names_out()
        words_in_text = set(re.findall(r"\b\w+\b", cleaned))
        matched_features = [w for w in words_in_text if w in feature_names]

        return {
            "prediction": prediction,
            "is_spam": bool(prediction == LABEL_SPAM),
            "confidence_score": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.1f}%",
            "spam_probability": round(spam_prob, 4),
            "spam_pct": f"{spam_prob * 100:.1f}%",
            "legitimate_probability": round(ham_prob, 4),
            "legitimate_pct": f"{ham_prob * 100:.1f}%",
            "detected_vocabulary_terms": matched_features[:8],
            "disclaimer": (
                "Notice: Model confidence scores represent posterior conditional probabilities, "
                "not an infallible guarantee. Always verify urgent financial or password requests."
            ),
        }

    def predict_batch(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes batch classification over a DataFrame.

        Args:
            df: Input DataFrame.
            text_column: Column name containing email messages.

        Returns:
            Tuple of (augmented_df, summary_stats_dict)
        """
        if df.empty:
            raise SpamClassifierError("Uploaded CSV is empty (0 rows).")

        if text_column not in df.columns:
            available = ", ".join(df.columns)
            raise SpamClassifierError(
                f"Column '{text_column}' not found. Available columns: {available}"
            )

        result_df = df.copy()
        clean_texts = [self.clean_text(str(t)) for t in result_df[text_column]]

        preds = self.pipeline.predict(clean_texts)
        probs = self.pipeline.predict_proba(clean_texts)

        classes = list(self.pipeline.classes_)
        spam_idx = classes.index(LABEL_SPAM) if LABEL_SPAM in classes else 1
        ham_idx = classes.index(LABEL_HAM) if LABEL_HAM in classes else 0

        result_df["Predicted_Label"] = preds
        result_df["Spam_Probability"] = [round(float(p[spam_idx]), 4) for p in probs]
        result_df["Legitimate_Probability"] = [round(float(p[ham_idx]), 4) for p in probs]

        total = len(result_df)
        spam_count = sum(1 for p in preds if p == LABEL_SPAM)
        ham_count = sum(1 for p in preds if p == LABEL_HAM)

        summary = {
            "total_messages": total,
            "spam_count": spam_count,
            "legitimate_count": ham_count,
            "spam_pct": round((spam_count / max(1, total)) * 100, 1),
            "legitimate_pct": round((ham_count / max(1, total)) * 100, 1),
        }

        return result_df, summary
