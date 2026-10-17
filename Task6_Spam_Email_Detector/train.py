"""
Model Training Script for AI Spam Email Detector.
Implements data loading, train/test split (no data leakage), TF-IDF feature extraction,
Multinomial Naive Bayes classification, evaluation metrics calculation, and joblib serialization.
"""

import json
import os
import sys
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from config_task6 import (
    DATASET_PATH,
    LABEL_HAM,
    LABEL_SPAM,
    METRICS_PATH,
    MODEL_PIPELINE_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)


def clean_text_for_training(text: str) -> str:
    """Cleans text while preserving informative spam tokens."""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def train_spam_detector(
    dataset_path: str = DATASET_PATH,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains and evaluates a Multinomial Naive Bayes spam classifier pipeline.

    Guarantees strict separation of training and test sets to prevent data leakage.
    Vectorization vocabulary is learned exclusively from the training split.

    Args:
        dataset_path: Path to labeled CSV file.

    Returns:
        Tuple of (trained_pipeline, evaluation_metrics_dict)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Spam dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Standardize label mapping
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["clean_label"] = df["label"].map({"spam": LABEL_SPAM, "ham": LABEL_HAM})
    df = df.dropna(subset=["clean_label", "text"]).reset_index(drop=True)

    X = df["text"].apply(clean_text_for_training)
    y = df["clean_label"]

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Construct unified scikit-learn Pipeline
    # TfidfVectorizer fits only on X_train, preventing test data leakage
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=TFIDF_NGRAM_RANGE,
                max_features=TFIDF_MAX_FEATURES,
                stop_words="english",
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.5),  # Laplace smoothing
        ),
    ])

    # Fit pipeline exclusively on training split
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test split
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=LABEL_SPAM, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=LABEL_SPAM, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=LABEL_SPAM, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[LABEL_HAM, LABEL_SPAM])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "dataset_total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(float(acc), 4),
        "accuracy_pct": f"{acc * 100:.2f}%",
        "precision": round(float(prec), 4),
        "precision_pct": f"{prec * 100:.2f}%",
        "recall": round(float(rec), 4),
        "recall_pct": f"{rec * 100:.2f}%",
        "f1_score": round(float(f1), 4),
        "f1_score_pct": f"{f1 * 100:.2f}%",
        "confusion_matrix": {
            "true_negatives_ham": int(tn),
            "false_positives_false_spam": int(fp),
            "false_negatives_missed_spam": int(fn),
            "true_positives_spam": int(tp),
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    # Save serialized model pipeline and evaluation report
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PIPELINE_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, metrics


if __name__ == "__main__":
    print("Training AI Spam Email Detector Pipeline...")
    model_pipeline, eval_metrics = train_spam_detector()
    print("Training Completed Successfully!")
    print(f"Accuracy:  {eval_metrics['accuracy_pct']}")
    print(f"Precision: {eval_metrics['precision_pct']}")
    print(f"Recall:    {eval_metrics['recall_pct']}")
    print(f"F1-Score:  {eval_metrics['f1_score_pct']}")
    print(f"Model saved to: {MODEL_PIPELINE_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
