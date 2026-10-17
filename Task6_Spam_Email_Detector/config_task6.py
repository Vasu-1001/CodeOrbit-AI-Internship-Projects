"""
Configuration and constants for Project 4: AI Spam Email Detector.
CodeOrbit Tech AI Internship - Task 6: Mini Project — AI-Based Mini Application.
"""

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "spam_dataset.csv")
MODEL_PIPELINE_PATH = os.path.join(MODELS_DIR, "spam_classifier_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")

TEST_SIZE = 0.20
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 2500
TFIDF_NGRAM_RANGE = (1, 2)

LABEL_SPAM = "Spam"
LABEL_HAM = "Legitimate"

INTERNSHIP_TASK_NAME = "Task 6: Mini Project — AI-Based Mini Application"
INTERNSHIP_ORGANIZATION = "CodeOrbit Tech"
PROJECT_TITLE = "AI Spam Email Detector"
