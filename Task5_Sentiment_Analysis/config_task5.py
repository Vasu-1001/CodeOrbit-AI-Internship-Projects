"""
Configuration and constants for Project 3: Customer Sentiment Analysis Platform.
CodeOrbit Tech AI Internship - Task 5: Sentiment Analysis on Text Data.
"""

import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_SAMPLE_CSV = os.path.join(DATA_DIR, "sample_reviews.csv")

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

LABEL_POSITIVE = "Positive"
LABEL_NEUTRAL = "Neutral"
LABEL_NEGATIVE = "Negative"

INTERNSHIP_TASK_NAME = "Task 5: Sentiment Analysis on Text Data"
INTERNSHIP_ORGANIZATION = "CodeOrbit Tech"
PROJECT_TITLE = "Customer Sentiment Analysis Platform"
