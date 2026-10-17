"""
Configuration and constants for Project 2: Smart Movie Recommendation System.
CodeOrbit Tech AI Internship - Task 4: Simple Recommendation System.
"""

import os
from dataclasses import dataclass
from typing import List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "movies_dataset.csv")

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"
MIN_SIMILARITY_THRESHOLD = 0.05
DEFAULT_TOP_N = 5
MAX_TOP_N = 15

INTERNSHIP_TASK_NAME = "Task 4: Simple Recommendation System"
INTERNSHIP_ORGANIZATION = "CodeOrbit Tech"
PROJECT_TITLE = "Smart Movie Recommendation System"

ALL_GENRES: List[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Music",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "Western",
]
