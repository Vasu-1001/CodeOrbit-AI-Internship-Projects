"""
Data loading and preprocessing pipeline for Smart Movie Recommendation System.
Loads, validates, cleans, and builds feature soup from movie metadata.
"""

import os
from typing import Optional, Tuple

import pandas as pd

from config_task4 import DEFAULT_DATASET_PATH


REQUIRED_COLUMNS = [
    "movie_id",
    "title",
    "genres",
    "overview",
    "director",
    "cast_keywords",
    "release_year",
    "vote_average",
]


class MovieDataError(Exception):
    """Raised when movie dataset loading or validation fails."""
    pass


def load_movies_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and cleans the movie dataset.

    Args:
        csv_path: Optional path to CSV file. Defaults to bundled movies_dataset.csv.

    Returns:
        Cleaned pandas DataFrame with 'metadata_soup' column.

    Raises:
        MovieDataError: If file not found or required columns are missing.
    """
    path = csv_path or DEFAULT_DATASET_PATH

    if not os.path.exists(path):
        raise MovieDataError(
            f"Movie dataset file not found at '{path}'. "
            "Please ensure data/movies_dataset.csv exists."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise MovieDataError(f"Failed to read movie CSV file: {exc}") from exc

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise MovieDataError(f"Dataset is missing required columns: {missing}")

    # Remove duplicates based on title
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    # Clean text columns and fill missing values
    for col in ["title", "genres", "overview", "director", "cast_keywords"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Fill numerical columns safely
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0).astype(float)

    # Build weighted metadata soup:
    # Giving double weight to genres and director to enhance semantic clustering
    df["metadata_soup"] = (
        df["title"] + " "
        + (df["genres"] + " ") * 2
        + df["overview"] + " "
        + (df["director"] + " ") * 2
        + df["cast_keywords"]
    ).str.lower()

    return df
