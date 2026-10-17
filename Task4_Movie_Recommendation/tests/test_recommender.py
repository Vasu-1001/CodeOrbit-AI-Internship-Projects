"""
Unit tests for Movie Recommendation System.
"""

import os
import sys
import pytest

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import MovieNotFoundError, MovieRecommender


@pytest.fixture(scope="module")
def rec():
    """Initializes MovieRecommender instance once for test suite."""
    return MovieRecommender()


def test_data_loaded_successfully(rec):
    """Tests that movie dataset is loaded with valid non-empty titles."""
    assert len(rec.df) >= 30
    assert "Inception" in rec.all_titles
    assert "The Dark Knight" in rec.all_titles


def test_exact_movie_match(rec):
    """Tests exact case-insensitive title matching."""
    title, idx, _ = rec.find_movie_match("inception")
    assert title == "Inception"
    assert idx is not None


def test_fuzzy_movie_match(rec):
    """Tests fuzzy search recovers title even with typos."""
    title, idx, _ = rec.find_movie_match("interstelar")
    assert title == "Interstellar"
    assert idx is not None


def test_recommendations_structure(rec):
    """Tests that recommendations return expected keys and descending scores."""
    result = rec.get_recommendations("Inception", top_n=5)

    assert "query_movie" in result
    assert "recommendations" in result
    assert len(result["recommendations"]) == 5

    # Self exclusion: Inception must not be recommended to itself
    rec_titles = [r["title"] for r in result["recommendations"]]
    assert "Inception" not in rec_titles

    # Similarity scores must be descending
    scores = [r["similarity_score"] for r in result["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_genre_filter(rec):
    """Tests that genre filtering restricts results to specified category."""
    result = rec.get_recommendations("Inception", top_n=5, genre_filter="Sci-Fi")

    for r in result["recommendations"]:
        assert "Sci-Fi" in r["genres"]


def test_unknown_movie_error(rec):
    """Tests that searching for a non-existent movie raises MovieNotFoundError."""
    with pytest.raises(MovieNotFoundError, match="was not found in the database"):
        rec.get_recommendations("CompletelyRandomMovieXYZ999", top_n=5)
