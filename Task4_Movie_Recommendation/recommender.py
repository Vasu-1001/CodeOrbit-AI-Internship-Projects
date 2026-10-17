"""
Content-Based Recommendation Engine using TF-IDF and Cosine Similarity.
Provides similarity ranking, fuzzy title lookup, genre filtering, and explainability.
"""

import difflib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config_task4 import (
    DEFAULT_TOP_N,
    MIN_SIMILARITY_THRESHOLD,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
)
from data_loader import load_movies_data


class MovieNotFoundError(Exception):
    """Raised when a requested movie title cannot be found in the dataset."""
    pass


class MovieRecommender:
    """
    Content-Based Movie Recommendation Engine.
    Uses TF-IDF Vectorization over metadata soups and computes Cosine Similarity.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initializes the recommender with the movie DataFrame and precomputes matrices."""
        self.df = df if df is not None else load_movies_data()
        self._build_similarity_matrix()

    def _build_similarity_matrix(self) -> None:
        """Fits TF-IDF vectorizer and calculates dense cosine similarity matrix."""
        self.vectorizer = TfidfVectorizer(
            stop_words=TFIDF_STOP_WORDS,
            ngram_range=TFIDF_NGRAM_RANGE,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["metadata_soup"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Build case-insensitive lookup index: lowercase_title -> row_idx
        self.title_to_idx = {
            title.lower().strip(): idx for idx, title in enumerate(self.df["title"])
        }
        self.all_titles = list(self.df["title"].values)

    def find_movie_match(
        self, query_title: str, cutoff: float = 0.5
    ) -> Tuple[Optional[str], Optional[int], List[str]]:
        """
        Looks up a movie title with exact matching and fuzzy matching fallback.

        Args:
            query_title: Title entered by user.
            cutoff: Minimum similarity ratio for fuzzy matching (0.0 to 1.0).

        Returns:
            Tuple of (matched_official_title, row_idx, suggestions_list)
        """
        clean_query = query_title.strip()
        lower_query = clean_query.lower()

        # 1. Exact match
        if lower_query in self.title_to_idx:
            idx = self.title_to_idx[lower_query]
            return self.df.iloc[idx]["title"], idx, []

        # 2. Substring match
        substring_matches = [
            title for title in self.all_titles if lower_query in title.lower()
        ]
        if substring_matches:
            matched_title = substring_matches[0]
            idx = self.title_to_idx[matched_title.lower()]
            return matched_title, idx, substring_matches[1:4]

        # 3. Fuzzy match (Difflib)
        fuzzy_matches = difflib.get_close_matches(
            clean_query, self.all_titles, n=4, cutoff=cutoff
        )
        if fuzzy_matches:
            matched_title = fuzzy_matches[0]
            idx = self.title_to_idx[matched_title.lower()]
            return matched_title, idx, fuzzy_matches[1:]

        return None, None, []

    def _explain_recommendation(
        self, query_row: pd.Series, target_row: pd.Series, sim_score: float
    ) -> Dict[str, Any]:
        """
        Generates human-readable explainability rationale for why a movie was recommended.
        """
        query_genres = set(query_row["genres"].split())
        target_genres = set(target_row["genres"].split())
        shared_genres = list(query_genres.intersection(target_genres))

        same_director = bool(
            query_row["director"]
            and target_row["director"]
            and query_row["director"].lower() == target_row["director"].lower()
        )

        # Find overlapping keywords
        query_kw = set(query_row["cast_keywords"].lower().split())
        target_kw = set(target_row["cast_keywords"].lower().split())
        shared_kw = list(query_kw.intersection(target_kw))

        reasons = []
        if same_director:
            reasons.append(f"Directed by the same filmmaker ({query_row['director']})")
        if shared_genres:
            reasons.append(f"Shares genre categories: {', '.join(shared_genres)}")
        if shared_kw:
            reasons.append(f"Thematic overlap: {', '.join(shared_kw[:3])}")
        if not reasons:
            reasons.append(f"High semantic narrative similarity ({sim_score * 100:.1f}%)")

        return {
            "shared_genres": shared_genres,
            "same_director": same_director,
            "shared_keywords": shared_kw,
            "summary_reason": " • ".join(reasons),
        }

    def get_recommendations(
        self,
        movie_title: str,
        top_n: int = DEFAULT_TOP_N,
        genre_filter: Optional[str] = None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Generates Top-N recommendations for a given movie title.

        Args:
            movie_title: Name of the reference movie.
            top_n: Maximum recommendations to return.
            genre_filter: Optional genre string to restrict recommendations (e.g., 'Sci-Fi').
            min_similarity: Minimum cosine similarity score threshold.

        Returns:
            Dictionary containing query_movie details and recommendations list.

        Raises:
            MovieNotFoundError: If title is unknown and cannot be matched.
        """
        official_title, query_idx, suggestions = self.find_movie_match(movie_title)

        if query_idx is None:
            raise MovieNotFoundError(
                f"Movie '{movie_title}' was not found in the database. "
                f"Suggestions: {', '.join(suggestions) if suggestions else 'Try browsing the movie catalog dropdown.'}"
            )

        query_row = self.df.iloc[query_idx]

        # Extract similarity vector for query movie against all movies
        sim_scores = list(enumerate(self.similarity_matrix[query_idx]))

        # Sort descending by similarity score
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for idx, score in sorted_scores:
            if idx == query_idx:
                continue  # Skip the query movie itself

            if score < min_similarity:
                continue

            target_row = self.df.iloc[idx]

            # Apply genre filter if requested
            if genre_filter and genre_filter != "All Genres":
                if genre_filter.lower() not in target_row["genres"].lower():
                    continue

            explanation = self._explain_recommendation(query_row, target_row, score)

            recommendations.append({
                "movie_id": int(target_row["movie_id"]),
                "title": target_row["title"],
                "genres": target_row["genres"],
                "overview": target_row["overview"],
                "director": target_row["director"],
                "release_year": int(target_row["release_year"]),
                "vote_average": float(target_row["vote_average"]),
                "similarity_score": round(float(score), 4),
                "similarity_percentage": f"{score * 100:.1f}%",
                "explanation": explanation,
            })

            if len(recommendations) >= top_n:
                break

        return {
            "query_movie": {
                "title": query_row["title"],
                "genres": query_row["genres"],
                "overview": query_row["overview"],
                "director": query_row["director"],
                "release_year": int(query_row["release_year"]),
                "vote_average": float(query_row["vote_average"]),
            },
            "recommendations": recommendations,
            "total_found": len(recommendations),
            "filter_applied": genre_filter or "None",
        }
