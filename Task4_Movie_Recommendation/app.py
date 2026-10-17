"""
Smart Movie Recommendation System - Web Application
CodeOrbit Tech AI Internship - Task 4: Simple Recommendation System.

Interactive Streamlit interface featuring:
- Content-Based Filtering using TF-IDF Vectorization and Cosine Similarity
- Search by Title with fuzzy matching and catalog dropdown
- Real-time genre filtering and Top-N recommendation control
- Explainable AI recommendations ("Why this was recommended")
- Technical architecture explanation and technical interview viva prep
"""

import streamlit as st
import pandas as pd

from config_task4 import (
    ALL_GENRES,
    DEFAULT_TOP_N,
    INTERNSHIP_ORGANIZATION,
    INTERNSHIP_TASK_NAME,
    MAX_TOP_N,
    PROJECT_TITLE,
)
from recommender import MovieNotFoundError, MovieRecommender


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {INTERNSHIP_ORGANIZATION}",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        color: white;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .movie-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .dark .movie-card {
        background: #1E293B;
        border-color: #334155;
    }

    .genre-tag {
        display: inline-block;
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        border-radius: 9999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }

    .reason-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.8rem;
        color: #475569;
        margin-top: 10px;
    }
    .dark .reason-box {
        background-color: #0F172A;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Recommender Instance
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing TF-IDF vector matrix and movie database...")
def get_recommender():
    return MovieRecommender()


recommender = get_recommender()
movie_df = recommender.df

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 CodeOrbit Tech")
    st.markdown(
        f'<span style="background:#EFF6FF; color:#1D4ED8; padding:3px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">{INTERNSHIP_TASK_NAME}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("⚙️ Filter & Controls")
    genre_filter = st.selectbox(
        "Filter Recommendations by Genre:",
        options=["All Genres"] + ALL_GENRES,
        index=0,
    )

    top_n_val = st.slider(
        "Number of Recommendations:",
        min_value=1,
        max_value=MAX_TOP_N,
        value=DEFAULT_TOP_N,
        step=1,
    )

    st.markdown("---")
    st.subheader("📊 Dataset Statistics")
    st.markdown(
        f"""
        - **Total Movies:** {len(movie_df)}
        - **Algorithm:** TF-IDF + Cosine Similarity
        - **Feature Soup:** Overview, Genres, Director, Keywords
        - **N-Gram Range:** (1, 2)
        - **Vocabulary Terms:** {recommender.tfidf_matrix.shape[1]:,} features
        """
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #94A3B8; text-align: center;">
            Two-Month AI Internship | CodeOrbit Tech<br>
            Developed by AI/ML Engineering Intern
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <h1 class="hero-title">🎬 {PROJECT_TITLE}</h1>
        <p class="hero-subtitle">
            Content-Based Filtering • TF-IDF Semantic Feature Modeling • Cosine Similarity Ranking • Explainable AI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_rec, tab_catalog, tab_math, tab_interview = st.tabs([
    "🎯 Movie Recommender",
    "📚 Browse Movie Catalog",
    "📐 Algorithm & Math Deep Dive",
    "🎓 Internship Task Mapping & Viva Q&A",
])


# =========================================================
# TAB 1: Movie Recommender
# =========================================================
with tab_rec:
    col_search, col_results = st.columns([1, 1], gap="large")

    with col_search:
        st.subheader("1. Select or Search Movie")

        search_mode = st.radio(
            "Movie Selection Mode:",
            options=["Select from Catalog Dropdown", "Type Custom Search Query"],
            horizontal=True,
        )

        selected_title = ""
        if search_mode == "Select from Catalog Dropdown":
            sorted_titles = sorted(recommender.all_titles)
            default_idx = sorted_titles.index("Inception") if "Inception" in sorted_titles else 0
            selected_title = st.selectbox(
                "Pick a reference film:",
                options=sorted_titles,
                index=default_idx,
            )
        else:
            search_query = st.text_input(
                "Enter movie title (handles typos and partial matches):",
                value="Interstellar",
                placeholder="e.g. Dark Knight, Matrix, Titanic...",
            )
            selected_title = search_query

        # Preview selected reference movie
        if selected_title:
            matched_title, idx, suggestions = recommender.find_movie_match(selected_title)

            if matched_title is not None and idx is not None:
                row = movie_df.iloc[idx]
                st.markdown(
                    f"""
                    <div style="background: #F1F5F9; border-left: 4px solid #0EA5E9; border-radius: 8px; padding: 14px 18px; margin-top: 12px;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #0369A1; text-transform: uppercase;">
                            Reference Movie Selected
                        </div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0F172A; margin: 4px 0;">
                            {row['title']} <span style="font-size: 0.9rem; font-weight: 500; color: #64748B;">({row['release_year']})</span>
                        </div>
                        <div style="margin: 6px 0;">
                            {''.join([f'<span class="genre-tag">{g}</span>' for g in row['genres'].split()])}
                        </div>
                        <div style="font-size: 0.85rem; color: #334155; margin-top: 6px;">
                            <b>Director:</b> {row['director']} • <b>IMDb Rating:</b> ⭐ {row['vote_average']}/10
                        </div>
                        <p style="font-size: 0.85rem; color: #475569; margin-top: 8px; line-height: 1.4;">
                            {row['overview']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif selected_title.strip():
                st.warning(f"⚠️ Movie '{selected_title}' not found.")
                if suggestions:
                    st.info(f"💡 Did you mean: **{', '.join(suggestions)}**?")

    with col_results:
        st.subheader("2. Recommended Movies")

        if selected_title and matched_title is not None:
            try:
                results = recommender.get_recommendations(
                    matched_title,
                    top_n=top_n_val,
                    genre_filter=genre_filter,
                )
                recs = results["recommendations"]

                if not recs:
                    st.warning(f"No recommendations found matching genre filter '{genre_filter}'. Try selecting 'All Genres'.")
                else:
                    st.success(f"Found {len(recs)} highly similar movies for **{matched_title}**!")

                    for r in recs:
                        sim_pct = r["similarity_score"] * 100
                        meter_color = "#10B981" if sim_pct >= 30 else "#3B82F6" if sim_pct >= 15 else "#64748B"

                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div>
                                        <span style="font-size: 1.15rem; font-weight: 800; color: #0F172A;">{r['title']}</span>
                                        <span style="font-size: 0.85rem; color: #64748B; margin-left: 6px;">({r['release_year']})</span>
                                    </div>
                                    <div style="background: {meter_color}15; border: 1px solid {meter_color}40; border-radius: 6px; padding: 2px 8px; font-weight: 700; color: {meter_color}; font-size: 0.85rem;">
                                        {r['similarity_percentage']} Match
                                    </div>
                                </div>
                                <div style="margin: 6px 0;">
                                    {''.join([f'<span class="genre-tag">{g}</span>' for g in r['genres'].split()])}
                                    <span style="font-size: 0.8rem; color: #64748B; margin-left: 6px;">⭐ {r['vote_average']}/10</span>
                                    <span style="font-size: 0.8rem; color: #64748B; margin-left: 6px;">• Directed by {r['director']}</span>
                                </div>
                                <p style="font-size: 0.82rem; color: #475569; margin: 6px 0; line-height: 1.4;">
                                    {r['overview']}
                                </p>
                                <div class="reason-box">
                                    <b>💡 Why Recommended:</b> {r['explanation']['summary_reason']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except MovieNotFoundError as err:
                st.error(str(err))
        else:
            st.info("👈 Please select or search for a movie on the left.")


# =========================================================
# TAB 2: Browse Catalog
# =========================================================
with tab_catalog:
    st.subheader("📚 Curated Movie Database Catalog")
    st.dataframe(
        movie_df[["movie_id", "title", "release_year", "genres", "director", "vote_average", "overview"]],
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TAB 3: Algorithm & Math Deep Dive
# =========================================================
with tab_math:
    st.subheader("📐 Content-Based Recommendation Architecture & Mathematics")

    st.markdown("### 1. TF-IDF (Term Frequency - Inverse Document Frequency)")
    st.markdown(
        """
        To transform unformatted movie overviews, genre strings, and cast tags into quantifiable numerical vectors, we use **TF-IDF Vectorization**:
        """
    )
    st.latex(r"\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1")

    st.markdown("### 2. Cosine Similarity Formula")
    st.markdown(
        """
        The angle between two sparse TF-IDF vectors $\mathbf{u}$ and $\mathbf{v}$ determines their directional alignment, independent of text length:
        """
    )
    st.latex(r"\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}")

    st.markdown("---")
    st.markdown("### 3. Content-Based vs. Collaborative Filtering Comparison")
    comp_df = pd.DataFrame([
        {
            "Feature": "Core Premise",
            "Content-Based (This Project)": "Recommends items similar to past item attributes",
            "Collaborative Filtering": "Recommends items based on user community rating patterns",
        },
        {
            "Feature": "Data Needed",
            "Content-Based (This Project)": "Item metadata only (genres, plots, directors)",
            "Collaborative Filtering": "Large user-item rating matrix ($R_{m \\times n}$)",
        },
        {
            "Feature": "Cold Start (New User)",
            "Content-Based (This Project)": "Works immediately; only needs 1 movie choice",
            "Collaborative Filtering": "Fails until user rates many items",
        },
        {
            "Feature": "Cold Start (New Item)",
            "Content-Based (This Project)": "Can be recommended immediately using its metadata",
            "Collaborative Filtering": "Cannot be recommended until other users rate it",
        },
        {
            "Feature": "Domain Independence",
            "Content-Based (This Project)": "Requires domain features (text parsing)",
            "Collaborative Filtering": "Domain-free; only relies on numeric ratings",
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# =========================================================
# TAB 4: Internship Task Mapping & Viva Q&A
# =========================================================
with tab_interview:
    st.subheader("🎓 CodeOrbit Tech Internship Task 4 Guide & Viva Prep")

    st.markdown(
        """
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:8px; margin-bottom:14px;">
            <b>Internship Task:</b> Task 4 — Simple Recommendation System.<br>
            <b>Internship Organization:</b> CodeOrbit Tech Two-Month AI Internship Program.<br>
            <b>Verified Core Requirements:</b> Documented dataset, content-based TF-IDF + Cosine Similarity, fuzzy title lookup, genre filter, explainability rationale, and unit test suite.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Q1: Why did you choose Content-Based Filtering over Collaborative Filtering for this task?"):
        st.markdown(
            """
            **Answer:**
            Content-Based Filtering is optimal when user rating interaction history is unavailable. Collaborative Filtering requires thousands of user ratings to construct a dense user-item interaction matrix and suffers from severe cold-start problems for new users and new items. Content-Based Filtering builds on item metadata (genres, director, synopsis), enabling immediate recommendations with zero user privacy concerns.
            """
        )

    with st.expander("Q2: Why use Cosine Similarity instead of Euclidean Distance?"):
        st.markdown(
            """
            **Answer:**
            Euclidean distance measures absolute distance between coordinates, which is distorted by document length (a movie with a long 5-paragraph plot would have higher magnitude coordinates than a 1-sentence plot). Cosine similarity measures the angle between vectors, normalizing for document length and focusing purely on the relative proportion of shared vocabulary.
            """
        )

    with st.expander("Q3: How do you handle typos in user searches?"):
        st.markdown(
            """
            **Answer:**
            We implement a 3-tier lookup pipeline:
            1. Exact case-insensitive title dictionary lookup ($O(1)$).
            2. Case-insensitive substring match.
            3. Fuzzy pattern matching using Gestalt pattern matching (`difflib.get_close_matches`), suggesting the closest candidate titles if an exact match isn't found.
            """
        )

    with st.expander("Q4: தமிழ் / Tanglish Summary for Viva/Interview Quick Recall"):
        st.markdown(
            """
            > *"Intha Task 4-la naanga Content-Based Movie Recommendation System build pannirukom. Movie titles, genres, plot overview, director, and keywords ellathaiyum combine panni oru 'metadata soup' create pandrom. Athuku TF-IDF vectorizer apply panni, Cosine Similarity formula moolama movies-ku idaila similarity compute pandrom.*
            > *User oru movie select pannapo, athuku closest matches with similarity percentage and 'Why Recommended' explanation clean-ah varum. Typos vanthalum difflib fuzzy matching vachu correct title suggest pannum."*
            """
        )
