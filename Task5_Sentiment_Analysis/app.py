"""
Customer Sentiment Analysis Platform - Web Application
CodeOrbit Tech AI Internship - Task 5: Sentiment Analysis on Text Data.

Interactive Streamlit interface featuring:
- Three-class sentiment classification: Positive, Neutral, and Negative
- Real-time single review analysis with sub-scores (compound, pos, neu, neg)
- Batch CSV review processing with sentiment distribution charts
- Downloadable CSV reports with predicted labels and scores
- Deep explanation of NLP preprocessing, VADER heuristics, and viva prep
"""

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyzer import SentimentAnalysisError, SentimentAnalyzer
from config_task5 import (
    DEFAULT_SAMPLE_CSV,
    INTERNSHIP_ORGANIZATION,
    INTERNSHIP_TASK_NAME,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    PROJECT_TITLE,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {INTERNSHIP_ORGANIZATION}",
    page_icon="💬",
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
        background: linear-gradient(135deg, #064E3B 0%, #065F46 50%, #047857 100%);
        color: white;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.12);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #A7F3D0;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .sentiment-positive {
        background: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
        border-radius: 8px;
        padding: 16px;
    }
    .sentiment-neutral {
        background: #F1F5F9;
        border: 1px solid #64748B;
        color: #334155;
        border-radius: 8px;
        padding: 16px;
    }
    .sentiment-negative {
        background: #FEF2F2;
        border: 1px solid #EF4444;
        color: #991B1B;
        border-radius: 8px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_analyzer():
    return SentimentAnalyzer()


analyzer = get_analyzer()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 CodeOrbit Tech")
    st.markdown(
        f'<span style="background:#ECFDF5; color:#065F46; padding:3px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">{INTERNSHIP_TASK_NAME}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.subheader("⚙️ Model Architecture")
    st.markdown(
        """
        - **Model:** VADER (Valence Aware Dictionary and sEntiment Reasoner)
        - **Class Ontology:** Positive, Neutral, Negative
        - **Compound Range:** -1.0 to +1.0
        - **Positive Rule:** $\\text{Compound} \\ge 0.05$
        - **Neutral Rule:** $-0.05 < \\text{Compound} < 0.05$
        - **Negative Rule:** $\\text{Compound} \\le -0.05$
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
        <h1 class="hero-title">💬 {PROJECT_TITLE}</h1>
        <p class="hero-subtitle">
            Natural Language Processing • Rule-Based & Lexicon Valence Modeling • Three-Class Sentiment Triage
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_single, tab_batch, tab_math, tab_interview = st.tabs([
    "🔍 Single Review Analysis",
    "📁 Batch CSV Processing",
    "🧠 NLP Pipeline & Methodology",
    "🎓 Internship Task Mapping & Viva Q&A",
])


# =========================================================
# TAB 1: Single Review Analysis
# =========================================================
with tab_single:
    st.subheader("1. Test an Individual Customer Review")

    col_input, col_out = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("##### Select Example Review or Type Custom Text:")
        preset_choice = st.selectbox(
            "Quick Preset Example:",
            options=[
                "Custom Text Input",
                "Positive: 'The laptop battery lasts 14 hours and the screen is gorgeous!'",
                "Neutral: 'The package arrived on Tuesday. The box was sealed.'",
                "Negative: 'Terrible customer support and the item broke on day one!'",
            ],
            index=0,
        )

        initial_text = ""
        if "Positive" in preset_choice:
            initial_text = "The laptop battery lasts 14 hours and the screen is gorgeous!"
        elif "Neutral" in preset_choice:
            initial_text = "The package arrived on Tuesday. The box was sealed."
        elif "Negative" in preset_choice:
            initial_text = "Terrible customer support and the item broke on day one!"

        user_review = st.text_area(
            "Customer Review Text:",
            value=initial_text,
            height=130,
            placeholder="Type or paste any product, service, or customer review here...",
        )

        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    with col_out:
        st.markdown("##### Sentiment Prediction & Polarity Scores:")

        if (analyze_btn or user_review) and user_review.strip():
            try:
                res = analyzer.analyze_single_text(user_review)
                sentiment = res["sentiment"]
                compound = res["compound_score"]

                css_class = (
                    "sentiment-positive"
                    if sentiment == LABEL_POSITIVE
                    else "sentiment-neutral"
                    if sentiment == LABEL_NEUTRAL
                    else "sentiment-negative"
                )
                emoji = "🟢" if sentiment == LABEL_POSITIVE else "⚪" if sentiment == LABEL_NEUTRAL else "🔴"

                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">
                            Predicted Sentiment
                        </div>
                        <div style="font-size: 1.8rem; font-weight: 800; margin: 4px 0;">
                            {emoji} {sentiment}
                        </div>
                        <div style="font-size: 0.9rem;">
                            <b>Compound Valence Score:</b> {compound:.4f} (Scale: -1.0 to +1.0)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("##### 📊 Valence Component Ratios")
                score_cols = st.columns(3)
                score_cols[0].metric("Positive Ratio", f"{res['positive_score'] * 100:.1f}%")
                score_cols[1].metric("Neutral Ratio", f"{res['neutral_score'] * 100:.1f}%")
                score_cols[2].metric("Negative Ratio", f"{res['negative_score'] * 100:.1f}%")

                # Compound gauge chart
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=compound,
                        title={"text": "<b>Normalized Compound Polarity</b>"},
                        gauge={
                            "axis": {"range": [-1.0, 1.0]},
                            "bar": {"color": "#10B981" if compound >= 0.05 else "#EF4444" if compound <= -0.05 else "#64748B"},
                            "steps": [
                                {"range": [-1.0, -0.05], "color": "#FEE2E2"},
                                {"range": [-0.05, 0.05], "color": "#F1F5F9"},
                                {"range": [0.05, 1.0], "color": "#DCFCE7"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            except SentimentAnalysisError as err:
                st.error(str(err))
        else:
            st.info("👈 Enter review text on the left and click 'Analyze Sentiment'.")


# =========================================================
# TAB 2: Batch CSV Processing
# =========================================================
with tab_batch:
    st.subheader("2. Batch Review CSV Processing & Visual Distribution")

    col_upload, col_actions = st.columns([2, 1])
    with col_upload:
        uploaded_csv = st.file_uploader(
            "Upload Customer Reviews CSV (must contain a text column):",
            type=["csv"],
        )
    with col_actions:
        st.markdown("##### Or load curated sample dataset:")
        load_sample = st.button("📂 Load Sample Reviews (15 Cases)", use_container_width=True)

    df_to_analyze = None
    if uploaded_csv is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
    elif load_sample:
        df_to_analyze = pd.read_csv(DEFAULT_SAMPLE_CSV)

    if df_to_analyze is not None:
        st.success(f"Loaded dataset with {len(df_to_analyze)} rows!")

        # Column selector
        text_cols = list(df_to_analyze.columns)
        default_col_idx = text_cols.index("review_text") if "review_text" in text_cols else 0
        selected_text_col = st.selectbox(
            "Select column containing review text:",
            options=text_cols,
            index=default_col_idx,
        )

        with st.spinner("Classifying sentiment across all rows..."):
            try:
                augmented_df, summary = analyzer.analyze_dataframe(
                    df_to_analyze, text_column=selected_text_col
                )

                # Summary KPIs
                kpi_cols = st.columns(5)
                kpi_cols[0].metric("Total Reviews", summary["total_reviews"])
                kpi_cols[1].metric("🟢 Positive", f"{summary['positive_count']} ({summary['positive_pct']}%)")
                kpi_cols[2].metric("⚪ Neutral", f"{summary['neutral_count']} ({summary['neutral_pct']}%)")
                kpi_cols[3].metric("🔴 Negative", f"{summary['negative_count']} ({summary['negative_pct']}%)")
                kpi_cols[4].metric("Avg Compound", summary["avg_compound"])

                # Distribution Chart
                dist_df = pd.DataFrame([
                    {"Sentiment": "Positive", "Count": summary["positive_count"], "Color": "#10B981"},
                    {"Sentiment": "Neutral", "Count": summary["neutral_count"], "Color": "#64748B"},
                    {"Sentiment": "Negative", "Count": summary["negative_count"], "Color": "#EF4444"},
                ])

                chart_cols = st.columns([1, 1])
                with chart_cols[0]:
                    pie_fig = px.pie(
                        dist_df,
                        names="Sentiment",
                        values="Count",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#EF4444"},
                        title="<b>Sentiment Share Breakdown</b>",
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                with chart_cols[1]:
                    bar_fig = px.bar(
                        dist_df,
                        x="Sentiment",
                        y="Count",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10B981", "Neutral": "#64748B", "Negative": "#EF4444"},
                        title="<b>Sentiment Volume Counts</b>",
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

                # Data Preview
                st.markdown("##### 📋 Analyzed Reviews Table Preview:")
                st.dataframe(augmented_df, use_container_width=True)

                # Download CSV
                csv_buffer = io.StringIO()
                augmented_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Analyzed Results CSV",
                    data=csv_buffer.getvalue(),
                    file_name="analyzed_customer_sentiment_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except SentimentAnalysisError as err:
                st.error(str(err))


# =========================================================
# TAB 3: NLP Methodology & Math
# =========================================================
with tab_math:
    st.subheader("🧠 VADER Sentiment Lexicon & Heuristic Methodology")

    st.markdown(
        """
        **VADER (Valence Aware Dictionary and sEntiment Reasoner)** is a gold-standard rule-based NLP sentiment model specifically calibrated for customer reviews and microblog texts.

        ### 1. Five Heuristic Rules of VADER
        1. **Punctuation Exclamation:** Exclamation marks amplify sentiment intensity without altering valence (*e.g.*, `"The food was great!"` vs `"The food was great!!!"`).
        2. **Capitalization:** ALL-CAPS words emphasize sentiment (*e.g.*, `"The movie was BAD"` vs `"The movie was bad"`).
        3. **Degree Modifiers (Intensifiers & Diminishers):** Modifiers like *"extremely"*, *"very"*, or *"barely"* shift valence (*e.g.*, `"extremely good"` vs `"barely tolerable"`).
        4. **Contrastive Conjunction 'BUT':** Shifts focus to the clause following the conjunction (*e.g.*, `"The food was delicious, but the service was awful"` yields negative overall polarity).
        5. **Negation Flipping:** Negation words (*e.g.*, *"not"*, *"never"*, *"hardly"*) invert the valence of preceding or succeeding words within a 3-token window (*e.g.*, `"not good"` flips from $+1.9$ to $-1.2$).
        """
    )

    st.markdown("### 2. Normalized Compound Score Formula")
    st.latex(r"c = \frac{x}{\sqrt{x^2 + \alpha}}")
    st.markdown(
        """
        Where:
        - $x$ is the sum of valence scores of all recognized words in the sentence.
        - $\\alpha$ is the normalization constant (default $= 15$).
        - $c$ is the resulting normalized compound score $\\in [-1.0, 1.0]$.
        """
    )


# =========================================================
# TAB 4: Internship Task Mapping & Viva Q&A
# =========================================================
with tab_interview:
    st.subheader("🎓 CodeOrbit Tech Internship Task 5 Guide & Viva Prep")

    st.markdown(
        """
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:8px; margin-bottom:14px;">
            <b>Internship Task:</b> Task 5 — Sentiment Analysis on Text Data.<br>
            <b>Internship Organization:</b> CodeOrbit Tech Two-Month AI Internship Program.<br>
            <b>Core Deliverables Completed:</b> Three-class sentiment categorization (Positive, Neutral, Negative), single review testing, batch CSV processing with Plotly visual charts, CSV export, and unit test suite.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Q1: How does VADER genuinely support Neutral sentiment?"):
        st.markdown(
            """
            **Answer:**
            Standard binary classifiers force every text into either 0 (negative) or 1 (positive), artificially misclassifying objective, matter-of-fact statements. VADER computes an unnormalized aggregate valence. When a statement contains purely factual language (*e.g.*, `"The package arrived on Tuesday"`), the compound score stays within $[-0.05, 0.05]$, correctly designating the review as **Neutral**.
            """
        )

    with st.expander("Q2: What are the main limitations of lexicon-based sentiment analysis?"):
        st.markdown(
            """
            **Answer:**
            1. **Sarcasm and Irony:** Phrases like *"Oh great, another flight delay"* have positive words (*great*) but convey heavy negative sentiment.
            2. **Domain-Specific Slang:** In video games, *"This weapon is sick"* is positive praise, but a generic medical lexicon might consider *"sick"* negative.
            3. **Multi-Aspect Reviews:** A review praising the screen but hating the battery requires Aspect-Based Sentiment Analysis (ABSA) rather than a single document-level score.
            """
        )

    with st.expander("Q3: தமிழ் / Tanglish Summary for Viva/Interview Quick Recall"):
        st.markdown(
            """
            > *"Intha Task 5-la customer reviews-ah Positive, Neutral, and Negative nu 3 categories-la classify pandrom using VADER. Single review analyze panni compound score (-1 to +1) calculate pannalaam, illana multiple reviews irukira CSV file upload panni batch analysis pannalaam. Factual sentences-ku neutral label kidaikum. Results ellaathaiyum interactive charts-la paarthutu download-um pannikalam."*
            """
        )
