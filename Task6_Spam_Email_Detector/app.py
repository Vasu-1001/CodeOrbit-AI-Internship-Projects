"""
AI Spam Email Detector - Web Application
CodeOrbit Tech AI Internship - Task 6: Mini Project — AI-Based Mini Application.

Interactive Streamlit interface featuring:
- Machine learning spam classification using Multinomial Naive Bayes & TF-IDF
- Single email classification with confidence scores and posterior probability breakdown
- Batch CSV email scanner with visual spam vs. ham analytics
- Downloadable classified CSV reports
- Real evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix)
- Comprehensive technical viva Q&A and Tamil/Tanglish summaries
"""

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from classifier import SpamClassifier, SpamClassifierError
from config_task6 import (
    DATASET_PATH,
    INTERNSHIP_ORGANIZATION,
    INTERNSHIP_TASK_NAME,
    LABEL_HAM,
    LABEL_SPAM,
    PROJECT_TITLE,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {INTERNSHIP_ORGANIZATION}",
    page_icon="🛡️",
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
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
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
        color: #C7D2FE;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .card-spam {
        background: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .card-ham {
        background: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained spam classifier model...")
def get_classifier():
    return SpamClassifier()


classifier = get_classifier()
metrics = classifier.metrics or {}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 CodeOrbit Tech")
    st.markdown(
        f'<span style="background:#EEF2FF; color:#4338CA; padding:3px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">{INTERNSHIP_TASK_NAME}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("⚙️ Model Pipeline")
    st.markdown(
        """
        - **Algorithm:** Multinomial Naive Bayes
        - **Feature Extractor:** TF-IDF (1, 2)-grams
        - **Smoothing ($\alpha$):** 0.5 (Laplace)
        - **Classes:** Spam vs. Legitimate
        """
    )

    if metrics:
        st.markdown("---")
        st.subheader("📈 Validated Test Metrics")
        st.write(f"**Accuracy:** {metrics.get('accuracy_pct', 'N/A')}")
        st.write(f"**Precision:** {metrics.get('precision_pct', 'N/A')}")
        st.write(f"**Recall:** {metrics.get('recall_pct', 'N/A')}")
        st.write(f"**F1-Score:** {metrics.get('f1_score_pct', 'N/A')}")

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
        <h1 class="hero-title">🛡️ {PROJECT_TITLE}</h1>
        <p class="hero-subtitle">
            Autonomous Machine Learning Classification • Naive Bayes Probabilistic Modeling • Phishing & Spam Defense
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_single, tab_batch, tab_eval, tab_interview = st.tabs([
    "📥 Live Email Scanner",
    "📁 Batch Email CSV Triage",
    "📊 Model Evaluation & Metrics",
    "🎓 Internship Task Mapping & Viva Q&A",
])


# =========================================================
# TAB 1: Live Email Scanner
# =========================================================
with tab_single:
    st.subheader("1. Test an Email or Message")

    col_input, col_out = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("##### Choose a Preset or Enter Custom Content:")
        preset = st.selectbox(
            "Quick Demo Examples:",
            options=[
                "Custom Message Input",
                "Spam Example: $1000 Walmart Gift Card Winner",
                "Spam Example: Urgent Account Suspension Notice",
                "Legitimate Example: Team Presentation Slide Review",
                "Legitimate Example: Meeting Up for Afternoon Coffee",
            ],
            index=0,
        )

        preset_text = ""
        if "Walmart" in preset:
            preset_text = "CONGRATULATIONS! You have been selected to win a $1000 Walmart Gift Card. Call 0800-298-8890 now to claim your prize! T&Cs apply."
        elif "Suspension" in preset:
            preset_text = "URGENT: Your account security has been compromised. Click here immediately to verify your password and avoid suspension."
        elif "Slide" in preset:
            preset_text = "Can you please review the attached slide deck before our team sync at 10 AM?"
        elif "Coffee" in preset:
            preset_text = "Hey, are we still meeting up for coffee tomorrow afternoon around 3 PM?"

        email_text = st.text_area(
            "Email / Message Content:",
            value=preset_text,
            height=140,
            placeholder="Paste email subject + body text here...",
        )

        scan_btn = st.button("🔍 Scan Message for Spam", type="primary", use_container_width=True)

    with col_out:
        st.markdown("##### Prediction & Probabilistic Breakdown:")

        if (scan_btn or email_text) and email_text.strip():
            try:
                res = classifier.predict_single(email_text)
                is_spam = res["is_spam"]
                pred = res["prediction"]

                card_class = "card-spam" if is_spam else "card-ham"
                icon = "🚨" if is_spam else "✅"

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">
                            Model Classification
                        </div>
                        <div style="font-size: 1.8rem; font-weight: 800; margin: 4px 0;">
                            {icon} {pred.upper()}
                        </div>
                        <div style="font-size: 0.9rem;">
                            <b>Confidence:</b> {res['confidence_pct']} • <b>Spam Probability:</b> {res['spam_pct']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Probabilities breakdown chart
                prob_fig = go.Figure(
                    go.Bar(
                        x=[res["legitimate_probability"] * 100, res["spam_probability"] * 100],
                        y=["Legitimate", "Spam"],
                        orientation="h",
                        text=[res["legitimate_pct"], res["spam_pct"]],
                        textposition="auto",
                        marker_color=["#10B981", "#EF4444"],
                    )
                )
                prob_fig.update_layout(
                    title="<b>Class Posterior Probabilities</b>",
                    xaxis_title="Probability (%)",
                    xaxis=dict(range=[0, 105]),
                    height=200,
                    margin=dict(l=20, r=20, t=35, b=20),
                )
                st.plotly_chart(prob_fig, use_container_width=True)

                if res["detected_vocabulary_terms"]:
                    st.info(f"💡 **Key Vocabulary Tokens Detected:** {', '.join(res['detected_vocabulary_terms'])}")

                st.caption(f"ℹ️ *{res['disclaimer']}*")

            except SpamClassifierError as err:
                st.error(str(err))
        else:
            st.info("👈 Enter message text on the left and click 'Scan Message for Spam'.")


# =========================================================
# TAB 2: Batch CSV Triage
# =========================================================
with tab_batch:
    st.subheader("2. Batch Email CSV Processing & Threat Distribution")

    col_up, col_sm = st.columns([2, 1])
    with col_up:
        uploaded_csv = st.file_uploader("Upload CSV containing email messages:", type=["csv"])
    with col_sm:
        st.markdown("##### Or load benchmark dataset:")
        load_bench = st.button("📂 Load Benchmark Dataset (100 Emails)", use_container_width=True)

    df_to_scan = None
    if uploaded_csv is not None:
        try:
            df_to_scan = pd.read_csv(uploaded_csv)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
    elif load_bench:
        df_to_scan = pd.read_csv(DATASET_PATH)

    if df_to_scan is not None:
        st.success(f"Loaded dataset with {len(df_to_scan)} records!")

        cols = list(df_to_scan.columns)
        default_idx = cols.index("text") if "text" in cols else 0
        selected_col = st.selectbox("Select column containing email text:", options=cols, index=default_idx)

        with st.spinner("Classifying emails..."):
            try:
                res_df, summary = classifier.predict_batch(df_to_scan, text_column=selected_col)

                m_cols = st.columns(4)
                m_cols[0].metric("Total Emails", summary["total_messages"])
                m_cols[1].metric("🚨 Spam Detected", f"{summary['spam_count']} ({summary['spam_pct']}%)")
                m_cols[2].metric("✅ Legitimate", f"{summary['legitimate_count']} ({summary['legitimate_pct']}%)")
                m_cols[3].metric("Spam Rate", f"{summary['spam_pct']}%")

                # Distribution Chart
                pie_df = pd.DataFrame([
                    {"Label": "Legitimate", "Count": summary["legitimate_count"]},
                    {"Label": "Spam", "Count": summary["spam_count"]},
                ])
                pie = px.pie(
                    pie_df,
                    names="Label",
                    values="Count",
                    color="Label",
                    color_discrete_map={"Legitimate": "#10B981", "Spam": "#EF4444"},
                    title="<b>Batch Email Triage Distribution</b>",
                )
                st.plotly_chart(pie, use_container_width=True)

                st.markdown("##### 📋 Scanned Emails Table Preview:")
                st.dataframe(res_df, use_container_width=True)

                csv_buf = io.StringIO()
                res_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download Triage Report CSV",
                    data=csv_buf.getvalue(),
                    file_name="spam_triage_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except SpamClassifierError as err:
                st.error(str(err))


# =========================================================
# TAB 3: Model Evaluation & Metrics
# =========================================================
with tab_eval:
    st.subheader("📊 Rigorous Model Evaluation & Zero-Data-Leakage Methodology")

    if metrics:
        st.markdown("### 1. Held-Out Test Set Performance (20% Split)")
        m_row = st.columns(4)
        m_row[0].metric("Accuracy", metrics.get("accuracy_pct", "N/A"))
        m_row[1].metric("Precision (Spam)", metrics.get("precision_pct", "N/A"))
        m_row[2].metric("Recall (Spam)", metrics.get("recall_pct", "N/A"))
        m_row[3].metric("F1-Score", metrics.get("f1_score_pct", "N/A"))

        st.markdown("### 2. Confusion Matrix")
        cm_data = metrics.get("confusion_matrix", {})
        cm_df = pd.DataFrame(
            [
                [cm_data.get("true_negatives_ham", 0), cm_data.get("false_positives_false_spam", 0)],
                [cm_data.get("false_negatives_missed_spam", 0), cm_data.get("true_positives_spam", 0)],
            ],
            index=["Actual Legitimate", "Actual Spam"],
            columns=["Predicted Legitimate", "Predicted Spam"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 3. Critical Real-World Evaluation: Precision vs. Recall Trade-Off")
    st.warning(
        """
        - **False Positive Risk (High Precision Priority):** A legitimate email (e.g., job offer, flight ticket, or password reset) falsely classified as spam can cause severe real-world harm. We tune the decision threshold and feature weights to maximize **Precision**.
        - **False Negative Risk:** A spam email that slips through to the inbox is an annoyance, but far less damaging than discarding an authentic high-priority email.
        """
    )


# =========================================================
# TAB 4: Internship Task Mapping & Viva Q&A
# =========================================================
with tab_interview:
    st.subheader("🎓 CodeOrbit Tech Internship Task 6 Guide & Viva Prep")

    st.markdown(
        """
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px; border-radius:8px; margin-bottom:14px;">
            <b>Internship Task:</b> Task 6 — Mini Project: AI-Based Mini Application.<br>
            <b>Internship Organization:</b> CodeOrbit Tech Two-Month AI Internship Program.<br>
            <b>Verified Core Requirements:</b> Documented labeled dataset (100 samples), train/test split without data leakage, scikit-learn Pipeline (TF-IDF + Naive Bayes), evaluation metrics, model persistence, and unit tests.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Q1: How did you prevent data leakage between training and testing?"):
        st.markdown(
            """
            **Answer:**
            Data leakage occurs when information from the test dataset influences feature extraction or model fitting. We prevent this by:
            1. Performing a stratified train/test split (80/20) before applying any text vectorization.
            2. Using a scikit-learn `Pipeline`, calling `fit()` exclusively on the training split so that vocabulary and inverse document frequencies are learned strictly from training data.
            3. Applying `predict()` and `predict_proba()` on the test split as an unseen evaluation benchmark.
            """
        )

    with st.expander("Q2: Why is Multinomial Naive Bayes well-suited for Spam Detection?"):
        st.markdown(
            """
            **Answer:**
            Multinomial Naive Bayes computes posterior class probabilities using Bayes' Theorem:
            $P(\\text{Spam} \\mid \\mathbf{x}) \\propto P(\\text{Spam}) \\prod_{i=1}^n P(x_i \\mid \\text{Spam})$.
            Even though the 'naive' assumption of feature independence is rarely strictly true, in practice, extreme tokens (*claim*, *urgent*, *winner*, *guaranteed*) provide overwhelming log-likelihood evidence, delivering high accuracy with minimal memory and sub-millisecond inference speeds.
            """
        )

    with st.expander("Q3: தமிழ் / Tanglish Summary for Viva/Interview Quick Recall"):
        st.markdown(
            """
            > *"Intha Task 6-la AI Spam Email Detector create pannirukom. Documented dataset-ah 80% train and 20% test-nu split panni, data leakage illama TF-IDF vectorizer and Multinomial Naive Bayes pipeline train pannom. Precision, Recall, F1-score, and Confusion Matrix calculate panni model-ah joblib moolama save pannom. Single email scanner, batch CSV triage, and CSV report download ellame Streamlit UI-la integrate aagi iruku."*
            """
        )
