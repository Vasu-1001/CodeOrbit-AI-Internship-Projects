"""
Rule-Based Conversational AI Assistant - Web Application
CodeOrbit Tech AI Internship - Task 1: Rule-Based Chatbot.

Interactive Streamlit interface featuring:
- Full conversational chat stream with memory across reruns
- Priority regex and keyword matching without external LLMs
- Instant suggestion chips for 1-click query evaluation
- Live Rule Inspector revealing matched regex/keywords and confidence
- Comprehensive Educational Viva guide on Rule-Based vs ML systems
- Chat export and reset controls
"""

import json
from typing import Any, Dict, List
import streamlit as st

from chatbot import MatchResult, RuleBasedChatbot
from config_task1 import (
    AUTHOR,
    INTERNSHIP_ORGANIZATION,
    INTERNSHIP_TASK_NAME,
    PROJECT_TITLE,
    PROJECT_VERSION,
    SUGGESTED_QUESTIONS,
    WELCOME_MESSAGE,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{PROJECT_TITLE} | {INTERNSHIP_ORGANIZATION}",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------
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
        padding: 24px 28px;
        margin-bottom: 20px;
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
        color: #C7D2FE;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
        margin-top: 8px;
        background-color: rgba(255, 255, 255, 0.15);
        color: #EEF2FF;
    }

    .metadata-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.82rem;
        color: #475569;
    }

    .stat-badge {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #3730A3;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chatbot Engine Singleton Initialization
# ---------------------------------------------------------------------------
@st.cache_resource
def get_bot_instance() -> RuleBasedChatbot:
    """Create and cache the rule-based chatbot engine."""
    return RuleBasedChatbot(random_seed=42)

bot = get_bot_instance()

# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "metadata": {
                "intent": "system_welcome",
                "match_type": "system",
                "matched_pattern": "INIT_GREETING",
                "confidence": 1.0,
                "category": "System",
            },
        }
    ]

if "query_input_val" not in st.session_state:
    st.session_state.query_input_val = ""

# Track session statistics
total_user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
fallbacks_count = sum(
    1 for m in st.session_state.messages
    if m.get("metadata", {}).get("match_type") == "fallback"
)
recognized_count = max(0, total_user_msgs - fallbacks_count)
accuracy_rate = (recognized_count / total_user_msgs * 100) if total_user_msgs > 0 else 100.0

# ---------------------------------------------------------------------------
# Sidebar Navigation & Diagnostics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/clouds/200/chat.png",
        width=85,
    )
    st.markdown(f"### {INTERNSHIP_TASK_NAME}")
    st.caption(f"{INTERNSHIP_ORGANIZATION} • Version {PROJECT_VERSION}")

    st.markdown("---")
    st.markdown("#### 📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Prompts", total_user_msgs)
        st.metric("Recognized", recognized_count)
    with col2:
        st.metric("Fallbacks", fallbacks_count)
        st.metric("Match Rate", f"{accuracy_rate:.0f}%")

    st.markdown("---")
    st.markdown("#### ⚙️ Diagnostic Settings")
    show_diagnostics = st.toggle("Show Live Rule Inspector", value=True, help="Display matched intent, pattern, and confidence per response.")

    st.markdown("---")
    st.markdown("#### 📂 Supported Knowledge Base")
    intents_summary = bot.get_supported_intents()
    with st.expander(f"📚 View All {len(intents_summary)} Intent Rules"):
        st.write("The bot operates on deterministic pattern-matching rules:")
        for it in intents_summary:
            st.markdown(
                f"- **{it['intent']}**  \n"
                f"  `Category:` {it['category']} | `Priority:` {it['priority']} | `Patterns:` {it['num_patterns']}"
            )

    st.markdown("---")
    st.markdown("#### 🧹 Controls & Actions")
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
                "metadata": {
                    "intent": "system_welcome",
                    "match_type": "system",
                    "matched_pattern": "INIT_GREETING",
                    "confidence": 1.0,
                    "category": "System",
                },
            }
        ]
        st.rerun()

    # Chat Transcript Download
    transcript_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="📥 Export Chat Transcript (JSON)",
        data=transcript_json,
        file_name="orbitbot_chat_transcript.json",
        mime="application/json",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <h1 class="hero-title">🤖 {PROJECT_TITLE}</h1>
        <p class="hero-subtitle">
            Deterministic Conversational Agent • Priority Regex & Heuristic Keyword Pattern Matcher
        </p>
        <div>
            <span class="badge-pill">🏢 {INTERNSHIP_ORGANIZATION}</span>
            <span class="badge-pill">📌 {INTERNSHIP_TASK_NAME}</span>
            <span class="badge-pill">⚡ 100% Deterministic</span>
            <span class="badge-pill">🔒 Zero External API / Offline Safe</span>
            <span class="badge-pill">🛡️ 26+ Intents</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_theory = st.tabs(["💬 Interactive Chat Studio", "📖 Architecture & Viva Theory"])

# ---------------------------------------------------------------------------
# Tab 1: Interactive Chat Studio
# ---------------------------------------------------------------------------
with tab_chat:
    # Quick Questions Bar
    st.markdown("##### 💡 Recommended Questions (Click to Ask):")
    cols = st.columns(4)
    quick_sample = SUGGESTED_QUESTIONS[:8]
    clicked_prompt: str = ""

    for idx, prompt_text in enumerate(quick_sample):
        with cols[idx % 4]:
            if st.button(prompt_text, key=f"quick_btn_{idx}", use_container_width=True):
                clicked_prompt = prompt_text

    st.markdown("---")

    # Render Conversation Stream
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            avatar = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if show_diagnostics and msg["role"] == "assistant" and "metadata" in msg:
                    meta = msg["metadata"]
                    conf_pct = int(meta.get("confidence", 0) * 100)
                    st.markdown(
                        f"""
                        <div class="metadata-card">
                            🎯 <strong>Intent:</strong> <code>{meta.get('intent')}</code> &nbsp;|&nbsp;
                            🏷️ <strong>Category:</strong> {meta.get('category')} &nbsp;|&nbsp;
                            ⚡ <strong>Match Type:</strong> <code>{meta.get('match_type')}</code> &nbsp;|&nbsp;
                            🔍 <strong>Rule:</strong> <code>{meta.get('matched_pattern')}</code> &nbsp;|&nbsp;
                            📊 <strong>Confidence:</strong> {conf_pct}%
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Chat Input Processing
    user_prompt = st.chat_input("Type your question here (e.g. 'What is GIL?', 'List vs Tuple', 'Tell me a joke')...")
    
    # Check if user submitted via text input OR clicked a quick button
    active_query = user_prompt or clicked_prompt

    if active_query:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": active_query})

        # Process with Chatbot Engine
        result: MatchResult = bot.find_intent(active_query)

        # Append Assistant Response with Metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.response,
            "metadata": {
                "intent": result.intent,
                "match_type": result.match_type,
                "matched_pattern": result.matched_pattern,
                "confidence": result.confidence,
                "category": result.category,
            },
        })

        st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Architecture & Viva Theory Guide
# ---------------------------------------------------------------------------
with tab_theory:
    st.markdown("### 🎓 Deep-Dive: Rule-Based Conversational AI Architecture")
    st.write(
        "This project adheres to the official CodeOrbit Tech AI Internship requirements by building "
        "a fully deterministic, explainable, and production-tested rule-based NLP system."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### ⚙️ Pipeline Execution Flow")
        st.markdown(
            """
            1. **Input Ingestion:** User query received via Streamlit UI.
            2. **Text Normalization:**
               - Case folding (`lower()`)
               - Whitespace collapsing and stripping
               - Punctuation removal (regex `[^\\w\\s]`) to prevent symbol mismatch
            3. **Empty Input Check:** Early-exit guarding against zero-length strings.
            4. **Priority Regex Scanning:** Pre-compiled compiled regex patterns sorted by priority descending.
            5. **Keyword/Subphrase Matching:** Word boundary checks (`\\b{keyword}\\b`) sorted by phrase length.
            6. **Deterministic Fallback:** Categorized out-of-domain response with contextual topic suggestions.
            """
        )

        st.markdown("#### ⚖️ Rule-Based AI vs Machine Learning LLMs")
        st.markdown(
            """
            | Feature | Rule-Based Chatbot | Modern LLM (e.g. GPT/Gemini) |
            | :--- | :--- | :--- |
            | **Execution Nature** | 100% Deterministic & Auditable | Probabilistic & Generative |
            | **Hallucination Risk** | **0% (Impossible)** | Non-zero |
            | **Compute / Latency** | Sub-millisecond CPU | High GPU latency & cost |
            | **External Dependencies**| Zero (Standard Python Library) | Paid APIs / Large weights |
            | **Auditability** | Direct mapping from rule to output | Opaque black-box attention |
            | **Best Use Cases** | Banking, Medical FAQ, IVR, Compliance | Creative writing, open exploration |
            """
        )

    with col_b:
        st.markdown("#### 🔬 Viva Voce Quick Revision")
        with st.expander("Q1: Why is priority ranking critical for rule-based systems?", expanded=True):
            st.write(
                "Without priority ranking, broad catch-all words (like 'python') would greedily capture "
                "specific queries (such as 'difference between list and tuple in Python'). Assigning "
                "specific compound questions a higher priority score (e.g. 60 vs 40) ensures specific intent "
                "resolution before broad keywords are evaluated."
            )

        with st.expander("Q2: How does regex boundary matching prevent false positive hits?"):
            st.write(
                "Using regex word boundaries (`\\b`) ensures that substrings do not accidentally trigger unrelated rules. "
                "For example, without `\\b`, the query 'I am looking for a **python** developer' could accidentally match "
                "rules containing partial syllables or words."
            )

        with st.expander("Q3: When are rule-based systems preferred in industry over LLMs?"):
            st.write(
                "Rule-based bots dominate mission-critical enterprise environments such as insurance claim routing, "
                "banking account transactions, healthcare triage, and safety checklists where hallucinations "
                "or non-deterministic answers carry severe financial or legal liability."
            )

        with st.expander("Q4: How does this bot handle out-of-vocabulary (OOV) queries?"):
            st.write(
                "When none of the priority patterns or keywords match, the engine smoothly triggers an honest fallback handler. "
                "Rather than crashing or giving misleading information, it explains that the query falls outside "
                "its defined rules and actively suggests valid topics the user can explore."
            )
