"""
Automated Unit Tests for Rule-Based Chatbot.
CodeOrbit Tech AI Internship - Task 1.

Verifies:
1. Intent recognition for greetings, farewells, gratitude, identity, and FAQs.
2. Case-insensitivity and punctuation insensitivity.
3. Whitespace normalization and empty input handling.
4. Correct fallback handling for unsupported queries.
5. Priority-based disambiguation for overlapping patterns.
6. Minimum of 20 distinct intents present in the knowledge base.
"""

import pytest
from chatbot import RuleBasedChatbot
from config_task1 import INTENTS, EMPTY_INPUT_RESPONSE


@pytest.fixture
def bot() -> RuleBasedChatbot:
    """Fixture to provide a deterministic chatbot instance."""
    return RuleBasedChatbot(random_seed=42)


# ---------------------------------------------------------------------------
# 1. Structural and Knowledge Base Requirements
# ---------------------------------------------------------------------------
def test_intents_count_exceeds_twenty():
    """Verify that the chatbot includes at least 20 well-organized intents."""
    assert len(INTENTS) >= 20, f"Expected at least 20 intents, found {len(INTENTS)}"


def test_each_intent_has_required_schema():
    """Verify that every intent contains category, priority, patterns/keywords, and responses."""
    for name, data in INTENTS.items():
        assert "priority" in data, f"Intent {name} missing priority"
        assert "category" in data, f"Intent {name} missing category"
        assert ("patterns" in data or "keywords" in data), f"Intent {name} has neither patterns nor keywords"
        assert "responses" in data, f"Intent {name} missing responses"
        assert len(data["responses"]) > 0, f"Intent {name} has empty responses"


# ---------------------------------------------------------------------------
# 2. Text Normalization Tests
# ---------------------------------------------------------------------------
def test_text_normalization(bot: RuleBasedChatbot):
    """Test lowercase, punctuation removal, and whitespace collapse."""
    raw = "   HELLO,   World!!!  What's   Up???   "
    cleaned = bot.normalize_text(raw)
    assert cleaned == "hello world what s up"


def test_empty_and_whitespace_input(bot: RuleBasedChatbot):
    """Test that empty or pure whitespace input triggers the empty input handler."""
    for empty_val in ["", "   ", "\t\n", None]:
        res = bot.find_intent(empty_val)  # type: ignore
        assert res.intent == "empty_input"
        assert res.response == EMPTY_INPUT_RESPONSE
        assert res.match_type == "empty"


# ---------------------------------------------------------------------------
# 3. Conversational Intents (Greetings, Goodbyes, Thanks, Identity)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phrase", [
    "hello",
    "Hi there!",
    "HEY!!",
    "Good morning bot",
    "howdy partner",
])
def test_greeting_intent(bot: RuleBasedChatbot, phrase: str):
    """Test various greeting expressions."""
    res = bot.find_intent(phrase)
    assert res.intent == "greeting"
    assert len(res.response) > 0


@pytest.mark.parametrize("phrase", [
    "bye",
    "goodbye!",
    "see you later",
    "farewell friend",
])
def test_goodbye_intent(bot: RuleBasedChatbot, phrase: str):
    """Test various farewell expressions."""
    res = bot.find_intent(phrase)
    assert res.intent == "goodbye"


@pytest.mark.parametrize("phrase", [
    "thanks",
    "Thank you very much!",
    "thx a lot",
    "appreciate it",
])
def test_thanks_intent(bot: RuleBasedChatbot, phrase: str):
    """Test gratitude expressions."""
    res = bot.find_intent(phrase)
    assert res.intent == "thanks"


def test_bot_identity_and_creator(bot: RuleBasedChatbot):
    """Test bot self-identification and creator inquiries."""
    id_res = bot.find_intent("who are you?")
    assert id_res.intent == "bot_identity"
    assert "OrbitBot" in id_res.response

    creator_res = bot.find_intent("who created you?")
    assert creator_res.intent == "bot_creator"
    assert "CodeOrbit Tech" in creator_res.response


def test_help_and_capabilities(bot: RuleBasedChatbot):
    """Test help menu and capabilities intent."""
    res = bot.find_intent("help me, what can you do?")
    assert res.intent == "help_capabilities"
    assert "What I Can Do" in res.response


# ---------------------------------------------------------------------------
# 4. Technical FAQ Intents (Python, AI/ML, CS)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected_intent", [
    ("What is Python?", "faq_python_definition"),
    ("Why use Python and what are its features?", "faq_python_features"),
    ("Explain the difference between list and tuple", "faq_python_lists_tuples"),
    ("What is the GIL in Python?", "faq_python_gil"),
    ("What is a virtual environment in Python?", "faq_python_virtualenv"),
    ("What is a lambda function?", "faq_python_lambda"),
    ("What are OOP concepts in Python?", "faq_python_oop"),
    ("What is artificial intelligence?", "faq_ai_definition"),
    ("What is the difference between AI and ML?", "faq_ai_vs_ml"),
    ("Explain supervised vs unsupervised learning", "faq_supervised_unsupervised"),
    ("What is overfitting and how do you prevent it?", "faq_overfitting"),
    ("What is deep learning and neural networks?", "faq_deep_learning"),
    ("Rule based chatbot vs machine learning", "faq_rule_based_vs_ml"),
    ("What is Big O time complexity?", "faq_time_complexity"),
    ("Difference between Git and GitHub", "faq_git_github"),
])
def test_technical_faq_intents(bot: RuleBasedChatbot, query: str, expected_intent: str):
    """Verify accurate detection of technical and domain FAQs."""
    res = bot.find_intent(query)
    assert res.intent == expected_intent, f"Query '{query}' resolved to '{res.intent}', expected '{expected_intent}'"


# ---------------------------------------------------------------------------
# 5. Case-Insensitivity & Punctuation Invariance
# ---------------------------------------------------------------------------
def test_case_insensitivity_and_punctuation(bot: RuleBasedChatbot):
    """Ensure arbitrary casing and heavy punctuation do not break matching."""
    queries = [
        "wHaT iS pYtHoN???",
        "WHAT IS PYTHON!",
        "what... is... python??!!",
    ]
    for q in queries:
        res = bot.find_intent(q)
        assert res.intent == "faq_python_definition", f"Failed on '{q}'"


# ---------------------------------------------------------------------------
# 6. Overlapping Rules and Priority Disambiguation
# ---------------------------------------------------------------------------
def test_overlapping_rules_priority(bot: RuleBasedChatbot):
    """
    Ensure specific multi-word questions take priority over broad keywords.
    E.g., 'difference between list and tuple' mentions 'list' and 'tuple',
    it must trigger 'faq_python_lists_tuples' rather than a generic intent.
    """
    res1 = bot.find_intent("What is the difference between a list and a tuple in Python?")
    assert res1.intent == "faq_python_lists_tuples"

    res2 = bot.find_intent("Can you explain AI vs ML?")
    assert res2.intent == "faq_ai_vs_ml"

    res3 = bot.find_intent("Who developed you for CodeOrbit Tech?")
    assert res3.intent == "bot_creator"


# ---------------------------------------------------------------------------
# 7. Fallback Recovery for Unsupported Queries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unknown_query", [
    "quantum teleportation in astrophysics",
    "recipe for chocolate strawberry cheesecake",
    "asdfqwerty12345!@#$",
    "who won the 1994 world cup final",
])
def test_fallback_for_unsupported_queries(bot: RuleBasedChatbot, unknown_query: str):
    """Ensure out-of-domain queries return polite fallback responses without crashing."""
    res = bot.find_intent(unknown_query)
    assert res.intent == "unknown_fallback"
    assert res.match_type == "fallback"
    assert res.confidence == 0.0
    assert "rule-based" in res.response.lower() or "knowledge base" in res.response.lower()
