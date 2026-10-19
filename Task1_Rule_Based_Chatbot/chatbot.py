"""
Rule-Based Chatbot Engine.
CodeOrbit Tech AI Internship - Task 1.

Implements input normalization, priority-ranked regex pattern matching,
keyword heuristics, fallback recovery, and response generation.
"""

from dataclasses import dataclass
import random
import re
from typing import Any, Dict, List, Optional

try:
    from config_task1 import (
        EMPTY_INPUT_RESPONSE,
        FALLBACK_RESPONSES,
        INTENTS,
        SUGGESTED_QUESTIONS,
        WELCOME_MESSAGE,
    )
except ImportError:
    from .config_task1 import (  # type: ignore
        EMPTY_INPUT_RESPONSE,
        FALLBACK_RESPONSES,
        INTENTS,
        SUGGESTED_QUESTIONS,
        WELCOME_MESSAGE,
    )


@dataclass
class MatchResult:
    """Encapsulates intent matching result and explanation metadata."""
    intent: str
    response: str
    match_type: str  # "regex", "keyword", "empty", or "fallback"
    matched_pattern: str
    confidence: float
    category: str


class RuleBasedChatbot:
    """
    Modular Rule-Based Conversational Engine.
    
    Uses deterministic pattern matching, regex compilation, and keyword rules
    to classify user intent and generate contextual responses.
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """Initialize the chatbot engine and pre-compile regular expressions."""
        self._rng = random.Random(random_seed)
        self.intents = INTENTS
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for each intent for optimal runtime performance."""
        for intent_name, data in self.intents.items():
            compiled_list: List[re.Pattern] = []
            for pat in data.get("patterns", []):
                try:
                    compiled_list.append(re.compile(pat, re.IGNORECASE))
                except re.error:
                    continue
            self._compiled_patterns[intent_name] = compiled_list

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """
        Normalize input text:
        1. Strip leading and trailing whitespace.
        2. Convert to lowercase.
        3. Replace punctuation (except internal alphanumeric) with single space.
        4. Collapse multiple spaces into a single space.
        """
        if not text:
            return ""
        # Lowercase
        normalized = text.strip().lower()
        # Replace non-alphanumeric characters with spaces
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        # Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def find_intent(self, user_input: str) -> MatchResult:
        """
        Classify user input into an intent using priority-ranked matching.
        
        Priority order:
        1. Empty input validation.
        2. Priority-sorted regex matching (high priority intents tested first).
        3. Priority-sorted keyword/phrase matching.
        4. Deterministic fallback message.
        """
        if user_input is None or not user_input.strip():
            return MatchResult(
                intent="empty_input",
                response=EMPTY_INPUT_RESPONSE,
                match_type="empty",
                matched_pattern="<whitespace_or_empty>",
                confidence=0.0,
                category="System",
            )

        raw_clean = user_input.strip()
        normalized = self.normalize_text(raw_clean)

        # Sort intents by priority descending (higher number = higher priority)
        sorted_intents = sorted(
            self.intents.items(),
            key=lambda item: item[1].get("priority", 0),
            reverse=True,
        )

        # Phase 1: Regex Pattern Matching (Highest Precision)
        for intent_name, data in sorted_intents:
            patterns = self._compiled_patterns.get(intent_name, [])
            for pat in patterns:
                # Test against both original text (for punctuation-dependent regex) and normalized text
                if pat.search(raw_clean) or pat.search(normalized):
                    responses = data.get("responses", [])
                    chosen_response = self._rng.choice(responses) if responses else ""
                    return MatchResult(
                        intent=intent_name,
                        response=chosen_response,
                        match_type="regex",
                        matched_pattern=pat.pattern,
                        confidence=1.0,
                        category=data.get("category", "General"),
                    )

        # Phase 2: Keyword and Phrase Matching
        for intent_name, data in sorted_intents:
            keywords = data.get("keywords", [])
            # Sort keywords by length descending so longer, more specific phrases match first
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            for kw in sorted_keywords:
                norm_kw = self.normalize_text(kw)
                # Word-boundary check or subphrase containment
                kw_pattern = rf"\b{re.escape(norm_kw)}\b"
                if re.search(kw_pattern, normalized):
                    responses = data.get("responses", [])
                    chosen_response = self._rng.choice(responses) if responses else ""
                    return MatchResult(
                        intent=intent_name,
                        response=chosen_response,
                        match_type="keyword",
                        matched_pattern=kw,
                        confidence=0.85,
                        category=data.get("category", "General"),
                    )

        # Phase 3: Fallback Recovery
        fallback_msg = self._rng.choice(FALLBACK_RESPONSES)
        return MatchResult(
            intent="unknown_fallback",
            response=fallback_msg,
            match_type="fallback",
            matched_pattern="<no_rule_match>",
            confidence=0.0,
            category="Fallback",
        )

    def respond(self, user_input: str) -> str:
        """Convenience method returning the text response."""
        result = self.find_intent(user_input)
        return result.response

    def get_response_with_metadata(self, user_input: str) -> Dict[str, Any]:
        """Return response and diagnostic metadata as a dictionary."""
        result = self.find_intent(user_input)
        return {
            "intent": result.intent,
            "response": result.response,
            "match_type": result.match_type,
            "matched_pattern": result.matched_pattern,
            "confidence": result.confidence,
            "category": result.category,
        }

    def get_supported_intents(self) -> List[Dict[str, Any]]:
        """Return the complete intent directory for documentation and testing."""
        directory: List[Dict[str, Any]] = []
        for name, data in self.intents.items():
            directory.append({
                "intent": name,
                "category": data.get("category", "General"),
                "priority": data.get("priority", 0),
                "num_keywords": len(data.get("keywords", [])),
                "num_patterns": len(data.get("patterns", [])),
                "num_responses": len(data.get("responses", [])),
            })
        return sorted(directory, key=lambda x: (x["category"], -x["priority"]))

    def get_welcome_message(self) -> str:
        """Return the initial greeting message."""
        return WELCOME_MESSAGE

    def get_suggested_questions(self) -> List[str]:
        """Return list of quick-start suggestion prompts."""
        return SUGGESTED_QUESTIONS
