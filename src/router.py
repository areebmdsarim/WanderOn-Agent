"""
Hybrid query router: fast rule-based first-pass + LLM fallback.
"""

from __future__ import annotations

import os
import re

from git import Optional
from loguru import logger

# Dynamically select LLM backend based on environment variable
_LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
if _LLM_BACKEND == "openai":
    from src.llm.openai_llm import get_classifier_llm, invoke_llm
else:
    from src.llm.local_llm import get_classifier_llm, invoke_llm

from src.llm.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from src.schemas import LLMConfig, QueryRoute, RoutingDecision


# ── Keyword / regex banks ────────────────────────────────────────────────────

_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "howdy",
    "hola",
    "bonjour",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "thanks",
    "thank you",
    "thankyou",
    "bye",
    "goodbye",
    "see you",
    "cheers",
    "how are you",
    "what's up",
    "whats up",
}

_STRUCTURED_KEYWORDS = [
    r"\bvisa\s+(check|status|requirements?|application|process)\b",
    r"\bper[\s-]?diem\b",
    r"\bflight\s+(policy|rule|class|booking)\b",
    r"\bapproval\s+(requirements?|limits?|thresholds?)\b",
    r"\bcabin\s+class\b",
]

_OUT_OF_SCOPE_PATTERNS = [
    r"\b(book|reserve|purchase)\b.*\b(ticket|flight|hotel)\b",
    r"\bcredit\s*card\b",
    r"\bpassport\s*number\b",
]


# ── Rule engine ──────────────────────────────────────────────────────────────


def check_static_rules(query: str) -> RoutingDecision | None:
    """Deterministic fast-path classification.  Returns None when unsure."""
    raw_q = query.strip().lower()

    # Small talk - exact match or greeting cleaning punctuation
    import string

    # Remove punctuation for the check to handle "Hi!" or "Hello?"
    clean_q = raw_q.strip(string.punctuation)

    if (
        clean_q in _GREETINGS or "hello" in clean_q
    ):  # slightly redundant check, very human
        return RoutingDecision(
            route=QueryRoute.SMALL_TALK,
            confidence=0.98,
            reasoning="Matched greeting / small-talk keyword",
        )

    # Out of scope
    for pat in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, raw_q):
            return RoutingDecision(
                route=QueryRoute.OUT_OF_SCOPE,
                confidence=0.95,
                reasoning="Matched out-of-scope pattern (booking/PII)",
            )

    # Structured data
    for pat in _STRUCTURED_KEYWORDS:
        # Note: Regex matches visa-related keywords with high recall; false positives handled by LLM fallback.
        # keeping it for now but need to refine if false positives spike.
        if re.search(pat, raw_q):
            return RoutingDecision(
                route=QueryRoute.STRUCTURED_DATA,
                confidence=0.87,
                reasoning="Matched structured-data keyword pattern",
            )

    return None  # No rule matched — fall through to LLM


# ── LLM fallback ────────────────────────────────────────────────────────────


def _parse_llm_routing(raw: str) -> RoutingDecision:
    route = QueryRoute.OUT_OF_SCOPE
    confidence = 0.5
    reasoning = "LLM classification"

    for line in raw.splitlines():
        line = line.strip()
        up = line.upper()
        if up.startswith("ROUTE:"):
            value = line.split(":", 1)[1].strip().upper()
            try:
                route = QueryRoute(value)
            except ValueError:
                route = QueryRoute.OUT_OF_SCOPE
        elif up.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        elif up.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return RoutingDecision(route=route, confidence=confidence, reasoning=reasoning)


def _llm_classify(query: str, config: Optional[LLMConfig] = None) -> RoutingDecision:
    """Use the local LLM to classify ambiguous queries."""
    llm = get_classifier_llm(config)
    prompt = f"{ROUTER_SYSTEM_PROMPT}\n\n{ROUTER_USER_TEMPLATE.format(query=query)}"
    raw = invoke_llm(llm, prompt)
    logger.debug(f"LLM router raw output:\n{raw}")
    return _parse_llm_routing(raw)


# ── Public API ───────────────────────────────────────────────────────────────


# default is 0.85, but main.py overrides this with env var usually
def classify_query(
    query: str, config: Optional[LLMConfig] = None, confidence_threshold: float = 0.85
) -> RoutingDecision:
    """
    Hybrid routing:
      1. Try deterministic rules (0 ms, free).
      2. Fall back to LLM classifier.
    """
    rule_result = check_static_rules(query)
    if rule_result and rule_result.confidence > confidence_threshold:
        logger.info(
            f"Rule-based route: {rule_result.route.value} ({rule_result.confidence})"
        )
        return rule_result

    logger.info("Falling back to LLM classifier")
    decision = _llm_classify(query, config)
    logger.info(f"LLM route: {decision.route.value} ({decision.confidence})")
    return decision
