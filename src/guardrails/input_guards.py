"""
Input guardrails — non-prompt safety checks applied BEFORE routing.
"""

from __future__ import annotations

import re
from typing import Tuple

from loguru import logger


class InputGuardrails:
    """Layered input validation: length, PII, injection patterns."""

    MAX_QUERY_LENGTH = 10_000

    # ── PII patterns ────────────────────────────────────────────────────────
    PII_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),  # US SSN
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit_card"),  # credit card
        (r"\b[A-Z]{1,2}\d{7,8}\b", "passport_number"),  # passport-like
        (r"\b\d{12}\b", "aadhaar"),  # Aadhaar-like
    ]

    # ── Prompt-injection patterns ───────────────────────────────────────────
    INJECTION_PATTERNS = [
        r"(ignore|disregard|forget)\s+(previous|above|prior|all)\s+(instructions|prompts|rules)",
        r"system\s*prompt",
        r"you\s+are\s+now\s+",
        r"act\s+as\s+if",
        r"pretend\s+(you|to)\s+",
        r"reveal\s+(your|the)\s+(system|hidden|secret)",
    ]

    def check(self, query: str) -> Tuple[bool, str]:
        """
        Returns (passed: bool, reason: str).
        If passed is False the request should be refused immediately.
        """
        # 1. Empty / whitespace
        if not query or not query.strip():
            return False, "empty_query"

        # 2. Length
        if len(query) > self.MAX_QUERY_LENGTH:
            return False, "query_too_long"

        query_lower = query.lower()

        # 3. PII
        for pattern, label in self.PII_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"PII detected ({label})")
                return False, f"blocked_pii:{label}"

        # 4. Injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning("Prompt injection attempt detected")
                return False, "prompt_injection"

        return True, "ok"


class ConfidenceGuardrail:
    """Reject routing decisions below the confidence threshold."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def check(self, confidence: float) -> Tuple[bool, str]:
        if confidence < self.threshold:
            return False, f"low_confidence:{confidence:.2f}"
        return True, "ok"


class TokenBudgetGuardrail:
    """Refuse excessively long queries that would blow the token budget."""

    def __init__(self, max_words: int = 2000):
        self.max_words = max_words

    def check(self, query: str) -> Tuple[bool, str]:
        word_count = len(query.split())
        if word_count > self.max_words:
            return False, f"token_budget_exceeded:{word_count}"
        return True, "ok"
