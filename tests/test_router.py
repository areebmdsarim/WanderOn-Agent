"""
Tests for the query router (rule-based + LLM parsing).
"""

import pytest

from src.router import check_static_rules, _parse_llm_routing, classify_query
from src.schemas import QueryRoute, RoutingDecision


# ── Rule-based routing tests ────────────────────────────────────────────────


class TestRuleBasedRouting:
    """Tests for the fast-path rule engine."""

    @pytest.mark.parametrize(
        "query",
        [
            "hi",
            "hello",
            "Hey",
            "good morning",
            "thanks",
            "thank you",
            "bye",
            "how are you",
        ],
    )
    def test_greetings_route_to_small_talk(self, query):
        result = check_static_rules(query)
        assert result is not None
        assert result.route == QueryRoute.SMALL_TALK
        assert result.confidence > 0.9

    @pytest.mark.parametrize(
        "query",
        [
            "book me a flight to London",
            "reserve a hotel in New York",
            "purchase a ticket to Mumbai",
        ],
    )
    def test_booking_routes_to_out_of_scope(self, query):
        result = check_static_rules(query)
        assert result is not None
        assert result.route == QueryRoute.OUT_OF_SCOPE
        assert result.confidence > 0.9

    @pytest.mark.parametrize(
        "query",
        [
            "do I need a visa for UK",
            "what is the per diem rate for London",
            "what is the flight policy for business class",
            "what are the approval requirements",
        ],
    )
    def test_structured_keywords_detected(self, query):
        result = check_static_rules(query)
        assert result is not None
        assert result.route == QueryRoute.STRUCTURED_DATA
        assert result.confidence > 0.85

    def test_ambiguous_query_returns_none(self):
        result = check_static_rules("Tell me about the company policies on travel")
        assert result is None  # Should fall through to LLM


# ── LLM output parsing tests ────────────────────────────────────────────────


class TestLLMParsing:

    def test_parse_valid_response(self):
        raw = "ROUTE: FACT_FROM_DOCS\nCONFIDENCE: 0.85\nREASONING: User asks about company policy"
        result = _parse_llm_routing(raw)
        assert result.route == QueryRoute.FACT_FROM_DOCS
        assert result.confidence == 0.85
        assert "company policy" in result.reasoning

    def test_parse_with_extra_whitespace(self):
        raw = "  ROUTE:  SMALL_TALK  \n  CONFIDENCE:  0.92  \n  REASONING:  Greeting detected  "
        result = _parse_llm_routing(raw)
        assert result.route == QueryRoute.SMALL_TALK
        assert result.confidence == 0.92

    def test_parse_invalid_route_defaults_to_out_of_scope(self):
        raw = "ROUTE: INVALID_ROUTE\nCONFIDENCE: 0.5\nREASONING: Unknown"
        result = _parse_llm_routing(raw)
        assert result.route == QueryRoute.OUT_OF_SCOPE

    def test_parse_invalid_confidence_defaults(self):
        raw = "ROUTE: SMALL_TALK\nCONFIDENCE: not_a_number\nREASONING: Test"
        result = _parse_llm_routing(raw)
        assert result.confidence == 0.5  # default

    def test_parse_confidence_clamped(self):
        raw = "ROUTE: SMALL_TALK\nCONFIDENCE: 1.5\nREASONING: Over-confident"
        result = _parse_llm_routing(raw)
        assert result.confidence == 1.0

    def test_parse_empty_string(self):
        result = _parse_llm_routing("")
        assert result.route == QueryRoute.OUT_OF_SCOPE
        assert result.confidence == 0.5
