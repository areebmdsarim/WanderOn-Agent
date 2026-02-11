"""
Schema validation tests — ensure all Pydantic models work correctly.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    FeedbackRequest,
    QueryRequest,
    QueryResponse,
    QueryRoute,
    RoutingDecision,
)


class TestQueryRequest:
    def test_valid(self):
        req = QueryRequest(query="Hello")
        assert req.query == "Hello"

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_with_user_id(self):
        req = QueryRequest(query="test", user_id="user-1")
        assert req.user_id == "user-1"


class TestRoutingDecision:
    def test_valid(self):
        d = RoutingDecision(
            route=QueryRoute.SMALL_TALK, confidence=0.95, reasoning="test"
        )
        assert d.route == QueryRoute.SMALL_TALK

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            RoutingDecision(
                route=QueryRoute.SMALL_TALK, confidence=1.5, reasoning="test"
            )
        with pytest.raises(ValidationError):
            RoutingDecision(
                route=QueryRoute.SMALL_TALK, confidence=-0.1, reasoning="test"
            )


class TestQueryResponse:
    def test_default_ok(self):
        r = QueryResponse(route=QueryRoute.SMALL_TALK, confidence=0.9, answer="Hi!")
        assert r.ok is True
        assert r.request_id  # should auto-generate

    def test_with_sources(self):
        r = QueryResponse(
            route=QueryRoute.FACT_FROM_DOCS,
            confidence=0.85,
            answer="Policy says...",
            sources=[],
            groundedness=True,
        )
        assert r.groundedness is True


class TestFeedbackRequest:
    def test_positive(self):
        f = FeedbackRequest(request_id="123", feedback="positive")
        assert f.feedback == "positive"

    def test_negative(self):
        f = FeedbackRequest(request_id="123", feedback="negative")
        assert f.feedback == "negative"

    def test_invalid(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(request_id="123", feedback="maybe")
