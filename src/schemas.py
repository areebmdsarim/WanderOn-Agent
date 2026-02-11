"""
Pydantic v2 schemas for all request/response contracts in the AI Travel Router.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Routing ──────────────────────────────────────────────────────────────────


class QueryRoute(str, Enum):
    """The four canonical route categories."""

    SMALL_TALK = "SMALL_TALK"
    FACT_FROM_DOCS = "FACT_FROM_DOCS"
    STRUCTURED_DATA = "STRUCTURED_DATA"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class RoutingDecision(BaseModel):
    """Output of the query router (rules or LLM)."""

    route: QueryRoute
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ── API request / response ───────────────────────────────────────────────────


class LLMConfig(BaseModel):
    """Dynamic configuration for LLM parameter tuning."""

    model: str | None = None
    temperature: float = 0.7  # simple default, no Field()
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, gt=0)
    num_predict: Optional[int] = Field(None, gt=0)
    format: Optional[str] = None
    max_tokens: Optional[int] = Field(None, gt=0)  # Deprecated but kept for safety


class QueryRequest(BaseModel):
    """Incoming API request."""

    query: str = Field(..., min_length=1, max_length=10000)
    user_id: Optional[str] = None
    config: Optional[LLMConfig] = None


class SourceChunk(BaseModel):
    """A single cited source chunk from RAG."""

    doc_id: str
    chunk_id: int
    text_snippet: str


class ThinkingStep(BaseModel):
    """A discrete event or 'thought' in the processing pipeline."""

    event: str  # e.g., "GUARDRAIL_CHECK", "ROUTING", "RAG_RETRIEVAL", "TOOL_EXECUTION"
    status: str  # "success", "failure", "warning"
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class QueryResponse(BaseModel):
    """Unified API response sent for every query."""

    ok: bool = True
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    route: QueryRoute
    confidence: float = Field(ge=0.0, le=1.0)
    answer: str
    reasoning: str = ""
    sources: List[SourceChunk] = Field(default_factory=list)
    groundedness: Optional[bool] = None
    tool_used: Optional[str] = None
    tool_data: Optional[Dict[str, Any]] = None
    trace: List[ThinkingStep] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    total_tokens: Optional[int] = 0


class ErrorResponse(BaseModel):
    """Standardized error envelope."""

    ok: bool = False
    error: str
    message: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ── Tool schemas ─────────────────────────────────────────────────────────────


class VisaCheckRequest(BaseModel):
    passport_country: str = Field(..., min_length=2, max_length=56)
    destination_country: str = Field(..., min_length=2, max_length=56)


class PerDiemRequest(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    country: str = Field(..., min_length=2, max_length=56)


class FlightPolicyRequest(BaseModel):
    origin: str = Field(..., min_length=3, max_length=100)
    destination: str = Field(..., min_length=3, max_length=100)
    cabin_class: str = Field(..., pattern=r"^(economy|premium_economy|business|first)$")


class ApprovalRequest(BaseModel):
    trip_cost: float = Field(..., gt=0)
    destination_type: str = Field(..., pattern=r"^(domestic|international|high_risk)$")


class ToolResponse(BaseModel):
    """Wrapper returned by every tool function."""

    ok: bool = True
    tool: str
    data: Dict[str, Any]
    source: str = "tool-db"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Feedback ─────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    request_id: str
    feedback: str = Field(..., pattern=r"^(positive|negative)$")
    comment: Optional[str] = None


class FeedbackRecord(BaseModel):
    request_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    feedback: str
    comment: Optional[str] = None


# ── Observability ────────────────────────────────────────────────────────────


class QueryLog(BaseModel):
    """Structured log entry for every processed query."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    request_id: str
    user_id: Optional[str] = None

    # Input
    query_text: str
    query_length: int = 0

    # Routing
    route_taken: str = ""
    route_confidence: float = 0.0
    route_reasoning: str = ""

    # Execution
    tools_called: List[str] = Field(default_factory=list)
    chunks_retrieved: int = 0
    groundedness: Optional[bool] = None

    # Output
    answer_length: int = 0

    # Performance
    latency_ms: float = 0.0
    total_tokens: int = 0

    # Guardrails
    guardrails_triggered: List[str] = Field(default_factory=list)
