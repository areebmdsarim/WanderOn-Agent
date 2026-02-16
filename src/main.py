"""
FastAPI application — the single entry-point for the AI Travel Policy Router.

Endpoints:
  POST /query       → classify + execute + return answer
  POST /feedback    → submit thumbs-up / thumbs-down    `
  POST /index/build → rebuild the FAISS index from data/policies/
  GET  /health      → liveness check
"""

from __future__ import annotations

import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.guardrails.input_guards import (
    ConfidenceGuardrail,
    InputGuardrails,
    TokenBudgetGuardrail,
)
from src.llm.thread_manager import get_thread_manager
from src.observability.logging import log_query, save_feedback
from src.rag.pipeline import get_retriever, run_rag_pipeline

# Dynamically select LLM backend based on environment variable
_LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
if _LLM_BACKEND == "openai":
    from src.llm.openai_llm import (
        get_generator_llm,
        get_total_tokens,
        invoke_llm,
        reset_token_counter,
    )
else:
    from src.llm.local_llm import (
        get_generator_llm,
        get_total_tokens,
        invoke_llm,
        reset_token_counter,
    )

from src.router import classify_query
from src.schemas import (
    ErrorResponse,
    FeedbackRecord,
    FeedbackRequest,
    QueryLog,
    QueryRequest,
    QueryResponse,
    QueryRoute,
    ThinkingStep,
)
from src.tools.executor import execute_tool

load_dotenv()

app = FastAPI(
    title="AI Travel Policy Router",
    description="Intelligent query router with RAG & structured tools for travel policies",
    version="1.0.0",
)


# ── Global Exception Handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="Something exploded on our end. Check the logs.",
            request_id=str(uuid.uuid4()),
        ).model_dump(),
    )


# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    # NOTE: In production, restrict to specific frontend URL instead of "*"
    allow_origins=["*"],  # Allow all for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initializing Guardrail instances ──────────────────────────────────────────────────────
_input_guard = InputGuardrails()
# Confidence threshold defaulting to 0.85
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
_confidence_guard = ConfidenceGuardrail(threshold=CONFIDENCE_THRESHOLD)
MAX_QUERY_WORDS = int(os.getenv("MAX_QUERY_WORDS", "2000"))
# Token budget defaulting to 2000 words (~3000-4000 tokens depending on language)
_token_guard = TokenBudgetGuardrail(max_words=MAX_QUERY_WORDS)

# ── Small-talk canned responses ──────────────────────────────────────────────
_SMALL_TALK_RESPONSES = [
    "Hello! 👋 I'm your travel-policy assistant. Ask me about travel policies, visa requirements, per-diem rates, or flight booking rules.",
    "Hi there! How can I help you with travel policies today?",
    "Hey! Feel free to ask me anything about your company's travel policies.",
]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    """Main query endpoint — routes, executes, and logs."""
    # Unique ID for this request because we want to log it and potentially link it to feedback later.
    # It is also needed for the FE to correlate the response to the original query in case of async processing.
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    reset_token_counter()
    trace: list[ThinkingStep] = []
    guardrails_triggered: list[str] = []

    # Initialize or retrieve conversation thread
    tm = get_thread_manager()
    thread_id = req.thread_id or tm.create_thread(user_id=req.user_id or "default")
    tm.add_message(thread_id, "user", req.query)

    # 1. Input guardrails
    ok, reason = _input_guard.check(req.query)
    trace.append(
        ThinkingStep(
            event="INPUT_GUARDRAIL",
            status="success" if ok else "failure",
            message=f"Input validation: {reason}",
            data={"reason": reason} if not ok else None,
        )
    )
    if not ok:
        guardrails_triggered.append(reason)
        _log(request_id, req, "OUT_OF_SCOPE", 0.0, reason, 0.0, guardrails_triggered)
        if "blocked_pii" in reason:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="blocked_pii",
                    message="Request contains sensitive personal data. Remove PII and retry.",
                    request_id=request_id,
                    trace=trace,
                ).model_dump(),
            )
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=reason,
                message=f"Input validation failed: {reason}",
                request_id=request_id,
                trace=trace,
            ).model_dump(),
        )

    ok, reason = _token_guard.check(req.query)
    if not ok:
        guardrails_triggered.append(reason)
        # Log the refusal before raising
        _log(request_id, req, "OUT_OF_SCOPE", 0.0, reason, 0.0, guardrails_triggered)
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="token_budget_exceeded",
                message=f"Request exceeds token budget. Please shorten your query. ({reason})",
                request_id=request_id,
                trace=trace,
            ).model_dump(),
        )

    # 2. Route
    decision = classify_query(req.query, req.config, CONFIDENCE_THRESHOLD)
    trace.append(
        ThinkingStep(
            event="ROUTING",
            status="success",
            message=f"Routed to {decision.route} (confidence: {decision.confidence:.2f})",
            data={
                "route": decision.route,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
            },
        )
    )

    # 3. Confidence guardrail
    ok, reason = _confidence_guard.check(decision.confidence)
    if not ok:
        guardrails_triggered.append(reason)
        # If confidence is extremely low (< 0.5), force OUT_OF_SCOPE
        if decision.confidence < 0.5:
            trace.append(
                ThinkingStep(
                    event="CONFIDENCE_GUARDRAIL",
                    status="failure",
                    message=f"Confidence too low ({decision.confidence:.2f}). Fallback to OUT_OF_SCOPE.",
                )
            )
            logger.warning(
                f"Confidence too low: {decision.confidence} < 0.5. Forcing OUT_OF_SCOPE."
            )
            decision.route = QueryRoute.OUT_OF_SCOPE
            decision.reasoning = f"Confidence {decision.confidence:.2f} < 0.5. Original route: {decision.route}"
        else:
            trace.append(
                ThinkingStep(
                    event="CONFIDENCE_GUARDRAIL",
                    status="warning",
                    message=f"Low confidence routing detected: {reason}",
                )
            )
            logger.warning(f"Low confidence routing: {decision.confidence}")

    # 4. Execute per route
    answer = ""
    sources = []
    groundedness = None
    tool_used = None
    tool_data = None

    if decision.route == QueryRoute.SMALL_TALK:
        import random

        answer = random.choice(_SMALL_TALK_RESPONSES)
        trace.append(
            ThinkingStep(
                event="EXECUTION",
                status="success",
                message="Handled as small talk",
            )
        )

    elif decision.route == QueryRoute.FACT_FROM_DOCS:
        # Build context from conversation history
        prev_messages = tm.get_thread_messages(thread_id)
        context = ""
        if len(prev_messages) > 1:
            context = "\n\nPrevious conversation:\n"
            for msg in prev_messages[:-1]:  # Exclude current message
                context += f"{msg['role'].upper()}: {msg['content']}\n"
            context += "\n"
        
        answer, sources, groundedness, explanation = run_rag_pipeline(
            context + req.query if context else req.query, req.config
        )
        trace.append(
            ThinkingStep(
                event="RAG_PIPELINE",
                status="success" if groundedness else "warning",
                message=f"RAG execution completed. Grounded: {groundedness}",
                data={
                    "chunks_retrieved": len(sources),
                    "grounded": groundedness,
                    "explanation": explanation,
                },
            )
        )

    elif decision.route == QueryRoute.STRUCTURED_DATA:
        # Build context from conversation history
        prev_messages = tm.get_thread_messages(thread_id)
        context = ""
        if len(prev_messages) > 1:
            context = "Previous context:\n"
            for msg in prev_messages[:-1]:
                context += f"{msg['role'].upper()}: {msg['content']}\n"
            context += "\n"
        
        tool_result, error = execute_tool(req.query, req.config)
        if error:
            answer = f"I couldn't process that structured request: {error}"
            trace.append(
                ThinkingStep(
                    event="TOOL_EXECUTION",
                    status="failure",
                    message=f"Tool execution failed: {error}",
                )
            )
        else:
            tool_used = tool_result.tool
            tool_data = tool_result.data

            # Synthesize natural language response
            llm = get_generator_llm(req.config)
            prompt = (
                f'{context}You are a helpful travel assistant. The user asked: "{req.query}"\n'
                f"The tool '{tool_used}' returned the following data:\n{tool_data}\n\n"
                "Please provide a natural language answer summarizing this information for the user."
            )
            answer = invoke_llm(llm, prompt)

            trace.append(
                ThinkingStep(
                    event="TOOL_EXECUTION",
                    status="success",
                    message=f"Tool {tool_used} executed successfully",
                    data=tool_data,
                )
            )

    elif decision.route == QueryRoute.OUT_OF_SCOPE:
        answer = (
            "I'm sorry, but that request is outside the scope of what I can help with. "
            "I can assist with travel policy questions, visa requirements, per-diem rates, "
            "and flight booking rules."
        )
        trace.append(
            ThinkingStep(
                event="EXECUTION",
                status="success",
                message="Query out of scope",
            )
        )

    # Add assistant response to thread
    tm.add_message(thread_id, "assistant", answer)

    elapsed_ms = (time.perf_counter() - start) * 1000
    total_tokens = get_total_tokens()

    # 5. Build response
    response = QueryResponse(
        request_id=request_id,
        thread_id=thread_id,
        route=decision.route,
        confidence=decision.confidence,
        answer=answer,
        reasoning=decision.reasoning,
        sources=sources,
        groundedness=groundedness,
        tool_used=tool_used,
        tool_data=tool_data,
        trace=trace,
        latency_ms=round(elapsed_ms, 1),
        total_tokens=total_tokens,
    )

    # 6. Log
    _log(
        request_id,
        req,
        decision.route.value,
        decision.confidence,
        decision.reasoning,
        elapsed_ms,
        guardrails_triggered,
        tools=[tool_used] if tool_used else [],
        chunks=len(sources),
        groundedness=groundedness,
        answer_length=len(answer),
        total_tokens=total_tokens,
    )

    return response


@app.post("/feedback")
async def handle_feedback(fb: FeedbackRequest):
    """Accept user feedback (thumbs up / down)."""
    record = FeedbackRecord(
        request_id=fb.request_id,
        feedback=fb.feedback,
        comment=fb.comment,
    )
    save_feedback(record)
    return {"ok": True, "message": "Feedback recorded. Thank you!"}


@app.post("/index/build")
async def rebuild_index():
    """Trigger a re-build of the FAISS index from data/policies/."""
    retriever = get_retriever()
    count = retriever.build_index()
    return {"ok": True, "chunks_indexed": count}


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "healthy", "version": "1.0.0"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _log(
    request_id: str,
    req: QueryRequest,
    route: str,
    confidence: float,
    reasoning: str,
    latency_ms: float,
    guardrails_triggered: list[str],
    total_tokens: int = 0,
    *,
    tools: list[str] | None = None,
    chunks: int = 0,
    groundedness: bool | None = None,
    answer_length: int = 0,
):
    entry = QueryLog(
        request_id=request_id,
        user_id=req.user_id,
        query_text=req.query,
        query_length=len(req.query),
        route_taken=route,
        route_confidence=confidence,
        route_reasoning=reasoning,
        tools_called=tools or [],
        chunks_retrieved=chunks,
        groundedness=groundedness,
        answer_length=answer_length,
        latency_ms=round(latency_ms, 1),
        total_tokens=total_tokens,
        guardrails_triggered=guardrails_triggered,
    )
    log_query(entry)
