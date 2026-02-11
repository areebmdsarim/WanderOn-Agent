"""
Structured JSON logging and feedback persistence via Loguru.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.schemas import FeedbackRecord, QueryLog


# ── Configure Loguru ─────────────────────────────────────────────────────────

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_FEEDBACK_PATH = Path("data/feedback.jsonl")

# Remove default handler and add custom ones
logger.remove()

# Console handler (human-readable)
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)

# JSONL file handler (machine-readable)
logger.add(
    str(_LOG_DIR / "app_{time:YYYY-MM-DD}.log"),
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    serialize=True,  # JSON Lines format
)


# ── Query log persistence ───────────────────────────────────────────────────

_QUERY_LOG_PATH = _LOG_DIR / "queries.jsonl"


def log_query(entry: QueryLog) -> None:
    """Append a structured query log entry to the JSONL log file."""
    with open(_QUERY_LOG_PATH, "a") as f:
        f.write(entry.model_dump_json() + "\n")
    logger.info(
        f"Logged query  request_id={entry.request_id}  route={entry.route_taken}  "
        f"confidence={entry.route_confidence:.2f}  latency={entry.latency_ms:.0f}ms"
    )


# ── Feedback persistence ────────────────────────────────────────────────────


def save_feedback(record: FeedbackRecord) -> None:
    """Append feedback to data/feedback.jsonl (JSON Lines format)."""
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_FEEDBACK_PATH, "a") as f:
        f.write(record.model_dump_json() + "\n")
    logger.info(
        f"Feedback saved  request_id={record.request_id}  feedback={record.feedback}"
    )
