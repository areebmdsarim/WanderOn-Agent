"""
Groundedness verification — checks whether a generated RAG answer
is fully supported by the retrieved context.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from loguru import logger

# Dynamically select LLM backend based on environment variable
_LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
if _LLM_BACKEND == "openai":
    from src.llm.openai_llm import get_classifier_llm, invoke_llm
else:
    from src.llm.local_llm import get_classifier_llm, invoke_llm

from src.llm.prompts import GROUNDEDNESS_PROMPT
from src.schemas import LLMConfig


def check_groundedness(
    context: str, question: str, answer: str, config: Optional[LLMConfig] = None
) -> Tuple[bool, str]:
    """
    Ask the LLM whether *answer* is supported by *context*.

    Returns (is_grounded, explanation).
    """
    prompt = GROUNDEDNESS_PROMPT.format(
        context=context,
        question=question,
        answer=answer,
    )

    llm = get_classifier_llm(config)
    raw = invoke_llm(llm, prompt)
    logger.debug(f"Groundedness check raw:\n{raw}")

    # Parse response
    explanation = "No explanation provided"
    is_grounded = False
    for line in raw.splitlines():
        line = line.strip().upper()
        if line.startswith("GROUNDED:"):
            value = line.split(":", 1)[1].strip()
            is_grounded = value == "YES"
        elif line.startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    return is_grounded, explanation
