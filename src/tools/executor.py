"""
Tool executor — validates input, selects the right tool, and executes it.
Uses the LLM to extract parameters from natural-language queries.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

from loguru import logger
from pydantic import ValidationError

# Dynamically select LLM backend based on environment variable
_LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
if _LLM_BACKEND == "openai":
    from src.llm.openai_llm import get_classifier_llm, invoke_llm
else:
    from src.llm.local_llm import get_classifier_llm, invoke_llm

from src.llm.prompts import TOOL_EXTRACTION_PROMPT
from src.schemas import ToolResponse, LLMConfig
from src.tools.travel_tools import TOOL_REGISTRY


def _parse_tool_extraction(raw: str) -> Tuple[Optional[str], Dict[str, str]]:
    """Parse TOOL: ... / PARAMS: ... from LLM output."""
    tool_name: Optional[str] = None
    params: Dict[str, str] = {}

    for line in raw.splitlines():
        line = line.strip()
        up = line.upper()
        if up.startswith("TOOL:"):
            tool_name = line.split(":", 1)[1].strip().lower()
        elif up.startswith("PARAMS:"):
            raw_params = line.split(":", 1)[1].strip()
            for pair in raw_params.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    params[k.strip()] = v.strip()

    return tool_name, params


def execute_tool(
    query: str, config: Optional[LLMConfig] = None
) -> Tuple[Optional[ToolResponse], Optional[str]]:
    """
    Extract tool + params from *query* via LLM, validate, and execute.

    Returns (ToolResponse, None) on success, or (None, error_message) on failure.
    """
    # 1. Extract tool call from query
    llm = get_classifier_llm(config)
    prompt = TOOL_EXTRACTION_PROMPT.format(query=query)
    raw = invoke_llm(llm, prompt)
    logger.debug(f"Tool extraction raw:\n{raw}")

    tool_name, params = _parse_tool_extraction(raw)

    if not tool_name or tool_name not in TOOL_REGISTRY:
        return (
            None,
            f"Could not identify a valid tool for this query (got: {tool_name})",
        )

    entry = TOOL_REGISTRY[tool_name]
    schema_cls = entry["schema"]
    fn = entry["fn"]

    # 2. Validate with Pydantic
    try:
        validated = schema_cls(**params)
    except ValidationError as e:
        logger.warning(f"Tool validation error: {e}")
        return None, f"Invalid parameters for {tool_name}: {e}"

    # 3. Execute
    try:
        result = fn(validated)
        logger.info(f"Tool {tool_name} executed successfully")
        return result, None
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return None, f"Tool execution failed: {str(e)}"
