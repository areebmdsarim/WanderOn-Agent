"""
Handles OpenAI API integration for the AI Travel Router.
Mirrors the `local_llm.py` interface for seamless switching.
Uses LangChain's ChatOpenAI with thread management for conversation state.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger

from src.schemas import LLMConfig

import json
from pathlib import Path

load_dotenv()

# ContextVar to track tokens across a single request lifecycle
token_counter: ContextVar[int] = ContextVar("token_counter", default=0)

CONFIG_PATH = Path("configs/model_config.json")


@lru_cache(maxsize=1)
def load_model_config() -> dict:
    """Load model settings from the json config file."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file {CONFIG_PATH} not found. Using defaults.")
        return {}
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
        return {}


# Fallback defaults if config file is missing
DEFAULT_CONFIGS = {
    "classifier": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 256,
    },
    "generator": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1024,
    },
}


def resolve_param(role: str, key: str, override_val: Any) -> Any:
    """
    Decide which param to use.
    Order: json config > manual override > hardcoded defaults.
    """
    # 1. JSON (Admin governance) - read from "openai" section
    config = load_model_config()
    # Get the OpenAI backend section
    backend_config = config.get("openai", {})
    role_config = backend_config.get(role, {})
    val = role_config.get(key)
    if val is not None:
        return val

    # 2. Override (User setup)
    if override_val is not None:
        return override_val

    # 3. Default (System level)
    return DEFAULT_CONFIGS.get(role, {}).get(key)


def reset_token_counter():
    """Reset the token counter for the current context."""
    token_counter.set(0)


def get_total_tokens() -> int:
    """Get the sum of tokens tracked in the current context."""
    return token_counter.get()


@lru_cache(maxsize=16)
def _get_llm_instance(
    model: str,
    temperature: float,
    max_tokens: int,
) -> ChatOpenAI:
    """
    Creates a ChatOpenAI instance with caching.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before using OpenAI LLM."
        )

    logger.info(
        f"Initializing ChatOpenAI: model={model}, temperature={temperature}, max_tokens={max_tokens}"
    )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def _get_resolved_llm(role: str, config: Optional[LLMConfig] = None) -> ChatOpenAI:
    """Resolve all parameters for a specific role and return the LLM."""
    model = resolve_param(role, "model", config.model if config else None)
    temp = resolve_param(role, "temperature", config.temperature if config else None)

    # Handle max_tokens (OpenAI uses max_tokens instead of num_predict)
    max_tokens_override = None
    if config:
        max_tokens_override = config.max_tokens or config.num_predict

    max_tokens = resolve_param(role, "max_tokens", max_tokens_override)

    return _get_llm_instance(
        model=str(model),
        temperature=float(temp),
        max_tokens=int(max_tokens),
    )


def get_classifier_llm(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    """Get classifier LLM with low temperature for routing."""
    return _get_resolved_llm("classifier", config)


def get_generator_llm(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    """Get generator LLM with slightly higher temperature for RAG generation."""
    return _get_resolved_llm("generator", config)


def invoke_llm(llm: ChatOpenAI, prompt: str) -> str:
    """
    Invoke the LLM with a prompt and track token usage.
    
    Args:
        llm: ChatOpenAI instance
        prompt: The prompt string to send to the model
        
    Returns:
        The model's response as a string
    """
    try:
        # Using invoke() to get the response message
        response = llm.invoke(prompt)

        # Extract token usage if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total = input_tokens + output_tokens

            if total > 0:
                current = token_counter.get()
                token_counter.set(current + total)
                logger.debug(
                    f"OpenAI LLM usage: input={input_tokens}, output={output_tokens}, total={total}"
                )

        # Extract the text content from the response
        if hasattr(response, "content"):
            return response.content.strip()
        else:
            return str(response).strip()

    except Exception as e:
        logger.error(f"OpenAI LLM invocation failed: {e}")
        raise


def invoke_llm_with_messages(llm: ChatOpenAI, messages: list) -> str:
    """
    Invoke the LLM with structured messages (system + user).
    Useful for maintaining conversation context with threads.
    
    Args:
        llm: ChatOpenAI instance
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        The model's response as a string
    """
    try:
        response = llm.invoke(messages)

        # Track token usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total = input_tokens + output_tokens

            if total > 0:
                current = token_counter.get()
                token_counter.set(current + total)
                logger.debug(
                    f"OpenAI LLM usage: input={input_tokens}, output={output_tokens}, total={total}"
                )

        if hasattr(response, "content"):
            return response.content.strip()
        else:
            return str(response).strip()

    except Exception as e:
        logger.error(f"OpenAI LLM message invocation failed: {e}")
        raise

