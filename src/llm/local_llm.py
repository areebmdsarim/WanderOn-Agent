"""
Local LLM wrapper around Ollama via langchain-ollama.
All LLM inference goes through this module — no 3rd-party LLM calls.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
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
    """Load model configuration from JSON file."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file {CONFIG_PATH} not found. Using defaults.")
        return {}
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
        return {}


# Hardcoded System Defaults (Priority 3)
DEFAULT_CONFIGS = {
    "classifier": {
        "model": "llama3.2:3b",
        "temperature": 0.1,
        "num_predict": 256,
        "top_p": 0.9,
        "top_k": 40,
        "format": "json",
    },
    "generator": {
        "model": "llama3.1:8b",
        "temperature": 0.3,
        "num_predict": 1024,
        "top_p": 0.9,
        "top_k": 40,
        "format": "",
    },
}


def resolve_param(role: str, key: str, override_val: Any) -> Any:
    """
    Priority-based resolution for LLM parameters:
    1. configs/model_config.json (Governance/Production control)
    2. Runtime Overrides (e.g., API request LLMConfig)
    3. Hardcoded DEFAULT_CONFIGS (Code-level safety)
    """
    # 1. JSON (Admin governance)
    config = load_model_config()
    role_config = config.get(role, {})
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
    num_predict: int,
    top_p: float,
    top_k: int,
    format: str,
) -> OllamaLLM:
    """Internal factory to create and cache the OllamaLLM instance."""
    logger.info(
        f"Initialising Ollama LLM: model={model}, temp={temperature}, predict={num_predict}, format={format}"
    )
    return OllamaLLM(
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        top_p=top_p,
        top_k=top_k,
        format="json" if format == "json" else "",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _get_resolved_llm(role: str, config: Optional[LLMConfig] = None) -> OllamaLLM:
    """Resolve all parameters for a specific role and return the LLM."""
    model = resolve_param(role, "model", config.model if config else None)
    temp = resolve_param(role, "temperature", config.temperature if config else None)

    # Handle num_predict vs max_tokens (deprecated) mapping
    n_predict_override = None
    if config:
        n_predict_override = config.num_predict or config.max_tokens

    num_predict = resolve_param(role, "num_predict", n_predict_override)
    top_p = resolve_param(role, "top_p", config.top_p if config else None)
    top_k = resolve_param(role, "top_k", config.top_k if config else None)
    fmt = resolve_param(role, "format", config.format if config else None)

    return _get_llm_instance(
        model=str(model),
        temperature=float(temp),
        num_predict=int(num_predict),
        top_p=float(top_p),
        top_k=int(top_k),
        format=str(fmt or ""),
    )


def get_classifier_llm(config: Optional[LLMConfig] = None) -> OllamaLLM:
    """Low-temperature instance for routing / classification."""
    return _get_resolved_llm("classifier", config)


def get_generator_llm(config: Optional[LLMConfig] = None) -> OllamaLLM:
    """Slightly higher temp for RAG answer generation."""
    return _get_resolved_llm("generator", config)


def invoke_llm(llm: OllamaLLM, prompt: str) -> str:
    """
    Single call-site for LLM invocation. We use llm.generate to capture
    token usage metadata and track it in context.
    """
    try:
        # Using generate() instead of invoke() to get metadata
        result = llm.generate([prompt])

        # Capture tokens if available (Ollama via LangChain returns these in generation_info)
        generation = result.generations[0][0]
        meta = generation.generation_info or {}

        p_tokens = meta.get("prompt_eval_count", 0)
        r_tokens = meta.get("eval_count", 0)
        total = p_tokens + r_tokens

        if total > 0:
            current = token_counter.get()
            token_counter.set(current + total)
            logger.debug(
                f"LLM usage: prompt={p_tokens}, response={r_tokens}, total={total}"
            )

        return generation.text.strip()
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise
