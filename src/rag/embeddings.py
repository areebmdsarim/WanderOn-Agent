"""
Sentence-transformer embeddings wrapper.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@lru_cache(maxsize=1)
def _load_model():
    """Lazily load the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of strings → (N, dim) float32 numpy array."""
    model = _load_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string → (1, dim) float32 array."""
    return embed_texts([query])
