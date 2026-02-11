"""
Load, chunk, and index travel-policy documents from data/policies/.
Supports .txt and .md files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from loguru import logger


def _read_file(path: Path) -> str:
    """Read a text file with utf-8 encoding."""
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 75,
) -> List[str]:
    """Split text into word-level chunks with overlap."""
    words = text.split()
    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def load_documents(
    directory: str = "data/policies",
) -> List[Dict]:
    """
    Scan *directory* for .txt / .md files, chunk them, and return a list of
    ``{"doc_id": ..., "chunk_id": ..., "text": ...}`` dicts.
    """
    docs_dir = Path(directory)
    if not docs_dir.exists():
        logger.warning(f"Policy directory not found: {docs_dir}")
        return []

    records: List[Dict] = []
    for fpath in sorted(docs_dir.iterdir()):
        if fpath.suffix.lower() not in {".txt", ".md"}:
            continue
        text = _read_file(fpath)
        chunks = chunk_text(text)
        doc_id = fpath.stem
        for idx, chunk in enumerate(chunks):
            records.append({"doc_id": doc_id, "chunk_id": idx, "text": chunk})
        logger.info(f"Loaded {len(chunks)} chunks from {fpath.name}")

    logger.info(f"Total document chunks: {len(records)}")
    return records
