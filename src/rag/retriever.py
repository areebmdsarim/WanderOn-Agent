"""
FAISS-based vector retriever.
Build an index from document chunks and retrieve top-K nearest neighbours.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from src.rag.embeddings import embed_query, embed_texts
from src.rag.load_docs import load_documents

load_dotenv()


class FAISSRetriever:
    """Manages a FAISS IndexFlatL2 over policy-document chunks."""

    def __init__(self):
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[Dict] = []  # parallel list of chunk metadata
        self._index_path = os.getenv(
            "FAISS_INDEX_PATH", "./data/faiss_index/policies.index"
        )
        self._meta_path = self._index_path + ".meta"

    # ── Build ────────────────────────────────────────────────────────────────

    def build_index(self, docs_dir: str = "data/policies") -> int:
        """Load documents, embed, and build the FAISS index.  Returns chunk count."""
        self.chunks = load_documents(docs_dir)
        if not self.chunks:
            logger.warning("No chunks to index.")
            return 0

        texts = [c["text"] for c in self.chunks]
        embeddings = embed_texts(texts)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        # persist
        self._save()
        return len(self.chunks)

    # ── Retrieve ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict]:
        """Return the top-K nearest document chunks for *query*."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results.")
            return []

        if top_k is None:
            top_k = int(os.getenv("TOP_K_CHUNKS", "3"))

        q_vec = embed_query(query)
        distances, indices = self.index.search(q_vec, top_k)

        results: List[Dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["distance"] = float(dist)
            results.append(chunk)

        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Index saved to {self._index_path}")

    def load(self) -> bool:
        """Load a previously saved index.  Returns True on success."""
        if not Path(self._index_path).exists():
            return False
        try:
            self.index = faiss.read_index(self._index_path)
            with open(self._meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
