"""
Tests for the RAG pipeline components: chunking and document loading.
"""

import pytest

from src.rag.load_docs import chunk_text, load_documents


class TestChunking:

    def test_basic_chunking(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) > 1
        # Each chunk should have at most 30 words
        for c in chunks:
            assert len(c.split()) <= 30

    def test_overlap_exists(self):
        text = " ".join([f"w{i}" for i in range(50)])
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        # With overlap, later chunks should share some words with earlier ones
        if len(chunks) >= 2:
            words_1 = set(chunks[0].split())
            words_2 = set(chunks[1].split())
            assert len(words_1 & words_2) > 0

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=300)
        assert len(chunks) == 1


class TestDocumentLoading:

    def test_load_from_policies(self):
        """Integration test — requires data/policies/ to exist with files."""
        docs = load_documents("data/policies")
        # Our sample docs should produce chunks
        assert len(docs) > 0
        # Each doc should have required keys
        for d in docs:
            assert "doc_id" in d
            assert "chunk_id" in d
            assert "text" in d

    def test_load_missing_dir(self):
        docs = load_documents("/tmp/nonexistent_dir_12345")
        assert docs == []
