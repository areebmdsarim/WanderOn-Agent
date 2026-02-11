"""
Full RAG pipeline: retrieve → generate → verify groundedness.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.llm.local_llm import get_generator_llm, invoke_llm
from src.llm.prompts import RAG_ANSWER_PROMPT
from src.rag.groundedness import check_groundedness
from src.rag.retriever import FAISSRetriever
from src.schemas import SourceChunk

# Module-level retriever (initialised once)
_retriever: Optional[FAISSRetriever] = None


def get_retriever() -> FAISSRetriever:
    """Return (or create) the shared FAISSRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
        if not _retriever.load():
            logger.info("No saved index found — building from data/policies/")
            _retriever.build_index()
    return _retriever


def run_rag_pipeline(
    query: str, config: Optional[LLMConfig] = None
) -> Tuple[str, List[SourceChunk], bool, str]:
    """
    Execute the full RAG pipeline for a FACT_FROM_DOCS query.

    Returns:
        (answer, sources, is_grounded, groundedness_explanation)
    """
    retriever = get_retriever()
    chunks = retriever.retrieve(query)

    if not chunks:
        return (
            "I don't have enough information in the travel policies to answer that.",
            [],
            False,
        )

    # Build context string
    context_parts = []
    sources: List[SourceChunk] = []
    for i, c in enumerate(chunks):
        context_parts.append(f"[Source {i+1}]: {c['text']}")
        sources.append(
            SourceChunk(
                doc_id=c["doc_id"],
                chunk_id=c["chunk_id"],
                text_snippet=c["text"][:200],
            )
        )
    context = "\n\n".join(context_parts)

    # Generate answer
    prompt = RAG_ANSWER_PROMPT.format(context=context, question=query)
    llm = get_generator_llm(config)
    answer = invoke_llm(llm, prompt)
    logger.info(f"RAG answer generated ({len(answer)} chars)")

    # Groundedness check
    is_grounded, explanation = check_groundedness(context, query, answer, config)
    if not is_grounded:
        logger.warning(
            f"Answer failed groundedness check: {explanation} — returning fallback"
        )
        answer = (
            "I don't have enough information in the travel policies to answer "
            "that. Please consult the full policy documents or contact HR."
        )

    return answer, sources, is_grounded, explanation
