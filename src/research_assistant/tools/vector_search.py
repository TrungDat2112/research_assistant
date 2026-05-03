from __future__ import annotations

import logging
from typing import Any

from research_assistant.config import get_settings
from research_assistant.graph.state import SearchHit
from research_assistant.observability import observe, update_span
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.embedding import EmbeddingModel
from research_assistant.rag.hybrid import (
    hybrid_result_to_search_hit,
    hybrid_search_stage1,
)
from research_assistant.rag.hyde import dense_embedding_for_retrieval
from research_assistant.rag.vector_store import ChromaStore

logger = logging.getLogger(__name__)

_BM25_CACHE: tuple[str, int, BM25CorpusIndex] | None = None


class VectorSearchError(RuntimeError):
    """Raised when hybrid search cannot run (misconfiguration, empty query, ...)."""


def clear_vector_search_cache() -> None:
    global _BM25_CACHE
    _BM25_CACHE = None


def _default_store() -> ChromaStore:
    s = get_settings()
    return ChromaStore(s.chroma_persist_dir, s.corpus_collection)


def _default_embedder() -> EmbeddingModel:
    s = get_settings()
    return EmbeddingModel(s.embedding_model, s.embedding_device)


def _cached_bm25_index(store: ChromaStore) -> BM25CorpusIndex:
    global _BM25_CACHE
    name = store.collection_name
    n = store.count()
    if _BM25_CACHE and _BM25_CACHE[0] == name and _BM25_CACHE[1] == n:
        return _BM25_CACHE[2]
    logger.info("Building BM25 index for collection=%s (n=%d)", name, n)
    idx = BM25CorpusIndex.from_chroma(store)
    _BM25_CACHE = (name, n, idx)
    return idx


@observe(name="vector_search", as_type="tool", capture_input=False, capture_output=False)
def vector_search(
    query: str,
    *,
    top_k: int = 20,
    filters: dict[str, Any] | None = None,
    weight_dense: float = 0.5,
    weight_bm25: float = 0.5,
    dense_top_k: int = 50,
    bm25_top_k: int = 50,
    store: ChromaStore | None = None,
    embedder: EmbeddingModel | None = None,
    bm25_index: BM25CorpusIndex | None = None,
) -> list[SearchHit]:

    if not query or not query.strip():
        raise VectorSearchError("Empty query — refusing vector search.")

    bounded = max(1, min(top_k, 100))
    st = store if store is not None else _default_store()
    emb = embedder if embedder is not None else _default_embedder()
    idx = bm25_index if bm25_index is not None else _cached_bm25_index(st)

    if st.count() == 0 or idx.size() == 0:
        logger.info("vector_search: empty corpus — returning 0 hits")
        update_span(
            input={"query": query, "top_k": bounded, "filters": filters},
            output={"n_hits": 0, "reason": "empty_corpus"},
        )
        return []

    qvec, hyde_meta = dense_embedding_for_retrieval(
        query,
        st,
        idx,
        emb,
        where=filters,
        hyde_enabled=None,
    )
    hybrid_hits = hybrid_search_stage1(
        st,
        idx,
        query,
        qvec,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        weight_dense=weight_dense,
        weight_bm25=weight_bm25,
        final_top_k=bounded,
        where=filters,
    )
    hits = [hybrid_result_to_search_hit(h) for h in hybrid_hits]
    update_span(
        input={
            "query": query,
            "top_k": bounded,
            "filters": filters,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
        },
        output={
            "n_hits": len(hits),
            "weight_dense": weight_dense,
            "weight_bm25": weight_bm25,
            "hyde_applied": hyde_meta.get("hyde_applied", False),
            "hyde_reason": hyde_meta.get("hyde_reason"),
            "urls": [str(h.url) for h in hits[:10]],
        },
    )
    return hits
