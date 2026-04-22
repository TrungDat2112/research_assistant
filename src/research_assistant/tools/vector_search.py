"""Vector + hybrid corpus search — wraps :class:`~research_assistant.rag.vector_store.ChromaStore`.

Contract (LLM-facing, same as :func:`~research_assistant.tools.web_search.web_search`):
    name:     vector_search
    category: information_retrieval (PLAN.md §4.1)
    purpose:  Retrieve chunks from the local ingested corpus using stage-1 hybrid
              retrieval (dense top-50 + BM25 top-50, 0.5/0.5 by default).
    when:     For prior art, paper details, or blog explanations likely present
              in the seed corpus (AI/ML RAG, agents, reasoning, etc.).

Returns :class:`~research_assistant.graph.state.SearchHit` rows with
``source="corpus"`` so the Synthesizer can treat them like web hits.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import HttpUrl, TypeAdapter

from research_assistant.config import get_settings
from research_assistant.graph.state import SearchHit
from research_assistant.observability import observe, update_span
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.embedding import EmbeddingModel
from research_assistant.rag.hybrid import HybridSearchResult, hybrid_search_stage1
from research_assistant.rag.vector_store import ChromaStore

logger = logging.getLogger(__name__)

_BM25_CACHE: tuple[str, int, BM25CorpusIndex] | None = None


class VectorSearchError(RuntimeError):
    """Raised when hybrid search cannot run (misconfiguration, empty query, ...)."""


def clear_vector_search_cache() -> None:
    """Drop the in-process BM25 cache (call from tests after rebuild/reset)."""
    global _BM25_CACHE
    _BM25_CACHE = None


def _default_store() -> ChromaStore:
    s = get_settings()
    return ChromaStore(s.chroma_persist_dir, s.corpus_collection)


def _default_embedder() -> EmbeddingModel:
    s = get_settings()
    return EmbeddingModel(s.embedding_model, s.embedding_device)


def _cached_bm25_index(store: ChromaStore) -> BM25CorpusIndex:
    """Reuse the BM25 index while the Chroma document count is stable."""
    global _BM25_CACHE
    name = store.collection_name
    n = store.count()
    if _BM25_CACHE and _BM25_CACHE[0] == name and _BM25_CACHE[1] == n:
        return _BM25_CACHE[2]
    logger.info("Building BM25 index for collection=%s (n=%d)", name, n)
    idx = BM25CorpusIndex.from_chroma(store)
    _BM25_CACHE = (name, n, idx)
    return idx


def _hybrid_result_to_search_hit(r: HybridSearchResult) -> SearchHit:
    meta = r.metadata
    url_s = str(meta.get("source_url") or "").strip()
    if not url_s:
        url_s = "https://example.invalid/missing-source-url"
    title = str(meta.get("title") or meta.get("source_id") or "Untitled")
    snippet = r.body if len(r.body) <= 1600 else f"{r.body[:1600]}…"
    published = str(meta.get("published_date") or "").strip()
    return SearchHit(
        url=TypeAdapter(HttpUrl).validate_python(url_s),
        title=title,
        snippet=snippet,
        score=r.combined_score,
        published_date=published or None,
        source="corpus",
        raw_content=r.body,
    )


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
    """Hybrid BM25 + dense search over the Chroma corpus.

    Parameters
    ----------
    query:
        Natural-language query (same embedding model as ingest, ADR-013).
    top_k:
        Number of fused hits to return after weighting (default 20).
    filters:
        Optional Chroma ``where`` metadata filter passed to the dense leg.
    weight_dense / weight_bm25:
        Non-negative weights; they need not sum to 1 — the hybrid module
        normalises by the sum of the two weights internally.
    dense_top_k / bm25_top_k:
        Candidate pool size per leg (PLAN §5.2 stage 1 — default 50/50).
    store / embedder / bm25_index:
        Injection points for unit tests. Production callers omit them so a
        real :class:`~research_assistant.rag.vector_store.ChromaStore` and
        :class:`~research_assistant.rag.embedding.EmbeddingModel` are built
        from :class:`~research_assistant.config.Settings`.
    """
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

    qvec = emb.embed_query(query)
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
    hits = [_hybrid_result_to_search_hit(h) for h in hybrid_hits]
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
            "urls": [str(h.url) for h in hits[:10]],
        },
    )
    return hits
