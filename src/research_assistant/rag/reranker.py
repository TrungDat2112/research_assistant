"""Stage-2 cross-encoder re-ranking (PLAN.md §5.2, ADR-002).

Uses ``BAAI/bge-reranker-v2-m3`` (or another CrossEncoder) to score
``(query, passage)`` pairs and return the top-k :class:`SearchHit` rows for
the Synthesizer. Passage text prefers ``raw_content`` (corpus chunk body),
then ``snippet`` / title.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

import numpy as np

from research_assistant.config import get_settings
from research_assistant.graph.state import SearchHit
from research_assistant.rag.hybrid import HybridSearchResult, hybrid_result_to_search_hit

logger = logging.getLogger(__name__)

_MAX_PASSAGE_CHARS = 8000
_RERANKER_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _load_cross_encoder(model_id: str, device: str) -> Any:
    """Lazy ``CrossEncoder`` with a download lock (same pattern as embeddings)."""
    from sentence_transformers import CrossEncoder

    with _RERANKER_LOCK:
        logger.info("Loading cross-encoder %s on %s", model_id, device)
        return CrossEncoder(model_id, device=device, max_length=1024)


def _passage_for_rerank(hit: SearchHit) -> str:
    raw = (hit.raw_content or "").strip()
    text = raw if len(raw) >= 48 else (hit.snippet or hit.title or "").strip()
    if len(text) > _MAX_PASSAGE_CHARS:
        return f"{text[:_MAX_PASSAGE_CHARS]}…"
    return text


def _min_max_unit(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rerank_search_hits(
    query: str,
    hits: list[SearchHit],
    *,
    top_k: int = 5,
    cross_encoder: Any | None = None,
) -> list[SearchHit]:
    """Re-order ``hits`` by cross-encoder relevance; keep top ``top_k``.

    Scores in the output are min--max normalised to ``[0, 1]`` over the
    returned slice. When ``top_k`` exceeds available hits, the list is
    simply truncated.
    """
    if not query or not query.strip():
        return hits[:top_k]
    if not hits:
        return []
    if top_k < 1:
        return []

    k = min(top_k, len(hits))
    if len(hits) == 1:
        h = hits[0]
        return [h.model_copy(update={"score": 1.0})]

    settings = get_settings()
    model_id = settings.reranker_model
    device = str(settings.reranker_device)
    ce = cross_encoder if cross_encoder is not None else _load_cross_encoder(model_id, device)

    passages = [_passage_for_rerank(h) for h in hits]
    pairs: list[list[str]] = [[query, p] for p in passages]
    raw_arr = np.asarray(ce.predict(pairs), dtype=np.float64)
    if raw_arr.size == 0:
        return hits[:k]

    scores_list = [float(s) for s in raw_arr.ravel()]
    if len(scores_list) != len(hits):
        logger.warning("reranker score count mismatch; falling back to order")
        return hits[:k]

    ranked = sorted(
        zip(hits, scores_list, strict=True),
        key=lambda t: t[1],
        reverse=True,
    )
    top = ranked[:k]
    out_scores = [t[1] for t in top]
    normalised = _min_max_unit(out_scores)
    out: list[SearchHit] = []
    for (h, _), s in zip(top, normalised, strict=True):
        out.append(h.model_copy(update={"score": s}))
    return out


def rerank_hybrid_results(
    query: str,
    pool: list[HybridSearchResult],
    *,
    top_k: int = 20,
    cross_encoder: Any | None = None,
) -> list[HybridSearchResult]:
    """Re-rank stage-1 hybrid rows with the cross-encoder; return top ``top_k`` chunks.

    Preserves :class:`HybridSearchResult` so eval can read ``metadata[\"source_id\"]``.
    """
    if not query or not query.strip():
        return pool[:top_k] if top_k > 0 else []
    if not pool:
        return []
    if top_k < 1:
        return []
    k = min(top_k, len(pool))
    if len(pool) == 1:
        return pool[:1]

    settings = get_settings()
    model_id = settings.reranker_model
    device = str(settings.reranker_device)
    ce = cross_encoder if cross_encoder is not None else _load_cross_encoder(model_id, device)

    hits = [hybrid_result_to_search_hit(r) for r in pool]
    passages = [_passage_for_rerank(h) for h in hits]
    pairs: list[list[str]] = [[query, p] for p in passages]
    raw_arr = np.asarray(ce.predict(pairs), dtype=np.float64)
    if raw_arr.size == 0:
        return pool[:k]

    scores_list = [float(s) for s in raw_arr.ravel()]
    if len(scores_list) != len(pool):
        logger.warning("rerank_hybrid_results: score count mismatch; using stage-1 order")
        return pool[:k]

    order = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)[:k]
    return [pool[i] for i in order]
