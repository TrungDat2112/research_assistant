from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research_assistant.graph.state import SearchHit
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.vector_store import ChromaStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    body: str
    metadata: dict[str, Any]
    dense_distance: float | None
    combined_score: float
    dense_norm: float
    bm25_norm: float


def _min_max_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return dict.fromkeys(scores, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _dense_similarity(distance: float) -> float:
    sim = 1.0 - float(distance)
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def hybrid_search_stage1(
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    query: str,
    query_embedding: NDArray[np.float32],
    *,
    dense_top_k: int = 50,
    bm25_top_k: int = 50,
    weight_dense: float = 0.5,
    weight_bm25: float = 0.5,
    final_top_k: int = 20,
    where: dict[str, Any] | None = None,
) -> list[HybridSearchResult]:
    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive")
    if dense_top_k <= 0 or bm25_top_k <= 0:
        raise ValueError("dense_top_k and bm25_top_k must be positive")
    w_d = float(weight_dense)
    w_b = float(weight_bm25)
    if w_d < 0 or w_b < 0 or (w_d + w_b) <= 0:
        raise ValueError("weight_dense and weight_bm25 must be non-negative and sum to > 0")
    sum_w = w_d + w_b

    dense_hits: list[SearchResult] = store.search(
        query_embedding,
        top_k=dense_top_k,
        where=where,
    )
    dense_sims: dict[str, float] = {}
    for h in dense_hits:
        dense_sims[h.chunk_id] = _dense_similarity(h.distance)

    bm25_pairs = bm25_index.top_n(query, bm25_top_k)
    bm25_raw: dict[str, float] = dict(bm25_pairs)

    norm_d = _min_max_norm(dense_sims)
    norm_b = _min_max_norm(bm25_raw)

    candidate_ids = set(dense_sims) | set(bm25_raw)
    combined: dict[str, float] = {}
    for cid in candidate_ids:
        d = norm_d.get(cid, 0.0)
        b = norm_b.get(cid, 0.0)
        combined[cid] = (w_d * d + w_b * b) / sum_w

    sorted_ids = sorted(
        combined.keys(),
        key=lambda x: combined[x],
        reverse=True,
    )

    by_id: dict[str, SearchResult] = {h.chunk_id: h for h in dense_hits}
    need = [i for i in sorted_ids[:final_top_k] if i not in by_id]
    if need:
        for extra in store.get_by_ids(need):
            by_id[extra.chunk_id] = extra

    out: list[HybridSearchResult] = []
    for cid in sorted_ids[:final_top_k]:
        row = by_id.get(cid)
        if row is None:
            logger.warning("Hybrid: chunk_id=%s missing from store after merge; skipping", cid)
            continue
        dist = next((h.distance for h in dense_hits if h.chunk_id == cid), None)
        d_n = norm_d.get(cid, 0.0)
        b_n = norm_b.get(cid, 0.0)
        out.append(
            HybridSearchResult(
                chunk_id=cid,
                body=row.body,
                metadata=row.metadata,
                dense_distance=dist,
                combined_score=float(combined[cid]),
                dense_norm=d_n,
                bm25_norm=b_n,
            ),
        )
    return out


def hybrid_result_to_search_hit(r: HybridSearchResult) -> SearchHit:
    from pydantic import HttpUrl, TypeAdapter

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
