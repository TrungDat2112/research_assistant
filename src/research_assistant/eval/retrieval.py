"""Load the Week-2 retrieval eval set and run stage-1 hybrid search metrics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from research_assistant.eval.metrics import per_query_metrics
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.embedding import EmbeddingModel
from research_assistant.rag.hybrid import hybrid_search_stage1
from research_assistant.rag.vector_store import ChromaStore


class RetrievalEvalItem(BaseModel):
    id: str
    query: str
    relevant_source_ids: list[str] = Field(min_length=1)


class RetrievalEvalFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int
    items: list[RetrievalEvalItem] = Field(min_length=1)


def load_retrieval_eval(path: Path) -> list[RetrievalEvalItem]:
    """Parse ``retrieval_eval_30.json`` (or compatible)."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    payload = RetrievalEvalFile.model_validate(data)
    return payload.items


def iter_source_ids(
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    query: str,
    *,
    final_top_k: int = 20,
    query_vec: NDArray[np.float32] | None = None,
    where: dict[str, Any] | None = None,
) -> list[str]:
    """Run stage-1 hybrid and return ``source_id`` for each chunk (ranked)."""
    qvec = embedder.embed_query(query) if query_vec is None else query_vec
    hits = hybrid_search_stage1(
        store,
        bm25_index,
        query,
        qvec,
        dense_top_k=50,
        bm25_top_k=50,
        final_top_k=final_top_k,
        where=where,
    )
    return [str(h.metadata.get("source_id", "")).strip() for h in hits]


def run_hybrid_retrieval_eval(
    *,
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    items: list[RetrievalEvalItem],
    final_top_k: int = 20,
    k_recall: tuple[int, ...] = (10, 20),
) -> dict[str, Any]:
    """Macro-averaged recall@k and mean NDCG@10; stage-1 hybrid only (no re-rank)."""
    if final_top_k < max(k_recall, default=10):
        raise ValueError("final_top_k must be >= max(k_recall)")

    per_query: list[dict[str, Any]] = []
    for it in items:
        gold: set[str] = set(it.relevant_source_ids)
        ranked = iter_source_ids(
            store,
            bm25_index,
            embedder,
            it.query,
            final_top_k=final_top_k,
        )
        m = per_query_metrics(ranked, gold, k_list=k_recall)
        per_query.append({"id": it.id, **m})

    n = len(per_query)
    if n == 0:
        return {"n": 0, "error": "no items", "per_query": []}

    ndcgs = [float(p["ndcg@10"]) for p in per_query]
    out: dict[str, Any] = {
        "n": n,
        "final_top_k": final_top_k,
        "k_recall": list(k_recall),
        "mean_ndcg@10": fmean(ndcgs),
        "per_query": per_query,
    }
    for k in k_recall:
        rk = f"recall@{k}"
        out[f"mean_{rk}"] = fmean([float(p[rk]) for p in per_query])
    return out
