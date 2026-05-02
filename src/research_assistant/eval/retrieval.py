from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from research_assistant.eval.metrics import per_query_metrics
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.embedding import EmbeddingModel
from research_assistant.rag.hybrid import hybrid_search_stage1
from research_assistant.rag.hyde import dense_embedding_for_retrieval
from research_assistant.rag.reranker import rerank_hybrid_results
from research_assistant.rag.vector_store import ChromaStore


class RetrievalEvalItem(BaseModel):
    id: str
    query: str
    language: Literal["en", "vi"] = "en"
    relevant_source_ids: list[str] = Field(min_length=1)


class RetrievalEvalFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int
    items: list[RetrievalEvalItem] = Field(min_length=1)


def load_retrieval_eval(path: Path) -> list[RetrievalEvalItem]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    payload = RetrievalEvalFile.model_validate(data)
    return payload.items


def iter_source_ids_with_meta(
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    query: str,
    *,
    final_top_k: int = 20,
    query_vec: NDArray[np.float32] | None = None,
    where: dict[str, Any] | None = None,
    use_hyde: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    hyde_meta: dict[str, Any] = {}
    if query_vec is None:
        qvec, hyde_meta = dense_embedding_for_retrieval(
            query,
            store,
            bm25_index,
            embedder,
            where=where,
            hyde_enabled=use_hyde,
        )
    else:
        qvec = query_vec
        hyde_meta = {"hyde_applied": False, "hyde_reason": "query_vec_provided"}
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
    ranked = [str(h.metadata.get("source_id", "")).strip() for h in hits]
    return ranked, hyde_meta


def iter_source_ids(
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    query: str,
    *,
    final_top_k: int = 20,
    query_vec: NDArray[np.float32] | None = None,
    where: dict[str, Any] | None = None,
    use_hyde: bool = False,
) -> list[str]:
    ranked, _ = iter_source_ids_with_meta(
        store,
        bm25_index,
        embedder,
        query,
        final_top_k=final_top_k,
        query_vec=query_vec,
        where=where,
        use_hyde=use_hyde,
    )
    return ranked


def iter_source_ids_reranked(
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    query: str,
    *,
    final_top_k: int = 20,
    query_vec: NDArray[np.float32] | None = None,
    where: dict[str, Any] | None = None,
    candidate_pool: int = 20,
    cross_encoder: Any | None = None,
    use_hyde: bool = False,
) -> list[str]:
    if final_top_k > candidate_pool:
        raise ValueError("final_top_k must be <= candidate_pool for re-rank eval")
    if query_vec is None:
        qvec, _ = dense_embedding_for_retrieval(
            query,
            store,
            bm25_index,
            embedder,
            where=where,
            hyde_enabled=use_hyde,
        )
    else:
        qvec = query_vec
    pool = hybrid_search_stage1(
        store,
        bm25_index,
        query,
        qvec,
        dense_top_k=50,
        bm25_top_k=50,
        final_top_k=candidate_pool,
        where=where,
    )
    reranked = rerank_hybrid_results(
        query,
        pool,
        top_k=final_top_k,
        cross_encoder=cross_encoder,
    )
    return [str(h.metadata.get("source_id", "")).strip() for h in reranked]


def run_hybrid_retrieval_eval(
    *,
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    items: list[RetrievalEvalItem],
    final_top_k: int = 20,
    k_recall: tuple[int, ...] = (10, 20),
    use_hyde: bool = False,
) -> dict[str, Any]:
    if final_top_k < max(k_recall, default=10):
        raise ValueError("final_top_k must be >= max(k_recall)")

    per_query: list[dict[str, Any]] = []
    for it in items:
        gold: set[str] = set(it.relevant_source_ids)
        ranked, hyde_meta = iter_source_ids_with_meta(
            store,
            bm25_index,
            embedder,
            it.query,
            final_top_k=final_top_k,
            use_hyde=use_hyde,
        )
        m = per_query_metrics(ranked, gold, k_list=k_recall)
        per_query.append(
            {
                "id": it.id,
                "hyde_applied": bool(hyde_meta.get("hyde_applied")),
                "hyde_reason": hyde_meta.get("hyde_reason"),
                **m,
            },
        )

    n = len(per_query)
    if n == 0:
        return {"n": 0, "error": "no items", "per_query": []}

    ndcgs = [float(p["ndcg@10"]) for p in per_query]
    mrrs = [float(p["mrr"]) for p in per_query]
    precs5 = [float(p["precision@5"]) for p in per_query]
    n_hyde = sum(1 for p in per_query if p.get("hyde_applied"))
    out: dict[str, Any] = {
        "n": n,
        "mode": "stage1_hybrid_hyde" if use_hyde else "stage1_hybrid",
        "use_hyde": use_hyde,
        "n_hyde_triggers": n_hyde,
        "final_top_k": final_top_k,
        "k_recall": list(k_recall),
        "mean_ndcg@10": fmean(ndcgs),
        "mean_mrr": fmean(mrrs),
        "mean_precision@5": fmean(precs5),
        "per_query": per_query,
    }
    for k in k_recall:
        rk = f"recall@{k}"
        out[f"mean_{rk}"] = fmean([float(p[rk]) for p in per_query])
    return out


def run_rerank_retrieval_eval(
    *,
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    items: list[RetrievalEvalItem],
    candidate_pool: int = 20,
    final_top_k: int = 20,
    k_recall: tuple[int, ...] = (10, 20),
    cross_encoder: Any | None = None,
    use_hyde: bool = False,
) -> dict[str, Any]:
    if final_top_k < max(k_recall, default=10):
        raise ValueError("final_top_k must be >= max(k_recall)")
    if final_top_k > candidate_pool:
        raise ValueError("final_top_k must be <= candidate_pool")

    per_query: list[dict[str, Any]] = []
    for it in items:
        gold: set[str] = set(it.relevant_source_ids)
        ranked = iter_source_ids_reranked(
            store,
            bm25_index,
            embedder,
            it.query,
            final_top_k=final_top_k,
            candidate_pool=candidate_pool,
            cross_encoder=cross_encoder,
            use_hyde=use_hyde,
        )
        m = per_query_metrics(ranked, gold, k_list=k_recall)
        per_query.append({"id": it.id, **m})

    n = len(per_query)
    if n == 0:
        return {"n": 0, "error": "no items", "per_query": []}

    ndcgs = [float(p["ndcg@10"]) for p in per_query]
    mrrs = [float(p["mrr"]) for p in per_query]
    precs5 = [float(p["precision@5"]) for p in per_query]
    mode = "stage1_hybrid_rerank"
    if use_hyde:
        mode = "stage1_hybrid_hyde_rerank"
    out: dict[str, Any] = {
        "n": n,
        "mode": mode,
        "use_hyde": use_hyde,
        "candidate_pool": candidate_pool,
        "final_top_k": final_top_k,
        "k_recall": list(k_recall),
        "mean_ndcg@10": fmean(ndcgs),
        "mean_mrr": fmean(mrrs),
        "mean_precision@5": fmean(precs5),
        "per_query": per_query,
    }
    for k in k_recall:
        rk = f"recall@{k}"
        out[f"mean_{rk}"] = fmean([float(p[rk]) for p in per_query])
    return out
