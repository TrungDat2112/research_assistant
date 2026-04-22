"""IR-style metrics for ranked chunk lists with binary (source-level) relevance."""

from __future__ import annotations

import math
from typing import TypeVar

_T = TypeVar("_T", bound=str)


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain; binary or graded relevance at each rank (0-based)."""
    if k <= 0:
        return 0.0
    s = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            s += rel / math.log2(i + 2)
    return s


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """NDCG@k: DCG@k / IDCG@k; IDCG = DCG of ideal (relevances sorted descending)."""
    if k <= 0:
        return 0.0
    rels = relevances[:k]
    if not rels or max(rels) == 0:
        return 0.0
    d = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return d / idcg if idcg > 0 else 0.0


def source_recall_in_top_k(
    ranked_source_ids: list[_T],
    gold_source_ids: set[_T],
    k: int,
) -> float:
    """|gold ∩ (first k chunks' sources)| / |gold|; gold must be non-empty for callers.

    A single document may appear in multiple top-k chunks; the union of sources
    in positions ``1..k`` is used.
    """
    if not gold_source_ids:
        return 0.0
    head = set(ranked_source_ids[:k])
    return len(gold_source_ids & head) / len(gold_source_ids)


def per_query_metrics(
    ranked_source_ids: list[str],
    gold_source_ids: set[str],
    k_list: tuple[int, ...] = (10, 20),
) -> dict[str, float]:
    """Compute recall@k and NDCG@10 (from first-10 relevances) for one query."""
    k_max = max(k_list) if k_list else 10
    rels = [1 if s in gold_source_ids else 0 for s in ranked_source_ids[:k_max]]
    while len(rels) < k_max:
        rels.append(0)
    rels_10 = rels[:10]
    out: dict[str, float] = {"ndcg@10": ndcg_at_k(rels_10, 10)}
    for k in k_list:
        if k > 0:
            out[f"recall@{k}"] = source_recall_in_top_k(
                ranked_source_ids,
                gold_source_ids,
                k,
            )
    return out
