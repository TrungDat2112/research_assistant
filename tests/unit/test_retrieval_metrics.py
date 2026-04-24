"""Tests for :mod:`research_assistant.eval.metrics`."""

from __future__ import annotations

import math

import pytest

from research_assistant.eval.metrics import (
    dcg_at_k,
    ndcg_at_k,
    per_query_metrics,
    precision_at_k,
    reciprocal_rank_first_relevant,
    source_recall_in_top_k,
)


def test_dcg_perfect_is_one() -> None:
    rels = [1, 1, 1]
    d = dcg_at_k(rels, 3)
    expected = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert d == pytest.approx(expected)


def test_ndcg_perfect() -> None:
    # Single relevant at rank 1 — already ideal
    rels = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ndcg_at_k(rels, 10) == pytest.approx(1.0)


def test_ndcg_worst_ordering() -> None:
    # Two relevant: compare ideal [1,1,0,...] vs bad [0,0,...,1,1] at end of 10
    bad = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    good = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ndcg_at_k(good, 10) == pytest.approx(1.0)
    assert ndcg_at_k(bad, 10) < ndcg_at_k(good, 10)


def test_source_recall() -> None:
    ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    gold = {"x", "c", "d"}
    # first 10 contain c, d but not x → 2/3
    assert source_recall_in_top_k(ranked, gold, 10) == pytest.approx(2.0 / 3.0)
    # first 2: only a,b → 0/3
    assert source_recall_in_top_k(ranked, gold, 2) == 0.0


def test_mrr_and_precision() -> None:
    ranked = ["a", "b", "c", "gold", "d"]
    gold = {"gold", "x"}
    assert reciprocal_rank_first_relevant(ranked, gold) == pytest.approx(1.0 / 4.0)
    assert precision_at_k(ranked, gold, 5) == pytest.approx(0.2)


def test_per_query_metrics() -> None:
    ranked = [f"s{i}" for i in range(1, 21)]
    gold = {"s3"}
    m = per_query_metrics(ranked, gold, k_list=(10, 20))
    assert m["recall@10"] == 1.0  # s3 in position 2 (index 1)
    assert m["recall@20"] == 1.0
    assert 0.0 < m["ndcg@10"] <= 1.0
    assert m["mrr"] == pytest.approx(1.0 / 3.0)
    # Top 5: s1..s5 -> one relevant (s3)
    assert m["precision@5"] == pytest.approx(0.2)
