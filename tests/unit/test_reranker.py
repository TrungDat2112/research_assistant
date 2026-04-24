"""Unit tests for :mod:`research_assistant.rag.reranker`."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from research_assistant.config import get_settings
from research_assistant.graph.state import SearchHit
from research_assistant.rag.hybrid import HybridSearchResult
from research_assistant.rag.reranker import (
    _passage_for_rerank,
    rerank_hybrid_results,
    rerank_search_hits,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def _hit(
    url: str, snippet: str, raw: str | None = "full body " * 20, source: str = "corpus"
) -> SearchHit:
    return SearchHit(
        url=url,  # type: ignore[arg-type]
        title="T",
        snippet=snippet,
        source=source,  # type: ignore[arg-type]
        raw_content=raw,
    )


def test_passage_prefers_raw_content() -> None:
    h = _hit("https://a.example", "snip", raw="X" * 100)
    assert _passage_for_rerank(h).startswith("X")


def test_passage_falls_back_to_snippet() -> None:
    h = SearchHit(
        url="https://a.example",  # type: ignore[arg-type]
        title="T",
        snippet="expected snip",
        raw_content="short",
        source="corpus",  # type: ignore[arg-type]
    )
    assert _passage_for_rerank(h) == "expected snip"


def test_rerank_orders_by_cross_encoder_score() -> None:
    h1 = _hit("https://a.example/x", "low")
    h2 = _hit("https://a.example/y", "high")
    h3 = _hit("https://a.example/z", "mid")
    mock_ce = MagicMock()
    mock_ce.predict = MagicMock(return_value=np.array([0.1, 0.99, 0.5], dtype=np.float32))

    out = rerank_search_hits("q", [h1, h2, h3], top_k=2, cross_encoder=mock_ce)
    assert len(out) == 2
    assert str(out[0].url) == "https://a.example/y"
    assert str(out[1].url) == "https://a.example/z"
    assert out[0].score is not None and 0.0 <= float(out[0].score) <= 1.0
    assert mock_ce.predict.call_count == 1


def test_rerank_single_hit_sets_score() -> None:
    h = _hit("https://solo.example", "s")
    mock_ce = MagicMock()
    out = rerank_search_hits("q", [h], top_k=3, cross_encoder=mock_ce)
    assert len(out) == 1
    assert out[0].score == pytest.approx(1.0)
    assert mock_ce.predict.call_count == 0


def _hr(chunk_id: str, url: str, body: str, sid: str = "s1") -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        body=body,
        metadata={"source_id": sid, "source_url": url, "title": "T"},
        dense_distance=0.1,
        combined_score=0.5,
        dense_norm=0.5,
        bm25_norm=0.5,
    )


def test_rerank_hybrid_reorders_by_cross_encoder() -> None:
    a = _hr("c1", "https://a.example", "low " * 30, "src_a")
    b = _hr("c2", "https://b.example", "high " * 30, "src_b")
    c = _hr("c3", "https://c.example", "mid " * 30, "src_c")
    mock_ce = MagicMock()
    mock_ce.predict = MagicMock(return_value=np.array([0.1, 0.99, 0.5], dtype=np.float32))
    out = rerank_hybrid_results("q", [a, b, c], top_k=2, cross_encoder=mock_ce)
    assert [x.chunk_id for x in out] == ["c2", "c3"]
    assert out[0].metadata.get("source_id") == "src_b"
    assert mock_ce.predict.call_count == 1
