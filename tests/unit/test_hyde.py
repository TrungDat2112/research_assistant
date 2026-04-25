"""Tests for HyDE probe and dense embedding selection."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import SecretStr

from research_assistant.config import Settings, get_settings
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.hybrid import HybridSearchResult
from research_assistant.rag.hyde import (
    dense_embedding_for_retrieval,
    generate_hyde_passage,
    hyde_probe_triggers,
)
from research_assistant.rag.schemas import Chunk, ChunkMetadata
from research_assistant.rag.vector_store import ChromaStore


def _chunk(chunk_id: str, body: str, source_id: str = "s1") -> Chunk:
    meta = ChunkMetadata(
        source_id=source_id,
        source_url="https://example.com/p",
        title="T",
        doc_type="blog",
        chunk_idx=0,
        chunk_total=1,
        section="",
        authors="A",
        published_date=date(2025, 6, 1).isoformat(),
    )
    return Chunk(chunk_id=chunk_id, text=body, body=body, metadata=meta)


def test_hyde_probe_triggers_empty() -> None:
    assert hyde_probe_triggers([], min_top1_fused_score=0.5, min_fused_margin=0.05) is True


def test_hyde_probe_triggers_low_top1() -> None:
    probe = [
        HybridSearchResult(
            chunk_id="a",
            body="",
            metadata={},
            dense_distance=0.5,
            combined_score=0.2,
            dense_norm=0.2,
            bm25_norm=0.2,
        ),
        HybridSearchResult(
            chunk_id="b",
            body="",
            metadata={},
            dense_distance=0.6,
            combined_score=0.19,
            dense_norm=0.1,
            bm25_norm=0.1,
        ),
    ]
    assert hyde_probe_triggers(probe, min_top1_fused_score=0.38, min_fused_margin=0.04) is True


def test_hyde_probe_skips_when_strong() -> None:
    probe = [
        HybridSearchResult(
            chunk_id="a",
            body="",
            metadata={},
            dense_distance=0.1,
            combined_score=0.95,
            dense_norm=0.9,
            bm25_norm=0.9,
        ),
        HybridSearchResult(
            chunk_id="b",
            body="",
            metadata={},
            dense_distance=0.5,
            combined_score=0.40,
            dense_norm=0.4,
            bm25_norm=0.4,
        ),
    ]
    assert hyde_probe_triggers(probe, min_top1_fused_score=0.38, min_fused_margin=0.04) is False


def test_dense_embedding_skips_hyde_when_disabled(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="corp1")
    c = _chunk("ch1", "neural retrieval dense vectors")
    store.upsert_chunks([c], np.array([[1.0, 0.0]], dtype=np.float32))
    idx = BM25CorpusIndex.from_chroma(store)

    class _Emb:
        model_id = "stub"
        device = "cpu"

        def embed_query(self, text: str) -> NDArray[np.float32]:
            if "HYPOTHETICAL" in text.upper():
                return np.array([0.0, 1.0], dtype=np.float32)
            return np.array([1.0, 0.0], dtype=np.float32)

    vec, meta = dense_embedding_for_retrieval(
        "machine learning",
        store,
        idx,
        _Emb(),
        hyde_enabled=False,
        settings=Settings(
            _env_file=None,  # type: ignore[call-arg]
            hyde_enabled=True,
            hyde_min_top1_fused_score=0.99,
            hyde_min_fused_margin=0.99,
        ),
    )
    assert meta["hyde_applied"] is False
    assert vec[0] == pytest.approx(1.0)


def test_dense_embedding_applies_stub_hypothesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="corp2")
    c = _chunk("ch1", "orthogonal dimension two", source_id="gold")
    store.upsert_chunks([c], np.array([[0.0, 1.0]], dtype=np.float32))
    idx = BM25CorpusIndex.from_chroma(store)

    class _Emb:
        model_id = "stub"
        device = "cpu"

        def embed_query(self, text: str) -> NDArray[np.float32]:
            if "HYPOTHETICAL" in text.upper():
                return np.array([0.0, 1.0], dtype=np.float32)
            return np.array([1.0, 0.0], dtype=np.float32)

    def _fake_hyp(_q: str) -> tuple[str, float]:
        return "HYPOTHETICAL passage about orthogonal dimension two.", 0.0

    monkeypatch.setattr(
        "research_assistant.rag.hyde.hyde_probe_triggers",
        lambda *_a, **_k: True,
    )
    vec, meta = dense_embedding_for_retrieval(
        "obscure query xyz123",
        store,
        idx,
        _Emb(),
        hyde_enabled=True,
        settings=Settings(
            _env_file=None,  # type: ignore[call-arg]
            hyde_enabled=True,
        ),
        hypothesis_fn=_fake_hyp,
    )
    assert meta["hyde_applied"] is True
    assert vec[1] == pytest.approx(1.0)


def test_generate_hyde_passage_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = Settings(
        _env_file=None,  # type: ignore[call-arg]
        anthropic_api_key=SecretStr(""),
    )
    monkeypatch.setattr(
        "research_assistant.agents._llm.get_settings",
        lambda: empty,
    )
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        generate_hyde_passage("test?", settings=empty)
