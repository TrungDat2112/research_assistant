"""Tests for :mod:`research_assistant.tools.vector_search`."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.schemas import Chunk, ChunkMetadata
from research_assistant.rag.vector_store import ChromaStore
from research_assistant.tools import vector_search as vs


def _chunk_body(chunk_id: str, body: str) -> Chunk:
    meta = ChunkMetadata(
        source_id="src",
        source_url="https://example.com/p1",
        title="T",
        doc_type="blog",
        chunk_idx=0,
        chunk_total=1,
        section="",
        authors="A",
        published_date=date(2025, 6, 1).isoformat(),
    )
    return Chunk(chunk_id=chunk_id, text=body, body=body, metadata=meta)


def test_vector_search_empty_corpus_returns_empty(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="empty_corpus")
    vs.clear_vector_search_cache()
    empty_idx = BM25CorpusIndex.from_chroma(store)
    assert empty_idx.size() == 0
    hits = vs.vector_search(
        "any query",
        top_k=5,
        store=store,
        bm25_index=empty_idx,
    )
    assert hits == []


def test_vector_search_hits_use_corpus_source(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="hit")
    c = _chunk_body("only", "RAG retrieval augmented generation survey paper")
    store.upsert_chunks([c], np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    idx = BM25CorpusIndex.from_chroma(store)

    class _StubEmb:
        def __init__(self) -> None:
            self.model_id = "stub"
            self.device = "cpu"

        def embed_query(self, _text: str) -> NDArray[np.float32]:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    vs.clear_vector_search_cache()
    hits = vs.vector_search(
        "RAG augmented generation",
        top_k=3,
        store=store,
        embedder=_StubEmb(),
        bm25_index=idx,
    )
    assert len(hits) == 1
    assert hits[0].source == "corpus"
    assert hits[0].title == "T"
    assert "rag" in hits[0].snippet.lower()
