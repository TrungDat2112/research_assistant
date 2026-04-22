"""Tests for :mod:`research_assistant.rag.vector_store`.

Uses a real Chroma PersistentClient on a tmp path — Chroma is pure-Python /
sqlite-backed so this stays fast (~150 ms). Embeddings are fixed vectors.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from research_assistant.rag.schemas import Chunk, ChunkMetadata
from research_assistant.rag.vector_store import ChromaStore


def _chunk(chunk_id: str, idx: int, body: str) -> Chunk:
    meta = ChunkMetadata(
        source_id="src",
        source_url="https://example.com/s",
        title="Doc",
        doc_type="blog",
        chunk_idx=idx,
        chunk_total=3,
        section="",
        authors="Alice",
        published_date=date(2025, 6, 1).isoformat(),
    )
    return Chunk(chunk_id=chunk_id, text=body, body=body, metadata=meta)


def test_upsert_then_search_returns_nearest(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="test_coll")
    chunks = [
        _chunk("src:0000", 0, "alpha topic text"),
        _chunk("src:0001", 1, "beta topic text"),
        _chunk("src:0002", 2, "gamma topic text"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    store.upsert_chunks(chunks, embeddings)

    assert store.count() == 3
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    hits = store.search(query, top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "src:0000"
    assert hits[0].body == "alpha topic text"
    assert hits[0].metadata["source_id"] == "src"


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="idem")
    c = _chunk("src:0000", 0, "only chunk")
    emb = np.array([[1.0, 0.0]], dtype=np.float32)
    store.upsert_chunks([c], emb)
    store.upsert_chunks([c], emb)
    assert store.count() == 1


def test_length_mismatch_raises(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="mismatch")
    c = _chunk("src:0000", 0, "x")
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert_chunks([c], emb)


def test_reset_clears_collection(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="reset_me")
    c = _chunk("src:0000", 0, "first")
    store.upsert_chunks([c], np.array([[1.0, 0.0]], dtype=np.float32))
    assert store.count() == 1
    store.reset()
    assert store.count() == 0


def test_where_filter_restricts_results(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="filter_me")
    blog = _chunk("src:0000", 0, "blog chunk")
    blog.metadata.doc_type = "blog"  # already blog
    paper = _chunk("src:0001", 1, "paper chunk")
    paper.metadata.doc_type = "arxiv"
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    store.upsert_chunks([blog, paper], embeddings)

    hits = store.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=5,
        where={"doc_type": "arxiv"},
    )
    assert len(hits) == 1
    assert hits[0].chunk_id == "src:0001"
