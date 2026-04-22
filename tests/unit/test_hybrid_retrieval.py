"""Tests for BM25 index + stage-1 hybrid fusion."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from research_assistant.rag.bm25_index import BM25CorpusIndex, tokenize_for_bm25
from research_assistant.rag.hybrid import hybrid_search_stage1
from research_assistant.rag.schemas import Chunk, ChunkMetadata
from research_assistant.rag.vector_store import ChromaStore


def _chunk(
    chunk_id: str,
    idx: int,
    body: str,
    *,
    title: str = "Doc",
    url: str = "https://example.com/p",
) -> Chunk:
    meta = ChunkMetadata(
        source_id="src",
        source_url=url,
        title=title,
        doc_type="blog",
        chunk_idx=idx,
        chunk_total=1,
        section="",
        authors="A",
        published_date=date(2025, 6, 1).isoformat(),
    )
    return Chunk(chunk_id=chunk_id, text=body, body=body, metadata=meta)


def test_tokenize_for_bm25_splits_words() -> None:
    t = tokenize_for_bm25("LoRA/QLoRA: fine-tuning!")
    assert "lora" in t
    assert "qlora" in t or "lora" in t


def test_bm25_corpus_index_top_n(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="b25")
    bodies = [
        "LoRA low rank adaptation trains small matrices",
        "BM25 is a lexical bag of words retriever",
        "GraphRAG uses knowledge graphs and retrieval",
    ]
    chunks = [_chunk(f"id:{i}", i, bodies[i], title=f"T{i}") for i in range(3)]
    emb = np.eye(3, dtype=np.float32)
    store.upsert_chunks(chunks, emb)
    idx = BM25CorpusIndex.from_chroma(store)
    top = idx.top_n("LoRA train matrices", 2)
    assert len(top) == 2
    assert top[0][0] == "id:0"


def test_hybrid_prefers_both_signals(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="hyb")
    chunks = [
        _chunk("ch:a", 0, "python asyncio networking library"),
        _chunk("ch:b", 1, "graph algorithms breadth first search"),
        _chunk("ch:c", 2, "neural network training optimizer adam"),
    ]
    emb = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    store.upsert_chunks(chunks, emb)
    idx = BM25CorpusIndex.from_chroma(store)

    # Query aligned with chunk "b" in embedding space, lexical "graph" + "search".
    q = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = hybrid_search_stage1(
        store,
        idx,
        "graph search",
        q,
        dense_top_k=3,
        bm25_top_k=3,
        weight_dense=0.5,
        weight_bm25=0.5,
        final_top_k=1,
    )
    assert len(out) == 1
    assert out[0].chunk_id == "ch:b"


def test_fetch_all_get_by_ids_roundtrip(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="all")
    c = _chunk("x:0", 0, "hello world body text")
    store.upsert_chunks([c], np.array([[1.0, 0.0]], dtype=np.float32))
    all_rows = store.fetch_all_documents()
    assert len(all_rows) == 1
    by_id = store.get_by_ids(["x:0"])
    assert len(by_id) == 1
    assert by_id[0].body == "hello world body text"


def test_hybrid_rejects_invalid_topk(tmp_path: Path) -> None:
    store = ChromaStore(persist_dir=tmp_path, collection="inv")
    idx = BM25CorpusIndex.from_chroma(store)
    with pytest.raises(ValueError, match="final_top_k"):
        hybrid_search_stage1(
            store,
            idx,
            "q",
            np.array([1.0], dtype=np.float32),
            final_top_k=0,
        )
