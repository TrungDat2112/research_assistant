"""Tests for :mod:`research_assistant.rag.schemas`."""

from __future__ import annotations

from datetime import date

import pytest

from research_assistant.rag.schemas import Chunk, ChunkMetadata, SourceDoc


def test_source_doc_make_source_id_is_deterministic() -> None:
    url = "https://example.com/foo"
    a = SourceDoc.make_source_id(url)
    b = SourceDoc.make_source_id(url)
    assert a == b
    assert a.startswith("h_")
    assert SourceDoc.make_source_id("https://example.com/bar") != a


def test_source_doc_strips_blank_authors() -> None:
    doc = SourceDoc(
        source_id="doc",
        url="https://example.com/x",
        title="t",
        text="body",
        authors=["Alice  ", "", "  Bob"],
    )
    assert doc.authors == ["Alice", "Bob"]


def test_source_doc_requires_nonempty_text() -> None:
    with pytest.raises(ValueError):
        SourceDoc(source_id="id", url="https://x", title="t", text="")


def test_chunk_metadata_to_chroma_is_scalar_only() -> None:
    meta = ChunkMetadata(
        source_id="s",
        source_url="https://x",
        title="T",
        doc_type="arxiv",
        chunk_idx=3,
        chunk_total=7,
        section="Intro",
        authors="Alice; Bob",
        published_date="2025-03-01",
    )
    flat = meta.to_chroma()
    assert flat["chunk_idx"] == 3
    assert flat["doc_type"] == "arxiv"
    for value in flat.values():
        assert isinstance(value, (str, int, float, bool)), value


def test_chunk_make_chunk_id_handles_slashes() -> None:
    assert Chunk.make_chunk_id("arxiv/2005.11401", 7) == "arxiv_2005.11401:0007"


def test_chunk_roundtrip() -> None:
    meta = ChunkMetadata(
        source_id="src",
        source_url="https://example.com/s",
        title="T",
        doc_type="blog",
        chunk_idx=0,
        chunk_total=1,
        published_date=date(2024, 1, 1).isoformat(),
    )
    chunk = Chunk(
        chunk_id="src:0000",
        text="[T] summary\n\nbody",
        body="body",
        metadata=meta,
    )
    dumped = chunk.model_dump()
    restored = Chunk.model_validate(dumped)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.metadata.source_id == "src"
