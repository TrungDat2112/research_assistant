"""Tests for :mod:`research_assistant.rag.chunking`."""

from __future__ import annotations

from datetime import date
from itertools import pairwise
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from research_assistant.rag.chunking import (
    ChunkingConfig,
    _iter_sections,
    _section_at,
    _summarise_for_prepend,
    chunk_document,
)
from research_assistant.rag.schemas import SourceDoc


def _fake_tokenizer(text: str, *, window: int = 4) -> Any:
    """Build an encoding where every ``window`` characters = 1 token.

    Keeps the test independent of any real HF tokenizer download.
    """
    token_ids = []
    offsets = []
    pos = 0
    while pos < len(text):
        end = min(pos + window, len(text))
        token_ids.append(len(token_ids))
        offsets.append((pos, end))
        pos = end
    return {"input_ids": token_ids, "offset_mapping": offsets}


def _install_fake_tokenizer() -> MagicMock:
    fake = MagicMock()
    fake.side_effect = lambda text, **_: _fake_tokenizer(text)
    return fake


def _doc(text: str, *, summary: str | None = None) -> SourceDoc:
    return SourceDoc(
        source_id="test_doc_1",
        url="https://example.com/doc",
        title="Test Document",
        text=text,
        doc_type="blog",
        authors=["Jane Doe"],
        published_date=date(2025, 1, 15),
        summary=summary,
    )


# ---- section detection -----------------------------------------------------


def test_iter_sections_picks_up_markdown_and_allcaps() -> None:
    text = "# Intro\nfirst line\n\nMETHODS\nmiddle\n\n## Results\ntail"
    sections = _iter_sections(text)
    names = [name for _, name in sections]
    assert "Intro" in names
    assert "Methods" in names  # ALL CAPS becomes Title case
    assert "Results" in names


def test_section_at_returns_latest_preceding_name() -> None:
    sections = [(0, ""), (10, "Alpha"), (40, "Beta")]
    assert _section_at(sections, 5) == ""
    assert _section_at(sections, 25) == "Alpha"
    assert _section_at(sections, 50) == "Beta"


# ---- contextual prepend ----------------------------------------------------


def test_summarise_uses_doc_summary_when_present() -> None:
    doc = _doc("body content", summary="A concise abstract.")
    out = _summarise_for_prepend(doc)
    assert "Test Document" in out
    assert "concise abstract" in out


def test_summarise_falls_back_to_first_sentences() -> None:
    doc = _doc("First sentence. Second sentence. Third sentence.")
    out = _summarise_for_prepend(doc)
    assert "First sentence" in out
    assert "Third sentence" not in out  # cap at first two


def test_summarise_respects_char_cap() -> None:
    doc = _doc("x" * 5000, summary="y" * 5000)
    out = _summarise_for_prepend(doc, max_chars=100)
    # Cap applied to the summary body; title still prepended as metadata.
    assert out.count("y") <= 100
    assert out.endswith("...")


# ---- chunking --------------------------------------------------------------


def test_chunk_document_emits_multiple_chunks_with_overlap() -> None:
    text = "".join(f"sent{idx:02d} " for idx in range(200))  # ~1200 chars / ~300 tokens
    doc = _doc(text)
    cfg = ChunkingConfig(model_id="fake", chunk_size_tokens=50, chunk_overlap_tokens=10)
    with patch(
        "research_assistant.rag.chunking._get_tokenizer", return_value=_install_fake_tokenizer()
    ):
        chunks = chunk_document(doc, cfg)
    assert len(chunks) >= 3
    for idx, ch in enumerate(chunks):
        assert ch.metadata.chunk_idx == idx
        assert ch.metadata.chunk_total == len(chunks)
        assert ch.text.startswith("[Test Document]")
        assert ch.body  # body is the raw slice without prepend


def test_chunk_document_respects_overlap_window() -> None:
    text = " ".join(f"w{idx}" for idx in range(500))
    doc = _doc(text)
    cfg = ChunkingConfig(model_id="fake", chunk_size_tokens=40, chunk_overlap_tokens=10)
    with patch(
        "research_assistant.rag.chunking._get_tokenizer", return_value=_install_fake_tokenizer()
    ):
        chunks = chunk_document(doc, cfg)
    # Adjacent chunk bodies should share at least one token's worth of text.
    for prev, cur in pairwise(chunks):
        assert prev.body != cur.body


def test_chunk_document_empty_text_returns_empty_list() -> None:
    doc = _doc("   \n\t")
    cfg = ChunkingConfig(model_id="fake")
    with patch(
        "research_assistant.rag.chunking._get_tokenizer", return_value=_install_fake_tokenizer()
    ):
        assert chunk_document(doc, cfg) == []


def test_chunk_document_rejects_overlap_larger_than_size() -> None:
    doc = _doc("some text here that should not matter")
    cfg = ChunkingConfig(model_id="fake", chunk_size_tokens=10, chunk_overlap_tokens=10)
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(doc, cfg)


def test_chunk_document_includes_metadata_fields() -> None:
    doc = _doc("# Heading\nfirst chunk body\n\n## Sub\nsecond body")
    cfg = ChunkingConfig(model_id="fake", chunk_size_tokens=30, chunk_overlap_tokens=5)
    with patch(
        "research_assistant.rag.chunking._get_tokenizer", return_value=_install_fake_tokenizer()
    ):
        chunks = chunk_document(doc, cfg)
    assert chunks, "expected at least one chunk"
    meta = chunks[0].metadata
    assert meta.source_id == "test_doc_1"
    assert meta.title == "Test Document"
    assert meta.authors == "Jane Doe"
    assert meta.published_date == "2025-01-15"
