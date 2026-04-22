"""Pydantic value objects for the RAG ingestion pipeline.

Sits below chunking / embedding / vector store — everything upstream emits
these models, everything downstream consumes them. Keep it provider-agnostic
so arXiv / HTML / future loaders share one contract.

The split is intentional:
  * :class:`SourceDoc` — a **whole document** post-cleaning (no chunking yet).
  * :class:`Chunk` — a **slice** of a document ready to embed + index.
  * :class:`ChunkMetadata` — flat, string-only fields (Chroma's metadata
    column accepts scalars only, so we serialise lists/dates to strings).
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DocType = Literal["arxiv", "blog", "html", "pdf", "other"]


class SourceDoc(BaseModel):
    """A single cleaned document ready for chunking.

    ``source_id`` is the stable identifier used everywhere downstream
    (chunk ids, dedup keys). For arXiv papers it's the arXiv id
    (``2404.16130v2``); for HTML it's a SHA-1 of the canonical URL.
    """

    source_id: str = Field(..., min_length=3, max_length=256)
    url: str = Field(..., description="Canonical source URL (may be a DOI/arxiv link).")
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, description="Cleaned body text, one doc worth.")
    doc_type: DocType = "other"
    authors: list[str] = Field(default_factory=list)
    published_date: date | None = None
    summary: str | None = Field(
        default=None,
        description="Abstract / lead paragraph used as contextual prepend "
        "on every chunk (ADR-003).",
    )

    @field_validator("authors")
    @classmethod
    def _strip_authors(cls, value: list[str]) -> list[str]:
        return [a.strip() for a in value if a and a.strip()]

    @classmethod
    def make_source_id(cls, url: str) -> str:
        """Deterministic id from a URL — useful for HTML sources without ids."""
        return "h_" + hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class ChunkMetadata(BaseModel):
    """Flat metadata attached to every chunk.

    Must stay serialisable to ``dict[str, str | int | float | bool]`` because
    Chroma (and most vector DBs) reject nested structures in metadata.
    """

    source_id: str
    source_url: str
    title: str
    doc_type: DocType
    chunk_idx: int = Field(..., ge=0)
    chunk_total: int = Field(..., ge=1)
    section: str = Field(default="", description="Section heading if detected.")
    authors: str = Field(default="", description="Semicolon-joined author list.")
    published_date: str = Field(default="", description="ISO-8601 date or empty string.")

    def to_chroma(self) -> dict[str, str | int | float | bool]:
        """Project to the scalar-only dict Chroma expects."""
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "title": self.title,
            "doc_type": self.doc_type,
            "chunk_idx": self.chunk_idx,
            "chunk_total": self.chunk_total,
            "section": self.section,
            "authors": self.authors,
            "published_date": self.published_date,
        }


class Chunk(BaseModel):
    """A piece of text ready to embed + index.

    ``text`` is the *embedding input* — it already includes the contextual
    prepend (doc summary / title) so retrieval sees the right context even
    when the slice is short (ADR-003). ``body`` is the raw slice without the
    prepend, kept for display / citation.
    """

    chunk_id: str = Field(..., pattern=r"^[A-Za-z0-9_\-:.]{3,128}$")
    text: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    metadata: ChunkMetadata

    @classmethod
    def make_chunk_id(cls, source_id: str, chunk_idx: int) -> str:
        safe_source = source_id.replace("/", "_").replace(" ", "_")
        return f"{safe_source}:{chunk_idx:04d}"
