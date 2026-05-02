from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator

DocType = Literal["arxiv", "blog", "html", "pdf", "other"]


class SourceDoc(BaseModel):


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



Document: TypeAlias = SourceDoc


class ChunkMetadata(BaseModel):
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

    chunk_id: str = Field(..., pattern=r"^[A-Za-z0-9_\-:.]{3,128}$")
    text: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    metadata: ChunkMetadata

    @classmethod
    def make_chunk_id(cls, source_id: str, chunk_idx: int) -> str:
        safe_source = source_id.replace("/", "_").replace(" ", "_")
        return f"{safe_source}:{chunk_idx:04d}"
