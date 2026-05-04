from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .schemas import Chunk, ChunkMetadata, SourceDoc

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 \-:,]{3,80}$")


@lru_cache(maxsize=4)
def _get_tokenizer(model_id: str) -> PreTrainedTokenizerBase:

    from transformers import AutoTokenizer

    logger.debug("Loading tokenizer %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    tok.model_max_length = 1_000_000
    return tok


def _iter_sections(text: str) -> list[tuple[int, str]]:

    sections: list[tuple[int, str]] = [(0, "")]
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        md = _HEADING_RE.match(stripped)
        if md:
            sections.append((offset, md.group(2)))
        elif _ALLCAPS_RE.match(stripped) and len(stripped) <= 80:
            sections.append((offset, stripped.title()))
        offset += len(line)
    return sections


def _section_at(sections: list[tuple[int, str]], offset: int) -> str:
    current = ""
    for pos, name in sections:
        if pos > offset:
            break
        current = name
    return current


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _summarise_for_prepend(doc: SourceDoc, max_chars: int = 400) -> str:

    if doc.summary:
        base = doc.summary.strip()
    else:
        sentences = _SENTENCE_SPLIT.split(doc.text.strip(), maxsplit=2)
        base = " ".join(sentences[:2]).strip()
    base = base or doc.title
    if len(base) > max_chars:
        base = base[: max_chars - 3].rstrip() + "..."
    return f"[{doc.title}] {base}".strip()


@dataclass(frozen=True)
class ChunkingConfig:
    model_id: str
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    prepend_max_chars: int = 400


def chunk_document(doc: SourceDoc, config: ChunkingConfig) -> list[Chunk]:

    if not doc.text.strip():
        logger.warning("Empty text for source_id=%s — skipping", doc.source_id)
        return []
    if config.chunk_overlap_tokens >= config.chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    tokenizer = _get_tokenizer(config.model_id)
    encoding = tokenizer(
        doc.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    token_ids: list[int] = encoding["input_ids"]
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    if not token_ids:
        return []

    sections = _iter_sections(doc.text)
    prepend = _summarise_for_prepend(doc, max_chars=config.prepend_max_chars)

    chunks: list[Chunk] = []
    step = config.chunk_size_tokens - config.chunk_overlap_tokens
    start = 0
    raw_slices: list[tuple[int, int]] = []
    while start < len(token_ids):
        end = min(start + config.chunk_size_tokens, len(token_ids))
        raw_slices.append((start, end))
        if end == len(token_ids):
            break
        start += step

    total = len(raw_slices)
    for idx, (s, e) in enumerate(raw_slices):
        char_start = offsets[s][0]
        char_end = offsets[e - 1][1]
        body = doc.text[char_start:char_end].strip()
        if not body:
            continue
        section = _section_at(sections, char_start)
        meta = ChunkMetadata(
            source_id=doc.source_id,
            source_url=doc.url,
            title=doc.title,
            doc_type=doc.doc_type,
            chunk_idx=idx,
            chunk_total=total,
            section=section,
            authors="; ".join(doc.authors),
            published_date=doc.published_date.isoformat() if doc.published_date else "",
        )
        embed_text = f"{prepend}\n\n{body}" if prepend else body
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_chunk_id(doc.source_id, idx),
                text=embed_text,
                body=body,
                metadata=meta,
            ),
        )
    return chunks


def chunk_documents(docs: list[SourceDoc], config: ChunkingConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, config))
    return out
