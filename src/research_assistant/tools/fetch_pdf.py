from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from research_assistant.config import get_settings
from research_assistant.observability import observe, update_span
from research_assistant.rag.schemas import Document, SourceDoc

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_EXTRACT_CHARS = 400_000
HTTP_TIMEOUT_S = 30.0

_ARXIV_NEW_ID = re.compile(r"^\s*(\d{4}\.\d{4,5}(?:v\d+)?)\s*$", re.IGNORECASE)
_ARXIV_OLD_ID = re.compile(
    r"^\s*([a-z][a-z0-9-]*(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\s*$",
    re.IGNORECASE,
)
_ARXIV_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", re.IGNORECASE)

ExtractFn = Callable[[Path, int], str]


class FetchPdfError(RuntimeError):
    """Raised when the PDF cannot be fetched, is too large, or is not a PDF."""


def _cache_key_url(canonical_pdf_url: str) -> str:
    normalized = canonical_pdf_url.strip()
    return hashlib.sha1(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()


def _parse_arxiv_id(user_input: str) -> str | None:
    s = user_input.strip()
    m_url = _ARXIV_IN_URL.search(s)
    if m_url:
        frag = m_url.group(1).removesuffix(".pdf").strip()
        return frag or None
    m_new = _ARXIV_NEW_ID.match(s)
    if m_new:
        return m_new.group(1)
    m_old = _ARXIV_OLD_ID.match(s)
    if m_old:
        return m_old.group(1)
    return None


def _parse_direct_pdf_url(user_input: str) -> str | None:
    s = user_input.strip()
    parsed = urlparse(s)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    if not (parsed.path or "").lower().endswith(".pdf"):
        return None
    return s


def _resolve_fetch_target(user_input: str) -> tuple[str, str, str]:
    """Return ``(canonical_pdf_url, source_id, record_url)``."""
    aid = _parse_arxiv_id(user_input)
    if aid:
        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
        abs_url = f"https://arxiv.org/abs/{aid}"
        return pdf_url, aid, abs_url
    direct = _parse_direct_pdf_url(user_input)
    if direct:
        sid = SourceDoc.make_source_id(direct)
        return direct, sid, direct
    raise FetchPdfError(
        "Expected an arXiv id or arXiv URL, or an https URL whose path ends with .pdf.",
    )


def _verify_pdf_magic(header: bytes) -> None:
    if len(header) < 4 or not header[:4] == b"%PDF":
        raise FetchPdfError("Downloaded bytes are not a PDF (missing %PDF header).")


def _stream_download_pdf(
    url: str,
    dest: Path,
    client: httpx.Client,
    *,
    max_bytes: int,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    chunks: list[bytes] = []
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        first = True
        for part in resp.iter_bytes(chunk_size=65_536):
            if not part:
                continue
            if first:
                _verify_pdf_magic(part[:5])
                first = False
            total += len(part)
            if total > max_bytes:
                raise FetchPdfError(
                    f"PDF exceeds max size ({max_bytes // (1024 * 1024)} MiB) — refusing download.",
                )
            chunks.append(part)
    if first:
        raise FetchPdfError("Empty response body when downloading PDF.")
    dest.write_bytes(b"".join(chunks))


def _default_extract_pdf_text(path: Path, max_chars: int) -> tuple[str, str | None]:
    import pymupdf

    chunks: list[str] = []
    total = 0
    title_meta: str | None = None
    doc: Any = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    try:
        meta = doc.metadata
        if isinstance(meta, dict) and meta.get("title"):
            t = str(meta["title"]).strip()
            if t:
                title_meta = t
        for page in doc:
            page_text = page.get_text("text")
            if not page_text:
                continue
            chunks.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break
    finally:
        doc.close()
    body = "\n".join(chunks)[:max_chars].strip()
    return body, title_meta


@observe(name="fetch_pdf", as_type="tool", capture_input=False, capture_output=False)
def fetch_pdf(
    arxiv_id_or_pdf_url: str,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
    extract_pdf_text_fn: ExtractFn | None = None,
    max_pdf_bytes: int = MAX_PDF_BYTES,
    max_chars: int = MAX_EXTRACT_CHARS,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> Document:

    if not arxiv_id_or_pdf_url or not arxiv_id_or_pdf_url.strip():
        raise FetchPdfError("Empty input — provide an arXiv id or a .pdf URL.")

    pdf_url, source_id, record_url = _resolve_fetch_target(arxiv_id_or_pdf_url)
    key = _cache_key_url(pdf_url)
    base = cache_dir if cache_dir is not None else get_settings().raw_docs_dir / "pdf_cache"
    cached = base / f"{key}.pdf"

    own_client = client is None
    hc = client or httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        follow_redirects=True,
    )
    try:
        if not cached.exists():
            logger.info("fetch_pdf: downloading %s -> %s", pdf_url, cached)
            try:
                _stream_download_pdf(pdf_url, cached, hc, max_bytes=max_pdf_bytes)
            except httpx.HTTPStatusError as exc:
                raise FetchPdfError(f"HTTP error fetching PDF: {exc.response.status_code}") from exc
            except httpx.RequestError as exc:
                raise FetchPdfError(f"Network error fetching PDF: {exc}") from exc
        else:
            logger.debug("fetch_pdf: cache hit %s", cached)

        if extract_pdf_text_fn is not None:
            text = extract_pdf_text_fn(cached, max_chars)
            title_guess: str | None = None
        else:
            text, title_guess = _default_extract_pdf_text(cached, max_chars)

        if not text:
            raise FetchPdfError("No text could be extracted from the PDF.")

        title = (title_guess or "").strip() or f"PDF ({source_id})"
        doc = SourceDoc(
            source_id=source_id,
            url=record_url,
            title=title,
            text=text,
            doc_type="pdf",
            authors=[],
            published_date=None,
            summary=None,
        )
        update_span(metadata={"source_id": source_id, "cached_path": str(cached)})
        return doc
    finally:
        if own_client:
            hc.close()
