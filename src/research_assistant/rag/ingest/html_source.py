from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from research_assistant.rag.schemas import SourceDoc

logger = logging.getLogger(__name__)

_MAX_CHARS = 300_000


def fetch_html_doc(url: str, *, title_hint: str | None = None) -> SourceDoc:

    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Fetch failed for {url}")

    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
        output_format="txt",
    )
    if not extracted or not extracted.strip():
        raise ValueError(f"Empty extraction for {url}")

    meta = _extract_metadata(downloaded, url)
    title = (meta.get("title") or title_hint or url).strip()
    authors = _parse_authors(meta.get("author"))
    published = _parse_date(meta.get("date"))
    summary = (meta.get("description") or "").strip() or None

    body = extracted.strip()[:_MAX_CHARS]

    return SourceDoc(
        source_id=SourceDoc.make_source_id(url),
        url=url,
        title=title,
        text=body,
        doc_type="blog",
        authors=authors,
        published_date=published,
        summary=summary,
    )


def _extract_metadata(html: str, url: str) -> dict[str, Any]:
    import trafilatura

    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
    except Exception as exc:  # metadata extraction is best-effort
        logger.debug("Metadata extraction failed for %s: %s", url, exc)
        return {}
    if meta is None:
        return {}
    return meta.as_dict() if hasattr(meta, "as_dict") else dict(meta.__dict__)


def _parse_authors(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    parts = [p.strip() for p in str(raw).split(";") if p.strip()]
    return parts


def _parse_date(raw: Any) -> date | None:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
