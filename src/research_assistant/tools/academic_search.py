from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

from pydantic import HttpUrl, TypeAdapter, ValidationError

from research_assistant.graph.state import SearchHit
from research_assistant.observability import observe, update_span
from research_assistant.rag.ingest.arxiv_source import search_arxiv as _default_search_arxiv

logger = logging.getLogger(__name__)


_HTTP_URL_ADAPTER: TypeAdapter[HttpUrl] = TypeAdapter(HttpUrl)


class AcademicSearchError(RuntimeError):
    """Raised when the academic search backend fails or returns malformed data."""


SearchArxivFn = Callable[..., list[dict[str, object]]]


def _coerce_stub(stub: dict[str, object]) -> SearchHit | None:
    try:
        entry_id = stub.get("entry_id") or stub.get("pdf_url")
        if not isinstance(entry_id, str) or not entry_id:
            raise KeyError("entry_id")
        title = stub.get("title")
        if not isinstance(title, str) or not title.strip():
            raise KeyError("title")
        summary = stub.get("summary")
        snippet = (summary if isinstance(summary, str) else "").strip()
        snippet = snippet[:500]

        published = stub.get("published")
        published_date = published if isinstance(published, str) and published else None

        url = _HTTP_URL_ADAPTER.validate_python(entry_id)
        return SearchHit(
            url=url,
            title=title.strip(),
            snippet=snippet,
            score=None,
            published_date=published_date,
            source="academic",
            raw_content=None,
        )
    except (KeyError, ValidationError) as exc:
        logger.warning("Dropping malformed arXiv result: %s", exc)
        return None


@observe(name="academic_search", as_type="tool", capture_input=False, capture_output=False)
def academic_search(
    query: str,
    *,
    max_results: int = 10,
    year_from: int | None = None,
    categories: list[str] | None = None,
    search_fn: SearchArxivFn | None = None,
) -> list[SearchHit]:
    if not query or not query.strip():
        raise AcademicSearchError("Empty query — refusing to call arXiv.")

    bounded = max(1, min(max_results, 20))
    backend = search_fn if search_fn is not None else _default_search_arxiv

    date_from: date | None = None
    if year_from is not None:
        if year_from < 1990 or year_from > 2100:
            raise AcademicSearchError(
                f"year_from={year_from!r} is outside the supported range [1990, 2100]."
            )
        date_from = date(year_from, 1, 1)

    try:
        stubs = backend(
            query,
            max_results=bounded,
            date_from=date_from,
            categories=categories,
        )
    except Exception as exc:
        raise AcademicSearchError(f"arXiv search failed: {exc}") from exc

    if not isinstance(stubs, list):
        raise AcademicSearchError(
            f"Unexpected arXiv payload shape: expected list, got {type(stubs).__name__}.",
        )

    hits: list[SearchHit] = []
    seen_ids: set[str] = set()
    for stub in stubs:
        if not isinstance(stub, dict):
            continue
        stub_dict: dict[str, object] = {str(k): v for k, v in stub.items()}
        arxiv_id = stub_dict.get("arxiv_id")
        if isinstance(arxiv_id, str):
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
        hit = _coerce_stub(stub_dict)
        if hit is not None:
            hits.append(hit)

    logger.info(
        "academic_search query=%r returned %d hits (requested %d, year_from=%s)",
        query,
        len(hits),
        bounded,
        year_from,
    )
    update_span(
        input={
            "query": query,
            "max_results": bounded,
            "year_from": year_from,
            "categories": categories or [],
        },
        output={
            "n_hits": len(hits),
            "arxiv_urls": [str(h.url) for h in hits[:10]],
        },
    )
    return hits


__all__ = ["AcademicSearchError", "academic_search"]
