from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from research_assistant.rag.schemas import SourceDoc

logger = logging.getLogger(__name__)


def search_arxiv(
    query: str,
    *,
    max_results: int = 10,
    date_from: date | None = None,
    categories: list[str] | None = None,
) -> list[dict[str, object]]:

    import arxiv

    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        query = f"({query}) AND ({cat_filter})"

    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    stubs: list[dict[str, object]] = []
    for result in _iter_results(client, search):
        if date_from and result.published.date() < date_from:
            continue
        stubs.append(
            {
                "arxiv_id": result.get_short_id(),
                "title": result.title.strip(),
                "summary": result.summary.strip(),
                "authors": [a.name for a in result.authors],
                "published": result.published.date().isoformat(),
                "primary_category": result.primary_category,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
            },
        )
    return stubs


def _iter_results(client: Any, search: Any) -> Iterator[Any]:
    return iter(client.results(search))


def fetch_arxiv_doc(
    arxiv_id: str,
    *,
    cache_dir: Path,
    max_chars: int = 400_000,
) -> SourceDoc:
    import arxiv

    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"{arxiv_id.replace('/', '_')}.pdf"

    client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(_iter_results(client, search), None)
    if result is None:
        raise ValueError(f"arXiv id not found: {arxiv_id}")

    if not pdf_path.exists():
        logger.info("Downloading %s -> %s", arxiv_id, pdf_path)
        result.download_pdf(dirpath=str(cache_dir), filename=pdf_path.name)

    text = _extract_pdf_text(pdf_path, max_chars=max_chars)
    return SourceDoc(
        source_id=result.get_short_id(),
        url=result.entry_id,
        title=result.title.strip(),
        text=text,
        doc_type="arxiv",
        authors=[a.name for a in result.authors],
        published_date=result.published.date(),
        summary=result.summary.strip() if result.summary else None,
    )


def _extract_pdf_text(path: Path, *, max_chars: int) -> str:
    import pymupdf

    chunks: list[str] = []
    total = 0
    doc: Any = pymupdf.open(str(path))  # type: ignore[no-untyped-call]
    try:
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
    return "\n".join(chunks)[:max_chars].strip()
