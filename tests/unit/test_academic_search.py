"""Tests for :mod:`research_assistant.tools.academic_search`.

The arXiv SDK is monkey-patched via ``search_fn`` injection so the real
API is never hit.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from research_assistant.tools.academic_search import (
    AcademicSearchError,
    academic_search,
)


def _stub(
    arxiv_id: str = "2305.14314v1",
    *,
    title: str = "QLoRA: Efficient Finetuning of Quantized LLMs",
    summary: str = (
        "We present QLoRA, an efficient finetuning approach that reduces "
        "memory usage enough to finetune a 65B parameter model on a single "
        "48GB GPU while preserving full 16-bit finetuning task performance."
    ),
    published: str = "2023-05-23",
    primary_category: str = "cs.LG",
    pdf_url: str | None = None,
    entry_id: str | None = None,
) -> dict[str, object]:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "summary": summary,
        "authors": ["Tim Dettmers", "Artidoro Pagnoni"],
        "published": published,
        "primary_category": primary_category,
        "pdf_url": pdf_url or f"http://arxiv.org/pdf/{arxiv_id}",
        "entry_id": entry_id or f"http://arxiv.org/abs/{arxiv_id}",
    }


class _RecordingSearch:
    """Capture call kwargs so tests can assert what the wrapper sent."""

    def __init__(self, payload: list[dict[str, object]]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        query: str,
        *,
        max_results: int = 10,
        date_from: date | None = None,
        categories: list[str] | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
                "date_from": date_from,
                "categories": list(categories) if categories else None,
            },
        )
        return list(self.payload)


def test_academic_search_returns_parsed_hits() -> None:
    backend = _RecordingSearch([_stub()])

    hits = academic_search("QLoRA", search_fn=backend)

    assert len(hits) == 1
    hit = hits[0]
    assert str(hit.url) == "http://arxiv.org/abs/2305.14314v1"
    assert hit.title.startswith("QLoRA")
    assert hit.snippet.startswith("We present QLoRA")
    assert hit.source == "academic"
    assert hit.score is None
    assert hit.published_date == "2023-05-23"
    # Only one upstream call by default.
    assert len(backend.calls) == 1


def test_academic_search_clamps_max_results() -> None:
    backend = _RecordingSearch([])

    academic_search("anything", max_results=99, search_fn=backend)
    assert backend.calls[0]["max_results"] == 20

    academic_search("anything", max_results=0, search_fn=backend)
    assert backend.calls[1]["max_results"] == 1


def test_academic_search_passes_year_filter_as_date() -> None:
    backend = _RecordingSearch([])

    academic_search("recent rag", year_from=2024, categories=["cs.CL"], search_fn=backend)

    call = backend.calls[0]
    assert call["date_from"] == date(2024, 1, 1)
    assert call["categories"] == ["cs.CL"]


def test_academic_search_rejects_empty_query() -> None:
    with pytest.raises(AcademicSearchError):
        academic_search("   ", search_fn=_RecordingSearch([]))


def test_academic_search_rejects_silly_year() -> None:
    with pytest.raises(AcademicSearchError, match="year_from"):
        academic_search("rag", year_from=1800, search_fn=_RecordingSearch([]))


def test_academic_search_wraps_backend_errors() -> None:
    def boom(*_args: Any, **_kwargs: Any) -> list[dict[str, object]]:
        raise RuntimeError("network down")

    with pytest.raises(AcademicSearchError, match="arXiv search failed"):
        academic_search("rag", search_fn=boom)


def test_academic_search_dedupes_by_arxiv_id() -> None:
    backend = _RecordingSearch(
        [
            _stub("2305.14314v1"),
            _stub("2305.14314v1"),  
            _stub("2106.09685v2", title="LoRA: Low-Rank Adaptation"),
        ],
    )

    hits = academic_search("lora", search_fn=backend)

    urls = [str(h.url) for h in hits]
    assert urls == [
        "http://arxiv.org/abs/2305.14314v1",
        "http://arxiv.org/abs/2106.09685v2",
    ]


def test_academic_search_drops_malformed_rows() -> None:
    backend = _RecordingSearch(
        [
            _stub("2305.14314v1"),
            {"title": "missing entry_id"},  # no entry_id / pdf_url
            {"entry_id": "not-a-url", "title": "Bad URL"},
            "not even a dict",  # type: ignore[list-item]
            _stub("2106.09685v2", title="LoRA"),
        ],
    )

    hits = academic_search("rag", search_fn=backend)

    assert [str(h.url) for h in hits] == [
        "http://arxiv.org/abs/2305.14314v1",
        "http://arxiv.org/abs/2106.09685v2",
    ]


def test_academic_search_caps_snippet_to_500_chars() -> None:
    long_summary = "A" * 800
    backend = _RecordingSearch([_stub(summary=long_summary)])

    hits = academic_search("rag", search_fn=backend)

    assert len(hits) == 1
    assert len(hits[0].snippet) == 500


def test_academic_search_rejects_non_list_payload() -> None:
    def bad(*_args: Any, **_kwargs: Any) -> Any:
        return {"unexpected": "dict"}

    with pytest.raises(AcademicSearchError, match="Unexpected arXiv payload"):
        academic_search("rag", search_fn=bad)
