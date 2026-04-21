"""Tests for :mod:`research_assistant.tools.web_search`.

All calls use an injected fake client so the real Tavily API is never hit.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant.tools.web_search import WebSearchError, web_search


class _FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        return self.payload


def _good_payload() -> dict[str, Any]:
    return {
        "results": [
            {
                "url": "https://example.com/a",
                "title": "Alpha",
                "content": "alpha snippet",
                "score": 0.91,
                "published_date": "2025-09-01",
            },
            {
                "url": "https://example.com/b",
                "title": "Beta",
                "content": "beta snippet",
                "score": 0.77,
            },
            # malformed row — should be dropped without raising:
            {"title": "no url"},
        ],
    }


def test_web_search_returns_parsed_hits() -> None:
    client = _FakeClient(_good_payload())
    hits = web_search("what is rag", client=client)
    assert len(hits) == 2
    assert str(hits[0].url) == "https://example.com/a"
    assert hits[0].score == pytest.approx(0.91)
    assert hits[1].title == "Beta"
    assert hits[0].source == "web"


def test_web_search_clamps_max_results() -> None:
    client = _FakeClient({"results": []})
    web_search("q", max_results=99, client=client)
    assert client.calls[0]["max_results"] == 20
    web_search("q", max_results=0, client=client)
    assert client.calls[1]["max_results"] == 1


def test_web_search_rejects_empty_query() -> None:
    client = _FakeClient({"results": []})
    with pytest.raises(WebSearchError):
        web_search("   ", client=client)


def test_web_search_raises_on_malformed_payload() -> None:
    client = _FakeClient({"oops": "no results key"})
    with pytest.raises(WebSearchError, match="Unexpected Tavily payload"):
        web_search("q", client=client)


def test_web_search_wraps_backend_errors() -> None:
    class BoomClient:
        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("network down")

    with pytest.raises(WebSearchError, match="Tavily search failed"):
        web_search("q", client=BoomClient())
