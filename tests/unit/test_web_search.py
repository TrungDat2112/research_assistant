"""Tests for :mod:`research_assistant.tools.web_search`.

All calls use an injected fake client so the real Tavily API is never hit.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant.tools.web_search import (
    WebSearchError,
    _simplify_query,
    web_search,
    web_search_with_fallback,
)


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


# ---------------------------------------------------------------------------
# Fallback ladder
# ---------------------------------------------------------------------------


class _StagedClient:
    """Fake client that returns a scripted payload per call.

    Each call pops the next payload off ``payloads``; once exhausted it
    returns an empty results list. Records ``calls`` so tests can assert
    the exact retry ladder.
    """

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        if self.payloads:
            return self.payloads.pop(0)
        return {"results": []}


def test_simplify_query_strips_years() -> None:
    assert _simplify_query("LoRA vs QLoRA năm 2026") == "LoRA vs QLoRA"
    assert _simplify_query("vector DB benchmark in 2025") == "vector DB benchmark"
    assert _simplify_query("no year here") == "no year here"


def test_fallback_returns_stage_1_when_hits_found() -> None:
    client = _StagedClient([_good_payload()])
    hits = web_search_with_fallback("q", client=client)
    assert len(hits) == 2
    # Only one upstream call, basic depth.
    assert len(client.calls) == 1
    assert client.calls[0]["search_depth"] == "basic"


def test_fallback_retries_advanced_when_basic_empty() -> None:
    empty: dict[str, Any] = {"results": []}
    client = _StagedClient([empty, _good_payload()])
    hits = web_search_with_fallback("some niche query", client=client)
    assert len(hits) == 2
    depths = [c["search_depth"] for c in client.calls]
    assert depths == ["basic", "advanced"]


def test_fallback_strips_year_as_last_resort() -> None:
    empty: dict[str, Any] = {"results": []}
    client = _StagedClient([empty, empty, _good_payload()])
    hits = web_search_with_fallback("LoRA benchmark năm 2026", client=client)
    assert len(hits) == 2
    assert len(client.calls) == 3
    # Stage 3 must use the simplified query AND advanced depth.
    assert client.calls[2]["query"] == "LoRA benchmark"
    assert client.calls[2]["search_depth"] == "advanced"


def test_fallback_gives_up_with_empty_list_when_query_has_no_year() -> None:
    empty: dict[str, Any] = {"results": []}
    client = _StagedClient([empty, empty])
    hits = web_search_with_fallback("generic topic", client=client)
    # Only two stages attempted (no simplification possible), returns [].
    assert hits == []
    assert len(client.calls) == 2
