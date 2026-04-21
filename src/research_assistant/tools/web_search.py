"""Web search tool — Tavily API wrapper.

Contract (LLM-facing):
    name:     web_search
    category: information_retrieval (PLAN.md §4.1)
    purpose:  Retrieve the most relevant public web snippets for a query.
    when:     Whenever the agent needs fresh / open-web facts that are
              unlikely to be already in the corpus.

Return type is always a ``list[SearchHit]``; the caller decides whether to
wrap them as ``Evidence`` for a specific sub-question.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

from pydantic import ValidationError
from tavily import TavilyClient

from research_assistant.config import get_settings
from research_assistant.graph.state import SearchHit

logger = logging.getLogger(__name__)

TimeRange = Literal["day", "week", "month", "year"]
SearchDepth = Literal["basic", "advanced"]


class _SearchClient(Protocol):
    """Minimal protocol implemented by :class:`tavily.TavilyClient`.

    Declared so unit tests can inject a fake without importing Tavily.
    """

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


class WebSearchError(RuntimeError):
    """Raised when the web search backend fails or returns malformed data."""


def _build_client() -> TavilyClient:
    settings = get_settings()
    api_key = settings.tavily_api_key.get_secret_value()
    if not api_key:
        raise WebSearchError(
            "TAVILY_API_KEY is not set. Populate `.env` with a real key before calling web_search.",
        )
    return TavilyClient(api_key=api_key)


def _coerce_hit(raw: dict[str, Any]) -> SearchHit | None:
    """Translate a Tavily result dict into a :class:`SearchHit`.

    Returns ``None`` when the result is missing required fields rather than
    raising, so one bad row does not poison a whole search.
    """
    try:
        return SearchHit(
            url=raw["url"],
            title=raw.get("title") or raw["url"],
            snippet=(raw.get("content") or "").strip(),
            score=raw.get("score"),
            published_date=raw.get("published_date"),
            source="web",
            raw_content=raw.get("raw_content"),
        )
    except (KeyError, ValidationError) as exc:
        logger.warning("Dropping malformed Tavily result: %s", exc)
        return None


def web_search(
    query: str,
    *,
    max_results: int = 10,
    time_range: TimeRange = "year",
    search_depth: SearchDepth = "basic",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    client: _SearchClient | None = None,
) -> list[SearchHit]:
    """Run a web search via Tavily and return a list of :class:`SearchHit`.

    Parameters
    ----------
    query:
        Natural-language search query. Should be self-contained (include
        entities the retriever needs); do not assume conversational context.
    max_results:
        Upper bound on the number of hits returned. Tavily caps this
        internally; we clamp to ``[1, 20]``.
    time_range:
        Recency filter. ``"year"`` is the safe default for rapidly-moving
        domains like AI/ML.
    search_depth:
        ``"basic"`` is cheaper and faster; ``"advanced"`` asks Tavily to
        perform extra extraction, useful when a sub-question needs more
        context than a snippet provides.
    include_domains / exclude_domains:
        Optional domain allow/block lists (see PLAN.md §7.2).
    client:
        Injection point for unit tests. Production callers leave this
        ``None`` so a real :class:`TavilyClient` is built lazily.

    Raises
    ------
    WebSearchError
        If the API key is missing or the upstream call fails.
    """
    if not query or not query.strip():
        raise WebSearchError("Empty query — refusing to call Tavily.")

    bounded = max(1, min(max_results, 20))

    search_client = client if client is not None else _build_client()

    try:
        response = search_client.search(
            query=query,
            max_results=bounded,
            time_range=time_range,
            search_depth=search_depth,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
            include_raw_content=False,
        )
    except Exception as exc:
        raise WebSearchError(f"Tavily search failed: {exc}") from exc

    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise WebSearchError(
            f"Unexpected Tavily payload shape: missing 'results' list "
            f"(got keys {list(response.keys())!r}).",
        )

    hits: list[SearchHit] = []
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        hit = _coerce_hit(raw)
        if hit is not None:
            hits.append(hit)

    logger.info(
        "web_search query=%r returned %d hits (requested %d)",
        query,
        len(hits),
        bounded,
    )
    return hits
