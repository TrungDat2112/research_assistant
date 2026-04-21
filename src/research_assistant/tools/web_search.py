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
import re
from typing import Any, Literal, Protocol

from pydantic import ValidationError
from tavily import TavilyClient

from research_assistant.config import get_settings
from research_assistant.graph.state import SearchHit

logger = logging.getLogger(__name__)

TimeRange = Literal["day", "week", "month", "year"]
SearchDepth = Literal["basic", "advanced"]

# Tokens that hurt recall when a query is already "year-bound" (e.g. Tavily
# mis-filters VN-language year phrases). Stripped only for the widened-recall
# retry — the original query is always tried first.
_YEAR_PATTERN = re.compile(r"\b(năm\s+)?20\d{2}\b", flags=re.IGNORECASE)
_IN_YEAR_PATTERN = re.compile(r"\bin\s+20\d{2}\b", flags=re.IGNORECASE)


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


def _simplify_query(query: str) -> str:
    """Strip overly-specific year tokens to widen recall on a retry.

    Kept small and deterministic — this is a safety net, not a rewriter.
    If nothing changes, the caller falls through to the original query.
    """
    simplified = _IN_YEAR_PATTERN.sub("", query)
    simplified = _YEAR_PATTERN.sub("", simplified)
    return " ".join(simplified.split())


def web_search_with_fallback(
    query: str,
    *,
    max_results: int = 10,
    time_range: TimeRange = "year",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    client: _SearchClient | None = None,
) -> list[SearchHit]:
    """Run :func:`web_search` with up to two progressive retries on empty hits.

    Retry ladder:

    1. ``search_depth="basic"`` with the original query — cheap, covers the
       vast majority of cases (this is what production used before the fix).
    2. If stage 1 returns zero hits: ``search_depth="advanced"`` on the
       same query. Tavily then performs extra extraction and usually
       recovers niche sub-questions (e.g. vector-DB benchmark pages).
    3. If stage 2 still returns zero hits AND the query contains a specific
       year token: strip the year and retry at ``advanced`` depth. Keeps
       answers reasonably fresh via ``time_range`` but widens recall for
       questions whose wording is too tight.

    Any :class:`WebSearchError` from the underlying call is propagated —
    the graph's retriever node handles them as "0 hits, ok to continue".
    Injected ``client`` is reused across all stages so unit tests can
    assert the retry ladder precisely.
    """
    hits = web_search(
        query,
        max_results=max_results,
        time_range=time_range,
        search_depth="basic",
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        client=client,
    )
    if hits:
        return hits

    logger.info(
        "web_search_with_fallback: 0 hits on basic for %r — retrying with advanced depth",
        query,
    )
    hits = web_search(
        query,
        max_results=max_results,
        time_range=time_range,
        search_depth="advanced",
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        client=client,
    )
    if hits:
        return hits

    simplified = _simplify_query(query)
    if simplified and simplified != query:
        logger.info(
            "web_search_with_fallback: 0 hits on advanced — retrying simplified %r",
            simplified,
        )
        hits = web_search(
            simplified,
            max_results=max_results,
            time_range=time_range,
            search_depth="advanced",
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            client=client,
        )

    return hits
