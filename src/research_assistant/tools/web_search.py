from __future__ import annotations

import logging
import re
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import ValidationError
from tavily import TavilyClient

from research_assistant.config import get_settings
from research_assistant.graph.state import SearchHit, WebTrustTier
from research_assistant.observability import observe, update_span

logger = logging.getLogger(__name__)

TimeRange = Literal["day", "week", "month", "year"]
SearchDepth = Literal["basic", "advanced"]

_YEAR_PATTERN = re.compile(r"\b(năm\s+)?20\d{2}\b", flags=re.IGNORECASE)
_IN_YEAR_PATTERN = re.compile(r"\bin\s+20\d{2}\b", flags=re.IGNORECASE)

_HIGH_TRUST_DOMAIN_SUFFIXES: tuple[str, ...] = (
    "anthropic.com",
    "openai.com",
    "deepmind.google",
    "ai.googleblog.com",
    "research.google",
    "blog.google",
    "nvidia.com",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",
    "arxiv.org",
    "semanticscholar.org",
    "aclanthology.org",
    "neurips.cc",
    "icml.cc",
    "iclr.cc",
    "microsoft.com",
    "ibm.com",
    "apple.com",
    "python.org",
    "docs.python.org",
    "readthedocs.io",
    "langchain.com",
    "langgraph.dev",
    "github.io",
    "wikipedia.org",
)
_LOW_TRUST_DOMAIN_SUFFIXES: tuple[str, ...] = (
    "indeed.com",
    "glassdoor.com",
    "linkedin.com",
    "ziprecruiter.com",
    "monster.com",
    "simplyhired.com",
    "udemy.com",
    "coursera.org",
    "skillshare.com",
    "pluralsight.com",
    "brainly.com",
    "chegg.com",
    "coursehero.com",
    "quizlet.com",
)


class _SearchClient(Protocol):
    def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...


class WebSearchError(RuntimeError):
    """Raised when the web search backend fails or returns malformed data."""


def _host_and_path(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").lower()
    return host, path


def _host_matches_suffix(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith(f".{suffix}")


def web_trust_tier_for_url(url: str) -> WebTrustTier:
 
    host, _ = _host_and_path(url)
    if not host:
        return "medium"
    if host.endswith((".edu", ".gov")):
        return "high"
    for suffix in _LOW_TRUST_DOMAIN_SUFFIXES:
        if _host_matches_suffix(host, suffix):
            return "low"
    for suffix in _HIGH_TRUST_DOMAIN_SUFFIXES:
        if _host_matches_suffix(host, suffix):
            return "high"
    return "medium"


def _build_client() -> TavilyClient:
    settings = get_settings()
    api_key = settings.tavily_api_key.get_secret_value()
    if not api_key:
        raise WebSearchError(
            "TAVILY_API_KEY is not set. Populate `.env` with a real key before calling web_search.",
        )
    return TavilyClient(api_key=api_key)


def _coerce_hit(raw: dict[str, Any]) -> SearchHit | None:

    try:
        url_str = str(raw["url"])
        return SearchHit(
            url=raw["url"],
            title=raw.get("title") or raw["url"],
            snippet=(raw.get("content") or "").strip(),
            score=raw.get("score"),
            published_date=raw.get("published_date"),
            source="web",
            raw_content=raw.get("raw_content"),
            web_trust_tier=web_trust_tier_for_url(url_str),
        )
    except (KeyError, ValidationError) as exc:
        logger.warning("Dropping malformed Tavily result: %s", exc)
        return None


@observe(name="web_search", as_type="tool", capture_input=False, capture_output=False)
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
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for h in hits:
        t = h.web_trust_tier
        if t is not None:
            tier_counts[t] += 1
    update_span(
        input={
            "query": query,
            "max_results": bounded,
            "search_depth": search_depth,
            "time_range": time_range,
        },
        output={
            "n_hits": len(hits),
            "web_trust_tier_counts": tier_counts,
            "urls": [str(h.url) for h in hits[:10]],
        },
    )
    return hits


def _simplify_query(query: str) -> str:

    simplified = _IN_YEAR_PATTERN.sub("", query)
    simplified = _YEAR_PATTERN.sub("", simplified)
    return " ".join(simplified.split())


@observe(
    name="web_search_with_fallback",
    as_type="tool",
    capture_input=False,
    capture_output=False,
)
def web_search_with_fallback(
    query: str,
    *,
    max_results: int = 10,
    time_range: TimeRange = "year",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    client: _SearchClient | None = None,
) -> list[SearchHit]:

    stage_used = "basic"
    hits = web_search(
        query,
        max_results=max_results,
        time_range=time_range,
        search_depth="basic",
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        client=client,
    )
    if not hits:
        logger.info(
            "web_search_with_fallback: 0 hits on basic for %r — retrying with advanced depth",
            query,
        )
        stage_used = "advanced"
        hits = web_search(
            query,
            max_results=max_results,
            time_range=time_range,
            search_depth="advanced",
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            client=client,
        )

    if not hits:
        simplified = _simplify_query(query)
        if simplified and simplified != query:
            logger.info(
                "web_search_with_fallback: 0 hits on advanced — retrying simplified %r",
                simplified,
            )
            stage_used = "advanced_simplified"
            hits = web_search(
                simplified,
                max_results=max_results,
                time_range=time_range,
                search_depth="advanced",
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                client=client,
            )

    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for h in hits:
        t = h.web_trust_tier
        if t is not None:
            tier_counts[t] += 1
    update_span(
        input={"query": query, "max_results": max_results, "time_range": time_range},
        output={
            "n_hits": len(hits),
            "stage_used": stage_used,
            "web_trust_tier_counts": tier_counts,
        },
    )
    return hits
