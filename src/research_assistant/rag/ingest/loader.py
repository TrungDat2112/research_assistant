from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from research_assistant.rag.ingest.arxiv_source import fetch_arxiv_doc, search_arxiv
from research_assistant.rag.ingest.html_source import fetch_html_doc
from research_assistant.rag.schemas import SourceDoc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArxivQuerySpec:
    query: str
    max_results: int = 5
    date_from: date | None = None
    categories: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeedConfig:
    arxiv_ids: list[str] = field(default_factory=list)
    arxiv_queries: list[ArxivQuerySpec] = field(default_factory=list)
    html_urls: list[tuple[str, str | None]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> SeedConfig:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        arxiv_cfg = raw.get("arxiv") or {}
        html_cfg = raw.get("html") or []

        ids: list[str] = [str(x) for x in (arxiv_cfg.get("ids") or [])]
        queries: list[ArxivQuerySpec] = []
        for q in arxiv_cfg.get("queries") or []:
            if not isinstance(q, dict) or "query" not in q:
                logger.warning("Skipping malformed arxiv query: %r", q)
                continue
            queries.append(
                ArxivQuerySpec(
                    query=str(q["query"]),
                    max_results=int(q.get("max_results", 5)),
                    date_from=date.fromisoformat(q["date_from"]) if q.get("date_from") else None,
                    categories=list(q.get("categories") or []),
                ),
            )

        html_urls: list[tuple[str, str | None]] = []
        for entry in html_cfg:
            if isinstance(entry, str):
                html_urls.append((entry, None))
            elif isinstance(entry, dict) and "url" in entry:
                html_urls.append((str(entry["url"]), entry.get("title")))
            else:
                logger.warning("Skipping malformed html entry: %r", entry)

        return cls(arxiv_ids=ids, arxiv_queries=queries, html_urls=html_urls)


@dataclass
class IngestResult:
    docs: list[SourceDoc]
    failures: list[tuple[str, str]]  # (identifier, reason)

    @property
    def success_count(self) -> int:
        return len(self.docs)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def load_seed_corpus(
    config: SeedConfig,
    *,
    arxiv_cache_dir: Path,
) -> IngestResult:

    docs: list[SourceDoc] = []
    failures: list[tuple[str, str]] = []
    seen: set[str] = set()

    wanted_ids: list[str] = list(dict.fromkeys(config.arxiv_ids))
    for spec in config.arxiv_queries:
        try:
            stubs = search_arxiv(
                spec.query,
                max_results=spec.max_results,
                date_from=spec.date_from,
                categories=spec.categories,
            )
        except Exception as exc:
            failures.append((f"query:{spec.query}", f"search_arxiv failed: {exc}"))
            continue
        for s in stubs:
            wanted_ids.append(str(s["arxiv_id"]))

    for aid in dict.fromkeys(wanted_ids):
        if aid in seen:
            continue
        seen.add(aid)
        try:
            docs.append(fetch_arxiv_doc(aid, cache_dir=arxiv_cache_dir))
        except Exception as exc:
            failures.append((f"arxiv:{aid}", str(exc)))

    for url, hint in config.html_urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            docs.append(fetch_html_doc(url, title_hint=hint))
        except Exception as exc:
            failures.append((f"html:{url}", str(exc)))

    return IngestResult(docs=docs, failures=failures)
