"""Tests for :mod:`research_assistant.rag.ingest`.

Real network is never touched — every fetch is monkeypatched.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from research_assistant.rag.ingest import loader as loader_mod
from research_assistant.rag.ingest.loader import ArxivQuerySpec, SeedConfig, load_seed_corpus
from research_assistant.rag.schemas import SourceDoc


def _fake_doc(source_id: str, doc_type: str = "arxiv") -> SourceDoc:
    return SourceDoc(
        source_id=source_id,
        url=f"https://example.com/{source_id}",
        title=f"Title {source_id}",
        text=f"Body of {source_id}. " * 10,
        doc_type=doc_type,  # type: ignore[arg-type]
        authors=["Alice"],
        published_date=date(2025, 1, 1),
        summary="short abstract",
    )


def test_seed_config_from_yaml_parses_everything(tmp_path: Path) -> None:
    yaml_path = tmp_path / "seed.yaml"
    yaml_path.write_text(
        """
arxiv:
  ids:
    - "2005.11401"
    - "2310.11511"
  queries:
    - query: "retrieval augmented generation"
      max_results: 3
      date_from: "2024-01-01"
      categories: ["cs.CL"]
html:
  - url: "https://anthropic.com/blog1"
    title: "Blog 1"
  - "https://openai.com/blog2"
""".strip(),
        encoding="utf-8",
    )
    cfg = SeedConfig.from_yaml(yaml_path)
    assert cfg.arxiv_ids == ["2005.11401", "2310.11511"]
    assert len(cfg.arxiv_queries) == 1
    q = cfg.arxiv_queries[0]
    assert q.query == "retrieval augmented generation"
    assert q.max_results == 3
    assert q.date_from == date(2024, 1, 1)
    assert q.categories == ["cs.CL"]
    assert cfg.html_urls == [
        ("https://anthropic.com/blog1", "Blog 1"),
        ("https://openai.com/blog2", None),
    ]


def test_seed_config_from_yaml_skips_malformed(tmp_path: Path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(
        """
arxiv:
  queries:
    - not_a_mapping
    - query: "good one"
html:
  - 12345
  - url: "https://ok.example/x"
""".strip(),
        encoding="utf-8",
    )
    cfg = SeedConfig.from_yaml(yaml_path)
    assert len(cfg.arxiv_queries) == 1
    assert cfg.arxiv_queries[0].query == "good one"
    assert cfg.html_urls == [("https://ok.example/x", None)]


def test_load_seed_corpus_dedups_across_queries_and_explicit_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SeedConfig(
        arxiv_ids=["2005.11401", "9999.00000"],
        arxiv_queries=[ArxivQuerySpec(query="rag", max_results=2)],
        html_urls=[("https://anthropic.com/blog1", "b1")],
    )
    # Query returns overlapping id + a new one.
    monkeypatch.setattr(
        loader_mod,
        "search_arxiv",
        lambda *_a, **_kw: [
            {"arxiv_id": "2005.11401", "title": "RAG"},  # dup with explicit id
            {"arxiv_id": "2310.11511", "title": "Self-RAG"},
        ],
    )
    fetched_ids: list[str] = []

    def _fetch_arxiv(aid: str, *, cache_dir: Path) -> SourceDoc:
        fetched_ids.append(aid)
        return _fake_doc(aid)

    monkeypatch.setattr(loader_mod, "fetch_arxiv_doc", _fetch_arxiv)
    monkeypatch.setattr(
        loader_mod,
        "fetch_html_doc",
        lambda url, *, title_hint=None: _fake_doc(url, doc_type="blog"),
    )

    result = load_seed_corpus(cfg, arxiv_cache_dir=tmp_path / "arxiv")

    # Each arxiv id fetched exactly once despite duplication.
    assert sorted(fetched_ids) == ["2005.11401", "2310.11511", "9999.00000"]
    assert result.success_count == 4  # 3 arxiv + 1 html
    assert result.failure_count == 0


def test_load_seed_corpus_records_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SeedConfig(
        arxiv_ids=["bad_id"],
        html_urls=[("https://unreachable.example/x", None)],
    )

    def _fail_arxiv(aid: str, *, cache_dir: Path) -> SourceDoc:
        raise RuntimeError(f"no paper {aid}")

    def _fail_html(url: str, *, title_hint: Any = None) -> SourceDoc:
        raise ValueError(f"fetch failed {url}")

    monkeypatch.setattr(loader_mod, "fetch_arxiv_doc", _fail_arxiv)
    monkeypatch.setattr(loader_mod, "fetch_html_doc", _fail_html)

    result = load_seed_corpus(cfg, arxiv_cache_dir=tmp_path / "arxiv")
    assert result.success_count == 0
    assert result.failure_count == 2
    assert any("arxiv:bad_id" in f[0] for f in result.failures)
    assert any("html:https://unreachable" in f[0] for f in result.failures)


def test_load_seed_corpus_survives_search_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SeedConfig(
        arxiv_ids=["2005.11401"],
        arxiv_queries=[ArxivQuerySpec(query="x")],
        html_urls=[],
    )

    def _boom(*_a: Any, **_kw: Any) -> list[dict[str, object]]:
        raise RuntimeError("arxiv down")

    monkeypatch.setattr(loader_mod, "search_arxiv", _boom)
    monkeypatch.setattr(
        loader_mod,
        "fetch_arxiv_doc",
        lambda aid, *, cache_dir: _fake_doc(aid),
    )

    result = load_seed_corpus(cfg, arxiv_cache_dir=tmp_path / "arxiv")
    assert result.success_count == 1  # explicit id still succeeds
    assert any("query:x" in f[0] for f in result.failures)
