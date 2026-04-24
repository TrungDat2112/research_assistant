"""Tests for smoke Markdown citation coverage parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant.eval.smoke_citation import (
    extract_report_body,
    parse_smoke_markdown,
    run_smoke_citation_eval,
)

_FIXTURE = """# Research Assistant — Smoke

# Query 1 (VI): Alpha question here

## Tóm lược kế hoạch
Some plan text.

---

## 1. First sub-question?

Claim one [^1]. Another sentence [^1].

## 2. Second?

Chưa đủ dữ liệu để kết luận phần này.

## Tài liệu tham khảo
[^1]: http://example.com
---
footer

---

# Query 2 (EN): Beta english

## Plan overview
x

---

## 1. Sub?

*(No synthesized answer available for this sub-question.)*

## References
[^1]: http://b.com
"""


def test_extract_report_body_starts_at_numbered_section() -> None:
    block = _FIXTURE.split("# Query 2")[0]
    body = extract_report_body(block)
    assert "## Tóm lược" not in body
    assert "Claim one" in body
    assert "http://example.com" not in body


def test_parse_smoke_markdown_two_queries() -> None:
    rows = parse_smoke_markdown(_FIXTURE)
    assert len(rows) == 2
    assert rows[0].query_index == 1
    assert rows[0].language == "vi"
    assert rows[1].query_index == 2
    assert rows[1].language == "en"
    assert rows[0].paragraph_citation_coverage == 1.0
    assert rows[1].paragraph_citation_coverage == 1.0


def test_parse_smoke_markdown_raises_without_headers() -> None:
    with pytest.raises(ValueError, match="no '# Query"):
        parse_smoke_markdown("# Just a title\n\nno query headers")


def test_run_smoke_citation_eval_writes_shape(tmp_path: Path) -> None:
    md = tmp_path / "smoke.md"
    md.write_text(_FIXTURE, encoding="utf-8")
    out = tmp_path / "citation_coverage.json"
    payload = run_smoke_citation_eval(md, target_mean_coverage=0.8)
    assert payload["n_queries"] == 2
    assert payload["meets_target"] is True
    assert "generated_at" in payload
    assert not out.exists()
