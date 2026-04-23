"""Tests for Jinja prompt loader + templates."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from jinja2 import UndefinedError

from research_assistant.graph.state import Draft, Evidence, SearchHit, SubQuestion
from research_assistant.prompts.loader import available_templates, render


def _sample_evidence(n: int, sq_id: str = "sq_1") -> list[Evidence]:
    return [
        Evidence(
            ref_label=f"ev_{sq_id}_{i}",
            sub_question_id=sq_id,
            hit=SearchHit(
                url=f"https://example.com/{i}",  # type: ignore[arg-type]
                title=f"Doc {i}",
                snippet=f"Snippet {i}",
                published_date="2025-10-01",
            ),
        )
        for i in range(1, n + 1)
    ]


def test_available_templates_lists_expected_files() -> None:
    templates = available_templates()
    for expected in (
        "planner_v1.jinja",
        "synthesizer_v1.jinja",
        "critic_v1.jinja",
        "reporter_v1.jinja",
    ):
        assert expected in templates, f"missing {expected} in {templates}"


def test_planner_template_renders_both_languages() -> None:
    vi = render("planner_v1.jinja", query="So sánh A vs B", output_language="vi")
    en = render("planner_v1.jinja", query="Compare A vs B", output_language="en")
    assert "Vietnamese" in vi
    assert "English" in en
    assert "sq_1" in vi and "sq_1" in en


def test_critic_template_renders() -> None:
    plan = SubQuestion(id="sq_1", question="What is LoRA?", rationale="Understand adaptation")
    evs = _sample_evidence(2, "sq_1")
    draft = Draft(sub_question_id="sq_1", content="LoRA is low-rank [^1].", model="m")
    out = render(
        "critic_v1.jinja",
        user_query="Explain LoRA",
        sub_question=plan,
        evidence=evs,
        draft=draft,
        paragraph_citation_coverage=1.0,
        output_language="en",
    )
    assert "What is LoRA?" in out
    assert "LoRA is low-rank" in out
    assert "Doc 1" in out


def test_synthesizer_template_includes_evidence() -> None:
    evs = _sample_evidence(3, "sq_2")
    out = render(
        "synthesizer_v1.jinja",
        sub_question="What is RAG?",
        evidence=evs,
        output_language="vi",
    )
    assert "Doc 1" in out and "Doc 2" in out and "Doc 3" in out
    assert "Snippet 3" in out
    assert "[^N]" in out  # citation rule mention


def test_reporter_template_renders_minimal_report() -> None:
    plan = [
        SubQuestion(id="sq_1", question="Câu hỏi 1"),
        SubQuestion(id="sq_2", question="Câu hỏi 2"),
    ]
    drafts = {
        "sq_1": Draft(sub_question_id="sq_1", content="A [^1]", model="m"),
        "sq_2": Draft(sub_question_id="sq_2", content="B [^1]", model="m"),
    }
    evidence = {"sq_1": _sample_evidence(1, "sq_1"), "sq_2": _sample_evidence(1, "sq_2")}
    out = render(
        "reporter_v1.jinja",
        query="Câu hỏi chính",
        output_language="vi",
        plan=plan,
        drafts=drafts,
        evidence=evidence,
        generated_at_iso=datetime(2026, 4, 21, tzinfo=UTC).isoformat(),
        total_cost_usd=0.12,
    )
    assert "# Câu hỏi chính" in out
    assert "## 1. Câu hỏi 1" in out
    assert "Tài liệu tham khảo" in out
    assert "[^1]" in out and "[^2]" in out


def test_strict_undefined_raises_on_missing_variable() -> None:
    with pytest.raises(UndefinedError):
        render("planner_v1.jinja", query="x")  # missing output_language
