"""Tests for language-quality rubric helpers (no live LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_assistant.agents._llm import LLMCallResult
from research_assistant.eval import language_quality as lq
from research_assistant.prompts.loader import render


def test_clip_report_truncates() -> None:
    long = "x" * (lq._MAX_REPORT_CHARS + 50)
    out = lq.clip_report_for_judge(long)
    assert len(out) < len(long)
    assert "truncated" in out


def test_mean_over_axes_empty() -> None:
    assert lq.mean_over_axes([]) == dict.fromkeys(lq.AXES, 0.0)


def test_mean_over_axes_macro_mean() -> None:
    rows = [
        {"scores": {"accuracy": 4, "fluency": 4, "terminology": 5, "citation": 3}},
        {"scores": {"accuracy": 2, "fluency": 4, "terminology": 3, "citation": 5}},
    ]
    m = lq.mean_over_axes(rows)
    assert m["accuracy"] == 3.0
    assert m["fluency"] == 4.0
    assert m["terminology"] == 4.0
    assert m["citation"] == 4.0
    assert lq.overall_mean(m) == 3.75


def test_load_reports_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    payload = {
        "queries": [
            {"query": "Q1", "language": "vi", "report": "# Hi"},
            {"query": "Q2", "language": "en", "report": "# Lo"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    items = lq.load_reports_json(path)
    assert len(items) == 2
    assert items[0].language == "vi"


def test_compare_to_baseline(tmp_path: Path) -> None:
    prev = tmp_path / "prev.json"
    prev.write_text(
        json.dumps(
            {
                "mean_by_axis": {
                    "accuracy": 3.0,
                    "fluency": 4.0,
                    "terminology": 3.0,
                    "citation": 4.0,
                },
                "mean_overall": 3.5,
                "created_at": "2026-01-01",
            },
        ),
        encoding="utf-8",
    )
    cur = {
        "mean_by_axis": {
            "accuracy": 4.0,
            "fluency": 4.0,
            "terminology": 3.0,
            "citation": 4.0,
        },
        "mean_overall": 3.75,
    }
    delta = lq.compare_to_baseline(cur, prev)
    assert delta["mean_overall_delta"] == 0.25
    assert delta["mean_by_axis_delta"]["accuracy"] == 1.0


def test_judge_language_quality_uses_mocked_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_invoke(
        model: str,
        prompt: str,
        schema: type[lq.LanguageQualityScores],
        **kwargs: object,
    ) -> tuple[lq.LanguageQualityScores, LLMCallResult]:
        _ = model, prompt, kwargs
        return (
            lq.LanguageQualityScores(
                accuracy=5,
                fluency=4,
                terminology=5,
                citation=4,
                rationale_brief="ok",
            ),
            LLMCallResult(
                text="{}",
                tokens_in=10,
                tokens_out=20,
                cost_usd=0.0001,
                model="stub",
            ),
        )

    monkeypatch.setattr(lq, "invoke_structured_llm", fake_invoke)
    scores, res = lq.judge_language_quality(
        query="What is RAG?",
        output_language="en",
        report_markdown="# Report\nClaim [^1].",
        current_cost_usd=0.0,
        per_query_cap_usd=1.0,
    )
    assert scores.accuracy == 5
    assert res.cost_usd == 0.0001


def test_language_quality_judge_templates_render() -> None:
    sys = render("language_quality_judge_system_v1.jinja", output_language="vi")
    assert "VI" in sys or "Vietnamese" in sys
    usr = render(
        "language_quality_judge_user_v1.jinja",
        query="Test?",
        output_language="vi",
        report_markdown="# Body",
    )
    assert "Test?" in usr
    assert "# Body" in usr
