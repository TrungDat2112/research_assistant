"""Tests for factuality eval (no live LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_assistant.agents._llm import LLMCallResult
from research_assistant.eval import factuality as fac
from research_assistant.prompts.loader import render

_REPO = Path(__file__).resolve().parents[2]
_EVAL_PATH = _REPO / "data" / "eval" / "factuality_eval_20.json"


def test_clip_report_truncates() -> None:
    long = "z" * (fac._MAX_REPORT_CHARS + 80)
    out = fac.clip_report_for_judge(long)
    assert len(out) < len(long)
    assert "truncated" in out


def test_load_factuality_eval_validates_counts() -> None:
    items = fac.load_factuality_eval(_EVAL_PATH)
    fac.validate_eval_counts(items)
    assert items[0].language == "en"
    assert items[-1].language == "vi"
    assert len(items[14].gold_claims) >= fac._MIN_CLAIMS


def test_normalize_judgments_fills_missing() -> None:
    claims = ["a", "b", "c"]
    parsed = fac.FactualityJudgeOutput(
        judgments=[
            fac.FactualityClaimJudgment(claim_index=0, verdict="supported", rationale_brief="ok"),
        ],
    )
    rows = fac.normalize_judgments(claims, parsed)
    assert len(rows) == 3
    assert rows[0]["verdict"] == "supported"
    assert rows[1]["verdict"] == "unsupported"
    assert rows[2]["verdict"] == "unsupported"


def test_per_query_and_macro_supported_ratio() -> None:
    qrow = {
        "status": "ok",
        "judgments": [
            {"verdict": "supported"},
            {"verdict": "contradicted"},
            {"verdict": "unsupported"},
        ],
    }
    assert fac.per_query_supported_ratio(qrow["judgments"]) == pytest.approx(1 / 3)
    assert fac.macro_mean_supported_ratio([qrow, {"status": "skip"}]) == pytest.approx(1 / 3)


def test_load_factuality_reports_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query": "Q?",
                        "language": "en",
                        "gold_claims": ["c1", "c2", "c3"],
                        "report": "# Hi",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    items = fac.load_factuality_reports_json(path)
    assert len(items) == 1
    assert items[0].gold_claims == ["c1", "c2", "c3"]


def test_judge_factuality_uses_mocked_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(  # type: ignore[no-untyped-def]
        model: str,
        prompt: str,
        schema: type[fac.FactualityJudgeOutput],
        *,
        system: str | None = None,
        **kwargs: object,
    ) -> tuple[fac.FactualityJudgeOutput, LLMCallResult]:
        _ = model, prompt, system, kwargs
        return (
            fac.FactualityJudgeOutput(
                judgments=[
                    fac.FactualityClaimJudgment(
                        claim_index=0,
                        verdict="supported",
                        rationale_brief="yes",
                    ),
                    fac.FactualityClaimJudgment(
                        claim_index=1,
                        verdict="unsupported",
                        rationale_brief="no mention",
                    ),
                ],
            ),
            LLMCallResult(
                text="{}",
                tokens_in=10,
                tokens_out=20,
                cost_usd=0.0002,
                model="stub",
            ),
        )

    monkeypatch.setattr(fac, "invoke_structured_llm", fake_invoke)
    rows, res = fac.judge_factuality(
        query="What is RAG?",
        output_language="en",
        report_markdown="# Report\nx",
        gold_claims=["g0", "g1"],
        current_cost_usd=0.0,
        per_query_cap_usd=1.0,
    )
    assert len(rows) == 2
    assert rows[0]["verdict"] == "supported"
    assert rows[1]["verdict"] == "unsupported"
    assert res.cost_usd == 0.0002


def test_factuality_judge_templates_render() -> None:
    sys = render("factuality_judge_system_v1.jinja")
    assert "supported" in sys.lower()
    usr = render(
        "factuality_judge_user_v1.jinja",
        query="Test?",
        output_language="en",
        gold_claims_block="0. A\n1. B",
        report_markdown="# Body",
    )
    assert "Test?" in usr
    assert "# Body" in usr
