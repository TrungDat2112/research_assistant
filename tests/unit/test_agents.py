"""Tests for planner / synthesizer / reporter agent nodes.

LLM calls are replaced at module level by monkeypatching the exported
``invoke_llm`` / ``invoke_structured_llm`` symbols inside each agent
module. No real Anthropic traffic is generated.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant.agents._llm import LLMCallResult
from research_assistant.agents.planner import _PlanDraft, _PlanItemDraft, planner_node
from research_assistant.agents.reporter import build_report, reporter_node
from research_assistant.agents.synthesizer import synthesize_one, synthesizer_node
from research_assistant.graph.state import (
    ConflictItem,
    Critique,
    Draft,
    Evidence,
    SearchHit,
    SubQuestion,
    new_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hit(i: int) -> SearchHit:
    return SearchHit(
        url=f"https://example.com/{i}",  # type: ignore[arg-type]
        title=f"Doc {i}",
        snippet=f"This is snippet {i} with facts.",
        published_date="2025-10-01",
    )


def _evidence(sq_id: str, n: int) -> list[Evidence]:
    return [
        Evidence(ref_label=f"ev_{sq_id}_{i}", sub_question_id=sq_id, hit=_hit(i))
        for i in range(1, n + 1)
    ]


def _stub_llm(text: str, *, tokens_in: int = 100, tokens_out: int = 50) -> Any:
    def _fn(model: str, prompt: str, **kwargs: Any) -> LLMCallResult:
        return LLMCallResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0005,
            model=model,
        )

    return _fn


def _stub_structured_llm(
    drafts: list[dict[str, Any]],
    *,
    tokens_in: int = 120,
    tokens_out: int = 80,
) -> Any:
    """Factory producing a stand-in for :func:`invoke_structured_llm`.

    Returns a ``(_PlanDraft, LLMCallResult)`` tuple with ``drafts`` as the
    payload, matching the real signature so the planner node can be unit-
    tested without touching Anthropic.
    """

    def _fn(
        model: str,
        prompt: str,
        schema: type[Any],
        **kwargs: Any,
    ) -> tuple[Any, LLMCallResult]:
        obj = _PlanDraft(sub_questions=[_PlanItemDraft(**d) for d in drafts])
        return obj, LLMCallResult(
            text="",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0005,
            model=model,
        )

    return _fn


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_PLANNER_DRAFTS_OK: list[dict[str, Any]] = [
    {"question": "Question one", "rationale": "r1", "dependency_ids": []},
    {"question": "Question two", "rationale": "r2", "dependency_ids": ["sq_1"]},
    {"question": "Question three", "rationale": "r3", "dependency_ids": []},
]


def test_planner_returns_plan_from_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        _stub_structured_llm(_PLANNER_DRAFTS_OK),
    )
    update = planner_node(new_state("Explain RAG"))
    plan = update["plan"]
    assert [sq.id for sq in plan] == ["sq_1", "sq_2", "sq_3"]
    assert plan[1].dependency_ids == ["sq_1"]
    assert update["trace"][0].status == "ok"
    assert update["total_cost_usd"] > 0
    assert update["max_iterations"] == 8  # max(8, 3x2) with default critic attempts


def test_planner_respects_higher_prior_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        _stub_structured_llm(_PLANNER_DRAFTS_OK),
    )
    update = planner_node(new_state("Explain RAG", max_iterations=24))
    assert update["max_iterations"] == 24


def test_planner_tightens_planned_cap_when_critic_overridden_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        _stub_structured_llm(_PLANNER_DRAFTS_OK),
    )
    st = new_state("Explain RAG")
    st["critic_enabled_override"] = False
    update = planner_node(st)
    # max(8, 3 sub-questions x 1 attempt) — no Critic retries budgeted.
    assert update["trace"][0].details["planned_max_iterations"] == 8
    assert update["max_iterations"] == 8


def test_planner_drops_unknown_dependency_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    drafts = [
        {"question": "Question one", "dependency_ids": ["sq_99"]},
        {"question": "Question two", "dependency_ids": []},
        {"question": "Question three", "dependency_ids": []},
    ]
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        _stub_structured_llm(drafts),
    )
    plan = planner_node(new_state("Q?"))["plan"]
    # "sq_99" is hallucinated → dropped silently.
    assert plan[0].dependency_ids == []


def test_planner_falls_back_when_llm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_kw: Any) -> tuple[Any, LLMCallResult]:
        raise RuntimeError("anthropic 429")

    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        boom,
    )
    update = planner_node(new_state("Original query?"))
    assert len(update["plan"]) == 1
    assert update["plan"][0].question == "Original query?"
    assert update["trace"][0].status == "error"
    assert update["max_iterations"] == 8  # max(8, 1x2)
    # Fallback path does not charge any additional cost.
    assert update.get("total_cost_usd", None) is None


def test_planner_falls_back_on_invalid_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty question violates SubQuestion.min_length=3 → PlannerError → fallback.
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        _stub_structured_llm([{"question": "x"}]),
    )
    update = planner_node(new_state("Explain X"))
    assert len(update["plan"]) == 1
    assert update["trace"][0].status == "error"


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


def test_synthesize_one_extracts_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_assistant.agents.synthesizer.invoke_llm",
        _stub_llm("Claim A [^1]. Claim B [^2][^3]. Bogus [^9]."),
    )
    sq = SubQuestion(id="sq_1", question="What is RAG?")
    evidence = _evidence("sq_1", 3)

    draft = synthesize_one(sq, evidence)
    assert draft.sub_question_id == "sq_1"
    markers = sorted(c.marker for c in draft.citations)
    assert markers == [1, 2, 3]
    # Evidence 1,2,3 should be flagged used; 4 does not exist.
    assert all(e.used for e in evidence)


def test_synthesizer_node_handles_no_evidence() -> None:
    sq = SubQuestion(id="sq_1", question="What is RAG?")
    state = new_state("What is RAG?")
    state["plan"] = [sq]
    update = synthesizer_node(state)
    assert "sq_1" in update["drafts"]
    assert update["drafts"]["sq_1"].cost_usd == 0.0
    assert "current_sub_question_index" not in update
    assert update["trace"][0].status == "skipped"


def test_synthesizer_node_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_assistant.agents.synthesizer.invoke_llm",
        _stub_llm("Hello [^1]."),
    )
    sq = SubQuestion(id="sq_1", question="What is RAG?")
    state = new_state("What is RAG?")
    state["plan"] = [sq]
    state["evidence"] = {"sq_1": _evidence("sq_1", 2)}

    update = synthesizer_node(state)
    assert update["drafts"]["sq_1"].citations[0].marker == 1
    assert update["trace"][0].status == "ok"
    assert "current_sub_question_index" not in update


# ---------------------------------------------------------------------------
# Reporter (deterministic — no LLM mocking)
# ---------------------------------------------------------------------------


def test_reporter_renumbers_citations_globally() -> None:
    plan = [
        SubQuestion(id="sq_1", question="First question"),
        SubQuestion(id="sq_2", question="Second question"),
    ]
    evidence = {"sq_1": _evidence("sq_1", 2), "sq_2": _evidence("sq_2", 3)}
    drafts = {
        "sq_1": Draft(sub_question_id="sq_1", content="Fact A [^1]. Fact B [^2].", model="m"),
        "sq_2": Draft(sub_question_id="sq_2", content="Fact C [^1]. Fact D [^3].", model="m"),
    }
    report = build_report(
        query="Explain something",
        output_language="vi",
        plan=plan,
        drafts=drafts,
        evidence=evidence,
        total_cost_usd=0.01,
    )
    assert "Fact A [^1]" in report
    assert "Fact B [^2]" in report
    # sq_2 local 1 → global 3 (offset = len(sq_1 evidence) = 2).
    assert "Fact C [^3]" in report
    # sq_2 local 3 → global 5.
    assert "Fact D [^5]" in report
    # References list should have 5 entries (2 + 3).
    assert report.count("](https://example.com/") == 5


def _minimal_critique(
    *,
    sub_id: str,
    conflicts: list[ConflictItem],
) -> Critique:
    return Critique(
        sub_question_id=sub_id,
        passed=True,
        forced_pass=False,
        overall_score=4,
        faithfulness_score=4,
        completeness_score=4,
        consistency_score=4,
        paragraph_citation_coverage=1.0,
        addresses_sub_question=True,
        issues=[],
        suggested_fixes=[],
        conflicts=conflicts,
        model="m",
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
    )


def test_reporter_conflicts_noted_for_medium_or_high() -> None:
    plan = [SubQuestion(id="sq_1", question="Compare A vs B?")]
    evidence = {"sq_1": _evidence("sq_1", 1)}
    drafts = {"sq_1": Draft(sub_question_id="sq_1", content="Text [^1].", model="m")}
    critiques = {
        "sq_1": _minimal_critique(
            sub_id="sq_1",
            conflicts=[
                ConflictItem(
                    summary="Sources disagree on latency numbers",
                    severity="medium",
                    involved_ref_labels=["ev_sq_1_1"],
                ),
            ],
        ),
    }
    report = build_report(
        query="Q",
        output_language="en",
        plan=plan,
        drafts=drafts,
        evidence=evidence,
        total_cost_usd=0.0,
        critiques=critiques,
    )
    assert "## Conflicts noted" in report
    assert "latency numbers" in report
    assert "[medium]" in report


def test_reporter_skips_conflicts_noted_for_low_only() -> None:
    plan = [SubQuestion(id="sq_1", question="What is X?")]
    evidence = {"sq_1": _evidence("sq_1", 1)}
    drafts = {"sq_1": Draft(sub_question_id="sq_1", content="Y [^1].", model="m")}
    critiques = {
        "sq_1": _minimal_critique(
            sub_id="sq_1",
            conflicts=[
                ConflictItem(
                    summary="Minor rounding only",
                    severity="low",
                    involved_ref_labels=[],
                ),
            ],
        ),
    }
    report = build_report(
        query="Q",
        output_language="en",
        plan=plan,
        drafts=drafts,
        evidence=evidence,
        total_cost_usd=0.0,
        critiques=critiques,
    )
    assert "## Conflicts noted" not in report


def test_reporter_reference_lines_strip_newlines_in_title() -> None:
    plan = [SubQuestion(id="sq_1", question="What is Z?")]
    bad_title = SearchHit(
        url="https://example.com/1",  # type: ignore[arg-type]
        title="Broken\nTitle\nHere",
        snippet="s",
    )
    evidence = {"sq_1": [Evidence(ref_label="ev_sq_1_1", sub_question_id="sq_1", hit=bad_title)]}
    drafts = {"sq_1": Draft(sub_question_id="sq_1", content="Z [^1].", model="m")}
    report = build_report(
        query="Main Q",
        output_language="en",
        plan=plan,
        drafts=drafts,
        evidence=evidence,
        total_cost_usd=0.0,
    )
    ref_line = next(ln for ln in report.splitlines() if ln.startswith("[^1]:"))
    assert "\n" not in ref_line
    assert "Broken Title Here" in ref_line


def test_reporter_node_writes_final_report() -> None:
    plan = [SubQuestion(id="sq_1", question="First question")]
    state = new_state("Main research query")
    state["plan"] = plan
    state["evidence"] = {"sq_1": _evidence("sq_1", 1)}
    state["drafts"] = {
        "sq_1": Draft(sub_question_id="sq_1", content="A [^1]", model="m"),
    }
    update = reporter_node(state)
    assert update["final_report"].startswith("# Main research query")
    assert update["trace"][0].status == "ok"
