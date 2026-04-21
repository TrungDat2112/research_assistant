"""Tests for planner / synthesizer / reporter agent nodes.

LLM calls are replaced at module level by monkeypatching
``research_assistant.agents._llm.invoke_llm`` with a stub that returns a
canned :class:`LLMCallResult`. No real Anthropic traffic is generated.
"""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant.agents._llm import LLMCallResult
from research_assistant.agents.planner import planner_node
from research_assistant.agents.reporter import build_report, reporter_node
from research_assistant.agents.synthesizer import synthesize_one, synthesizer_node
from research_assistant.graph.state import (
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


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_PLANNER_JSON_OK = """
[
  {"id": "sq_1", "question": "Question one", "rationale": "r1", "suggested_tools": ["web_search"], "dependency_ids": []},
  {"id": "sq_2", "question": "Question two", "rationale": "r2", "suggested_tools": ["web_search"], "dependency_ids": ["sq_1"]},
  {"id": "sq_3", "question": "Question three", "rationale": "r3", "suggested_tools": ["web_search"], "dependency_ids": []}
]
"""


def test_planner_parses_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_llm",
        _stub_llm(_PLANNER_JSON_OK),
    )
    state = new_state("Explain RAG")
    update = planner_node(state)
    plan = update["plan"]
    assert [sq.id for sq in plan] == ["sq_1", "sq_2", "sq_3"]
    assert plan[1].dependency_ids == ["sq_1"]
    assert update["trace"][0].status == "ok"
    assert update["total_cost_usd"] > 0


def test_planner_handles_markdown_wrapped_json(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped = "```json\n" + _PLANNER_JSON_OK + "\n```"
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_llm",
        _stub_llm(wrapped),
    )
    update = planner_node(new_state("Some query"))
    assert len(update["plan"]) == 3


def test_planner_falls_back_when_llm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_kw: Any) -> LLMCallResult:
        raise RuntimeError("anthropic 429")

    monkeypatch.setattr("research_assistant.agents.planner.invoke_llm", boom)
    update = planner_node(new_state("Original query?"))
    assert len(update["plan"]) == 1
    assert update["plan"][0].question == "Original query?"
    assert update["trace"][0].status == "error"


def test_planner_falls_back_on_garbage_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_llm",
        _stub_llm("not json at all"),
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
    assert update["current_sub_question_index"] == 1
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
    assert update["current_sub_question_index"] == 1


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


def test_reporter_node_writes_final_report() -> None:
    plan = [SubQuestion(id="sq_1", question="First question")]
    state = new_state("Main research query")
    state["plan"] = plan
    state["evidence"] = {"sq_1": _evidence("sq_1", 1)}
    state["drafts"] = {
        "sq_1": Draft(sub_question_id="sq_1", content="A [^1]", model="m"),
    }
    update = reporter_node(state)
    assert update["final_report"].startswith("# Main research query")  # type: ignore[union-attr]
    assert update["trace"][0].status == "ok"
