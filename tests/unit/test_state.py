"""Tests for :mod:`research_assistant.graph.state`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_assistant.graph.state import (
    Citation,
    Draft,
    Evidence,
    SearchHit,
    StepLog,
    SubQuestion,
    new_state,
)


def test_sub_question_id_pattern() -> None:
    SubQuestion(id="sq_1", question="What is RAG?")
    with pytest.raises(ValidationError):
        SubQuestion(id="bad-id", question="x")
    with pytest.raises(ValidationError):
        SubQuestion(id="sq_1", question="")


def test_search_hit_url_validation() -> None:
    hit = SearchHit(
        url="https://example.com/a",  # type: ignore[arg-type]
        title="t",
        snippet="s",
    )
    assert hit.source == "web"
    assert hit.score is None

    with pytest.raises(ValidationError):
        SearchHit(url="not a url", title="t", snippet="s")  # type: ignore[arg-type]


def test_evidence_and_citation_round_trip() -> None:
    hit = SearchHit(
        url="https://example.com/a",  # type: ignore[arg-type]
        title="t",
        snippet="s",
    )
    ev = Evidence(ref_label="ev_sq_1_1", sub_question_id="sq_1", hit=hit)
    cit = Citation(marker=1, ref_label=ev.ref_label)
    assert cit.marker == 1
    assert cit.ref_label == "ev_sq_1_1"


def test_draft_validation() -> None:
    Draft(sub_question_id="sq_1", content="hello", model="claude-test")
    with pytest.raises(ValidationError):
        Draft(sub_question_id="sq_1", content="x", model="m", tokens_in=-1)
    with pytest.raises(ValidationError):
        Draft(sub_question_id="sq_1", content="x", model="m", cost_usd=-0.01)


def test_step_log_defaults_are_populated() -> None:
    log = StepLog(node="planner")
    assert log.status == "ok"
    assert log.duration_ms == 0.0
    assert log.started_at is not None


def test_new_state_has_expected_shape() -> None:
    state = new_state("Explain RAG", output_language="en", max_iterations=5)
    assert state["query"] == "Explain RAG"
    assert state["output_language"] == "en"
    assert state["plan"] == []
    assert state["evidence"] == {}
    assert state["drafts"] == {}
    assert state["iterations"] == 0
    assert state["max_iterations"] == 5
    assert state["current_sub_question_index"] == 0
    assert state["final_report"] is None
    assert state["total_cost_usd"] == 0.0
