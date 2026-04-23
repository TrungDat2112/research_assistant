"""Tests for the Critic agent — deterministic metrics + graph wiring hooks."""

from __future__ import annotations

import pytest

from research_assistant.agents.critic import critic_node, paragraph_citation_coverage
from research_assistant.config import get_settings
from research_assistant.graph.state import Draft, SubQuestion, new_state


def test_paragraph_citation_coverage_full() -> None:
    text = "First claim with a cite [^1].\n\nSecond claim also cited [^2]."
    assert paragraph_citation_coverage(text) == 1.0


def test_paragraph_citation_coverage_insufficient_template_is_ok() -> None:
    text = "Chưa đủ dữ liệu để kết luận câu hỏi này từ các nguồn web công khai."
    assert paragraph_citation_coverage(text) == 1.0


def test_paragraph_citation_coverage_detects_gap() -> None:
    text = (
        "A long paragraph that states a factual claim without any citation marker.\n\n"
        "Another long paragraph that does cite properly [^1]."
    )
    assert paragraph_citation_coverage(text) == 0.5


def test_critic_disabled_auto_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    get_settings.cache_clear()
    sq = SubQuestion(id="sq_1", question="What is RAG?")
    state = new_state("Main Q")
    state["plan"] = [sq]
    state["current_sub_question_index"] = 0
    state["evidence"] = {"sq_1": []}
    state["drafts"] = {
        "sq_1": Draft(sub_question_id="sq_1", content="Short [^1].", model="m"),
    }

    update = critic_node(state)
    assert update["critiques"]["sq_1"].passed is True
    assert update["current_sub_question_index"] == 1
    assert update["critic_route_next"] == "tick"
    assert update["iterations"] == 1
