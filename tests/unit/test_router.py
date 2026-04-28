"""Unit tests for :mod:`research_assistant.tools.router` heuristics."""

from __future__ import annotations

from research_assistant.tools.router import (
    build_tool_plan,
    classify_intent,
    plan_for_sub_question,
    retrieval_tool_plan_differs_from_planner,
    sanitize_planner_suggested_tools,
)


def test_classify_factual_default() -> None:
    assert classify_intent("What is the weather?") == "factual"


def test_classify_comparative_vs() -> None:
    assert classify_intent("RAG vs fine-tuning for small data") == "comparative"


def test_classify_comparative_vi() -> None:
    assert classify_intent("So sánh LoRA với full fine-tuning?") == "comparative"


def test_classify_internal_corpus() -> None:
    assert classify_intent("Summarize findings from our indexed corpus") == "internal_corpus"


def test_classify_academic() -> None:
    assert classify_intent("Latest arxiv papers on retrieval") == "academic"


def test_build_tool_plan_clamps_max_tools() -> None:
    p = build_tool_plan("academic", max_tools=2)
    assert p.intent == "academic"
    assert p.ordered_tools == ["academic_search", "vector_search"]
    assert p.primary == "academic_search"
    assert p.optional == ["vector_search"]


def test_plan_for_sub_question_integrates() -> None:
    p = plan_for_sub_question(
        "Explain benchmark results",
        "methodology",
        max_tools=3,
    )
    assert p.intent == "academic"
    assert p.ordered_tools == ["academic_search", "vector_search", "web_search"]


def test_sanitize_drops_unknown_and_dedupes() -> None:
    assert sanitize_planner_suggested_tools(
        ["web_search", "vector_search", "no_such_tool", "web_search"],
    ) == ["web_search", "vector_search"]


def test_sanitize_empty_defaults_to_web() -> None:
    assert sanitize_planner_suggested_tools([]) == ["web_search"]
    assert sanitize_planner_suggested_tools(["unknown_only"]) == ["web_search"]


def test_retrieval_differs_when_order_or_members_differ() -> None:
    assert retrieval_tool_plan_differs_from_planner(
        ["academic_search", "web_search"],
        ["web_search", "vector_search"],
    )
    assert not retrieval_tool_plan_differs_from_planner(
        ["web_search", "vector_search"],
        ["web_search", "vector_search"],
    )
