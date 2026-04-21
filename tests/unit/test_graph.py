"""End-to-end graph test with every external dependency mocked."""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant.agents._llm import LLMCallResult
from research_assistant.graph.research_graph import build_graph
from research_assistant.graph.state import SearchHit, new_state


def _make_search_stub() -> Any:
    def _fn(query: str, *, max_results: int = 5) -> list[SearchHit]:
        return [
            SearchHit(
                url=f"https://example.com/{i}",  # type: ignore[arg-type]
                title=f"Doc {i} for: {query[:30]}",
                snippet=f"Relevant snippet {i} about {query[:20]}.",
                score=0.9 - i * 0.05,
                published_date="2025-10-15",
            )
            for i in range(1, max_results + 1)
        ]

    return _fn


_PLANNER_JSON = """
[
  {"id": "sq_1", "question": "What is RAG?", "rationale": "definition", "suggested_tools": ["web_search"], "dependency_ids": []},
  {"id": "sq_2", "question": "When use RAG vs fine-tune?", "rationale": "trade-off", "suggested_tools": ["web_search"], "dependency_ids": ["sq_1"]},
  {"id": "sq_3", "question": "What are limits of RAG?", "rationale": "gaps", "suggested_tools": ["web_search"], "dependency_ids": []}
]
"""


def _llm_stub_factory() -> Any:
    """Return a monkeypatch-able stub producing planner JSON for the first
    call and short citation-annotated answers for subsequent calls.
    """
    call_counter = {"n": 0}

    def _fn(model: str, prompt: str, **kwargs: Any) -> LLMCallResult:
        call_counter["n"] += 1
        if "research planner" in prompt.lower():
            text = _PLANNER_JSON
        else:
            text = "This is a synthesized answer grounded in the sources [^1][^2]."
        return LLMCallResult(
            text=text,
            tokens_in=200,
            tokens_out=80,
            cost_usd=0.001,
            model=model,
        )

    return _fn, call_counter


def test_graph_runs_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_fn, counter = _llm_stub_factory()
    monkeypatch.setattr("research_assistant.agents.planner.invoke_llm", stub_fn)
    monkeypatch.setattr("research_assistant.agents.synthesizer.invoke_llm", stub_fn)

    graph = build_graph(search_fn=_make_search_stub())
    initial = new_state("Explain RAG and its trade-offs.", max_iterations=10)

    final: Any = graph.invoke(initial)

    assert final["final_report"] is not None
    assert "Explain RAG" in final["final_report"]
    assert len(final["plan"]) == 3
    # Every sub-question should have at least one draft and some evidence.
    assert set(final["drafts"].keys()) == {"sq_1", "sq_2", "sq_3"}
    assert all(len(final["evidence"][sq.id]) > 0 for sq in final["plan"])
    # Planner (1) + 3 synthesizer calls = 4 LLM calls total.
    assert counter["n"] == 4
    # Trace should include every node at least once.
    nodes = {step.node for step in final["trace"]}
    assert {"planner", "retriever", "synthesizer", "reporter"} <= nodes


def test_graph_respects_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_fn, _ = _llm_stub_factory()
    monkeypatch.setattr("research_assistant.agents.planner.invoke_llm", stub_fn)
    monkeypatch.setattr("research_assistant.agents.synthesizer.invoke_llm", stub_fn)

    graph = build_graph(search_fn=_make_search_stub())
    # max_iterations=1 forces the loop to exit after the first synthesizer
    # call, leaving sub_q 2 and 3 unanswered but still producing a report.
    initial = new_state("A simple query", max_iterations=1)

    final: Any = graph.invoke(initial)
    assert final["final_report"] is not None
    assert len(final["drafts"]) == 1  # only sq_1 got synthesized
