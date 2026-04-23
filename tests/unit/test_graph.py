"""End-to-end graph test with every external dependency mocked."""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant.agents._llm import LLMCallResult
from research_assistant.agents.planner import _PlanDraft, _PlanItemDraft
from research_assistant.config import get_settings
from research_assistant.graph.research_graph import _corpus_then_web_hits, build_graph
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


def _empty_corpus_vector_search(_query: str, **_: Any) -> list[SearchHit]:
    return []


def _rerank_take_top5(_q: str, hits: list[SearchHit]) -> list[SearchHit]:
    return hits[:5]


_PLANNER_DRAFTS: list[dict[str, Any]] = [
    {"question": "What is RAG?", "rationale": "definition"},
    {
        "question": "When use RAG vs fine-tune?",
        "rationale": "trade-off",
        "dependency_ids": ["sq_1"],
    },
    {"question": "What are limits of RAG?", "rationale": "gaps"},
]


def _structured_planner_stub() -> Any:
    def _fn(
        model: str,
        prompt: str,
        schema: type[Any],
        **kwargs: Any,
    ) -> tuple[Any, LLMCallResult]:
        obj = _PlanDraft(sub_questions=[_PlanItemDraft(**d) for d in _PLANNER_DRAFTS])
        return obj, LLMCallResult(
            text="",
            tokens_in=200,
            tokens_out=80,
            cost_usd=0.001,
            model=model,
        )

    return _fn


def _synthesizer_stub_factory() -> Any:
    """Counts synthesizer calls and returns a canned cited answer each time."""
    counter = {"n": 0}

    def _fn(model: str, prompt: str, **kwargs: Any) -> LLMCallResult:
        counter["n"] += 1
        return LLMCallResult(
            text="This is a synthesized answer grounded in the sources [^1][^2].",
            tokens_in=200,
            tokens_out=80,
            cost_usd=0.001,
            model=model,
        )

    return _fn, counter


def test_graph_runs_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    get_settings.cache_clear()
    planner_stub = _structured_planner_stub()
    synth_stub, synth_counter = _synthesizer_stub_factory()
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        planner_stub,
    )
    monkeypatch.setattr("research_assistant.agents.synthesizer.invoke_llm", synth_stub)

    graph = build_graph(
        search_fn=_make_search_stub(),
        vector_search_fn=_empty_corpus_vector_search,
        rerank_fn=_rerank_take_top5,
        retrieval_candidate_pool=5,
    )
    initial = new_state("Explain RAG and its trade-offs.", max_iterations=10)

    final: Any = graph.invoke(initial)

    assert final["final_report"] is not None
    assert "Explain RAG" in final["final_report"]
    assert len(final["plan"]) == 3
    # Every sub-question should have at least one draft and some evidence.
    assert set(final["drafts"].keys()) == {"sq_1", "sq_2", "sq_3"}
    assert all(len(final["evidence"][sq.id]) > 0 for sq in final["plan"])
    # Exactly 3 synthesizer calls (planner uses the structured path).
    assert synth_counter["n"] == 3
    # Trace should include every node at least once.
    nodes = {step.node for step in final["trace"]}
    assert {"planner", "retriever", "synthesizer", "critic", "reporter", "tick"} <= nodes


def test_graph_respects_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRITIC_ENABLED", "false")
    get_settings.cache_clear()
    planner_stub = _structured_planner_stub()
    synth_stub, _ = _synthesizer_stub_factory()
    monkeypatch.setattr(
        "research_assistant.agents.planner.invoke_structured_llm",
        planner_stub,
    )
    monkeypatch.setattr("research_assistant.agents.synthesizer.invoke_llm", synth_stub)

    graph = build_graph(
        search_fn=_make_search_stub(),
        vector_search_fn=_empty_corpus_vector_search,
        rerank_fn=_rerank_take_top5,
        retrieval_candidate_pool=5,
    )
    # max_iterations=1 forces the loop to exit after the first synthesizer
    # call, leaving sub_q 2 and 3 unanswered but still producing a report.
    initial = new_state("A simple query", max_iterations=1)

    final: Any = graph.invoke(initial)
    assert final["final_report"] is not None
    assert len(final["drafts"]) == 1  # only sq_1 got synthesized


def test_corpus_then_web_fills_from_web_with_right_budget() -> None:
    def _vsearch(q: str, **kwargs: Any) -> list[SearchHit]:
        _ = q, kwargs
        return [
            SearchHit(
                url=f"https://corpus.example/doc{i}",  # type: ignore[arg-type]
                title="Corpus",
                snippet="body",
                source="corpus",
            )
            for i in range(2)
        ]

    web_calls: list[int] = []

    def _web(q: str, *, max_results: int = 5) -> list[SearchHit]:
        _ = q
        web_calls.append(max_results)
        return [
            SearchHit(
                url=f"https://web.example/p{i}",  # type: ignore[arg-type]
                title="Web",
                snippet="snippet",
                source="web",
            )
            for i in range(max_results)
        ]

    hits, stats = _corpus_then_web_hits(
        "What is LoRA?",
        max_results=5,
        vector_fn=_vsearch,
        web_fn=_web,
    )
    assert len(hits) == 5
    assert stats["n_corpus"] == 2
    assert stats["n_web"] == 3
    assert stats["retrieval_path"] == "corpus_then_web"
    assert web_calls == [3]


def test_corpus_five_hits_skips_web_call() -> None:
    def _vsearch(q: str, **kwargs: Any) -> list[SearchHit]:
        _ = q, kwargs
        return [
            SearchHit(
                url=f"https://corpus.example/doc{i}",  # type: ignore[arg-type]
                title="C",
                snippet="b",
                source="corpus",
            )
            for i in range(5)
        ]

    def _web(_q: str, *, max_results: int = 5) -> list[SearchHit]:
        raise AssertionError("web_search should not run when corpus fills budget")

    hits, stats = _corpus_then_web_hits(
        "q",
        max_results=5,
        vector_fn=_vsearch,
        web_fn=_web,
    )
    assert len(hits) == 5
    assert stats["n_corpus"] == 5
    assert stats["n_web"] == 0
    assert stats["retrieval_path"] == "corpus_only"
