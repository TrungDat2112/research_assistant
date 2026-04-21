"""LangGraph wiring for the Week-1 research pipeline.

Topology::

                    ┌──────────┐
     START ────────▶│ planner  │
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │ retriever│◀────────────┐
                    └────┬─────┘             │
                         ▼                   │ (more sub-qs left)
                    ┌──────────┐             │
                    │synthesiz.│─────────────┘
                    └────┬─────┘
                         │ (all sub-qs done)
                         ▼
                    ┌──────────┐
                    │ reporter │────▶ END
                    └──────────┘

The retriever node is defined inline here rather than as a separate agent
module because for Week 1 it only wraps the single ``web_search`` tool. It
will be extracted into :mod:`research_assistant.rag` in Week 2 when hybrid
retrieval + re-ranking land.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from research_assistant.agents.planner import planner_node
from research_assistant.agents.reporter import reporter_node
from research_assistant.agents.synthesizer import synthesizer_node
from research_assistant.graph.state import Evidence, ResearchState, StepLog
from research_assistant.tools.web_search import WebSearchError, web_search_with_fallback

logger = logging.getLogger(__name__)


# Evidence budget per sub-question — keeps prompts bounded and cost sane.
_MAX_EVIDENCE_PER_SUB_QUESTION = 5


SearchFn = Callable[..., list[Any]]


def _retriever_node_factory(search_fn: SearchFn) -> Callable[[ResearchState], dict[str, Any]]:
    """Build a retriever node closing over ``search_fn``.

    ``search_fn`` defaults to :func:`tools.web_search.web_search` in production
    but tests inject a stub to avoid hitting the Tavily API.
    """

    def retriever_node(state: ResearchState) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state.get("plan", [])
        idx = state.get("current_sub_question_index", 0)

        if idx >= len(plan):
            return {
                "trace": [
                    StepLog(
                        node="retriever",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        status="skipped",
                        details={"reason": "index_past_plan_end"},
                    ),
                ],
            }

        sub_q = plan[idx]
        try:
            hits = search_fn(sub_q.question, max_results=_MAX_EVIDENCE_PER_SUB_QUESTION)
        except WebSearchError as exc:
            logger.warning("web_search failed for %s: %s", sub_q.id, exc)
            hits = []
        except Exception as exc:
            logger.exception("Unexpected retriever error for %s", sub_q.id)
            hits = []
            error_details: dict[str, str | int | float | bool | None] = {
                "sub_question_id": sub_q.id,
                "error": str(exc),
            }
            return {
                "trace": [
                    StepLog(
                        node="retriever",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        status="error",
                        details=error_details,
                    ),
                ],
            }

        evidence_list: list[Evidence] = [
            Evidence(
                ref_label=f"ev_{sub_q.id}_{n}",
                sub_question_id=sub_q.id,
                hit=hit,
            )
            for n, hit in enumerate(hits, start=1)
        ]

        status: Literal["ok", "skipped"] = "ok" if evidence_list else "skipped"
        return {
            "evidence": {sub_q.id: evidence_list},
            "trace": [
                StepLog(
                    node="retriever",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status=status,
                    details={
                        "sub_question_id": sub_q.id,
                        "n_hits": len(evidence_list),
                    },
                ),
            ],
        }

    return retriever_node


def _loop_condition(state: ResearchState) -> Literal["retriever", "reporter"]:
    """Return next node name: loop back to retriever or hand off to reporter."""
    plan = state.get("plan", [])
    idx = state.get("current_sub_question_index", 0)
    iterations = state.get("iterations", 0)
    max_iters = state.get("max_iterations", 8)

    if iterations >= max_iters:
        logger.warning("Max iterations reached (%d); forcing report.", max_iters)
        return "reporter"
    if idx < len(plan):
        return "retriever"
    return "reporter"


def _increment_iterations(state: ResearchState) -> dict[str, Any]:
    """Tiny bookkeeping node that bumps the ReAct iteration counter.

    Decoupled so ``_loop_condition`` can read a monotonic counter without
    relying on any single business-logic node to maintain it.
    """
    return {"iterations": state.get("iterations", 0) + 1}


def build_graph(
    *,
    search_fn: SearchFn | None = None,
) -> Any:
    """Compile and return the research graph.

    Return type is ``Any`` because LangGraph's ``CompiledStateGraph`` is
    parameterised over internal types we don't need to expose to callers;
    only ``.invoke`` / ``.stream`` are used downstream.

    Parameters
    ----------
    search_fn:
        Override used in tests. Must accept ``(query, *, max_results=int)``
        and return a ``list[SearchHit]``.
    """
    fn: SearchFn = search_fn if search_fn is not None else web_search_with_fallback
    retriever_node = _retriever_node_factory(fn)

    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    # retriever is a local closure so its annotated signature doesn't match
    # LangGraph's structural node type as cleanly as module-level nodes.
    builder.add_node("retriever", retriever_node)  # type: ignore[arg-type]
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("tick", _increment_iterations)
    builder.add_node("reporter", reporter_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "synthesizer")
    builder.add_edge("synthesizer", "tick")
    builder.add_conditional_edges(
        "tick",
        _loop_condition,
        {"retriever": "retriever", "reporter": "reporter"},
    )
    builder.add_edge("reporter", END)

    return builder.compile()
