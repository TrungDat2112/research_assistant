"""LangGraph wiring for the research pipeline (planner → retrieve → synthesize → report).

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

The retriever (1) merges hybrid corpus + Tavily
(:func:`~research_assistant.tools.vector_search.vector_search` +
:func:`web_search_with_fallback`) up to ``retrieval_candidate_pool``; (2)
re-ranks with a cross-encoder
(:func:`~research_assistant.rag.reranker.rerank_search_hits`) to
``synthesizer_evidence_top_k`` (default 5) when enabled.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from research_assistant.agents.planner import planner_node
from research_assistant.agents.reporter import reporter_node
from research_assistant.agents.synthesizer import synthesizer_node
from research_assistant.config import get_settings
from research_assistant.graph.state import Evidence, ResearchState, SearchHit, StepLog, new_state
from research_assistant.observability import (
    current_trace_id,
    current_trace_url,
    observe,
    update_span,
    update_trace_io,
)
from research_assistant.observability import (
    flush as _lf_flush,
)
from research_assistant.rag.reranker import rerank_search_hits
from research_assistant.tools.vector_search import VectorSearchError, vector_search
from research_assistant.tools.web_search import WebSearchError, web_search_with_fallback

logger = logging.getLogger(__name__)

SearchFn = Callable[..., list[Any]]
VectorSearchFn = Callable[..., list[SearchHit]]
RerankFn = Callable[[str, list[SearchHit]], list[SearchHit]]


def _default_rerank_fn() -> RerankFn:
    """Stage-2 cross-encoder, or pass-through when ``reranker_enabled`` is off."""

    def _go(query: str, hits: list[SearchHit]) -> list[SearchHit]:
        s = get_settings()
        k = s.synthesizer_evidence_top_k
        if not s.reranker_enabled or len(hits) <= 1:
            return hits[:k]
        try:
            return rerank_search_hits(query, hits, top_k=k)
        except Exception:
            logger.exception("Cross-encoder rerank failed; using first k raw hits")
            return hits[:k]

    return _go


def _dedupe_hits_by_url(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out: list[SearchHit] = []
    for h in hits:
        u = str(h.url)
        if u in seen:
            continue
        seen.add(u)
        out.append(h)
    return out


def _corpus_then_web_hits(
    question: str,
    *,
    max_results: int,
    vector_fn: VectorSearchFn,
    web_fn: SearchFn,
) -> tuple[list[SearchHit], dict[str, str | int]]:
    """Prefer ingested corpus; add Tavily results only when slots remain (dedup by URL)."""
    try:
        corpus = vector_fn(question, top_k=max_results)
    except VectorSearchError as exc:
        logger.warning("vector_search not available or invalid query: %s", exc)
        corpus = []
    except Exception:
        logger.exception("Unexpected vector_search error; falling back to web only")
        corpus = []

    merged = _dedupe_hits_by_url(corpus)
    n_corpus_in_final = sum(1 for h in merged if h.source == "corpus")
    if len(merged) >= max_results:
        path = "corpus_only" if n_corpus_in_final else "web_only"
        return merged[:max_results], {
            "n_corpus": n_corpus_in_final,
            "n_web": 0,
            "retrieval_path": path,
        }

    need = max_results - len(merged)
    seen = {str(h.url) for h in merged}
    try:
        web_hits_raw = web_fn(question, max_results=need)
    except WebSearchError as exc:
        logger.warning("web_search failed: %s", exc)
        web_hits_raw = []
    except Exception:
        logger.exception("Unexpected web_search error — using corpus partials if any")
        n_c = sum(1 for h in merged if h.source == "corpus")
        return merged, {
            "n_corpus": n_c,
            "n_web": 0,
            "retrieval_path": "corpus_only" if n_c else "web_only",
        }

    for h in web_hits_raw:
        u = str(h.url)
        if u in seen:
            continue
        seen.add(u)
        merged.append(h)
        if len(merged) >= max_results:
            break

    n_corpus_in_final = sum(1 for h in merged if h.source == "corpus")
    n_web_in_final = sum(1 for h in merged if h.source == "web")
    if n_corpus_in_final and n_web_in_final:
        path = "corpus_then_web"
    elif n_corpus_in_final:
        path = "corpus_only"
    else:
        path = "web_only"
    return merged[:max_results], {
        "n_corpus": n_corpus_in_final,
        "n_web": n_web_in_final,
        "retrieval_path": path,
    }


def _retriever_node_factory(
    search_fn: SearchFn,
    vector_fn: VectorSearchFn,
    rerank_fn: RerankFn,
    *,
    candidate_pool: int,
) -> Callable[[ResearchState], dict[str, Any]]:
    """Build a retriever: corpus + web pool, then optional cross-encoder rerank."""

    @observe(name="retriever", as_type="retriever", capture_input=False, capture_output=False)
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
            merged, rstats = _corpus_then_web_hits(
                sub_q.question,
                max_results=candidate_pool,
                vector_fn=vector_fn,
                web_fn=search_fn,
            )
            n_pool = len(merged)
            try:
                hits = rerank_fn(sub_q.question, merged)
            except Exception as exc:
                logger.exception("rerank failed for %s", sub_q.id)
                k_fallback = get_settings().synthesizer_evidence_top_k
                hits = merged[:k_fallback]
                rstats = {
                    **rstats,
                    "n_pool": n_pool,
                    "n_after_rerank": len(hits),
                    "rerank": "error",
                    "rerank_error": str(exc),
                }
            else:
                rstats = {**rstats, "n_pool": n_pool, "n_after_rerank": len(hits)}
        except Exception as exc:
            logger.exception("Unexpected retriever error for %s", sub_q.id)
            return {
                "trace": [
                    StepLog(
                        node="retriever",
                        duration_ms=(time.perf_counter() - started) * 1000,
                        status="error",
                        details={"sub_question_id": sub_q.id, "error": str(exc)},
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
        update_span(
            input={"sub_question_id": sub_q.id, "query": sub_q.question},
            output={
                "n_hits": len(evidence_list),
                "n_corpus": rstats.get("n_corpus", 0),
                "n_web": rstats.get("n_web", 0),
                "retrieval_path": rstats.get("retrieval_path", "unknown"),
                "n_pool": rstats.get("n_pool", 0),
                "n_after_rerank": rstats.get("n_after_rerank", 0),
                "status": status,
                "top_urls": [str(ev.hit.url) for ev in evidence_list[:5]],
            },
        )
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
                        "n_corpus": rstats.get("n_corpus", 0),
                        "n_web": rstats.get("n_web", 0),
                        "n_pool": rstats.get("n_pool", 0),
                        "n_after_rerank": rstats.get("n_after_rerank", 0),
                        "retrieval_path": rstats.get("retrieval_path", ""),
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
    vector_search_fn: VectorSearchFn | None = None,
    rerank_fn: RerankFn | None = None,
    retrieval_candidate_pool: int | None = None,
) -> Any:
    """Compile and return the research graph.

    Return type is ``Any`` because LangGraph's ``CompiledStateGraph`` is
    parameterised over internal types we don't need to expose to callers;
    only ``.invoke`` / ``.stream`` are used downstream.

    Parameters
    ----------
    search_fn:
        Web search, typically :func:`web_search_with_fallback`. In tests, stub
        with ``(query, *, max_results=int) -> list[SearchHit]``.
    vector_search_fn:
        Hybrid corpus search, typically :func:`vector_search`. In tests, pass
        a stub (often returning ``[]``) to avoid loading embeddings/Chroma.
    rerank_fn:
        Maps ``(sub_question, merged_hits)`` to a shorter list. Defaults to
        :func:`rerank_search_hits` when ``Settings.reranker_enabled``; tests
        should pass a lambda (e.g. ``h[:5]``) to skip the cross-encoder.
    retrieval_candidate_pool:
        Stage-1 cap on merged hits per sub-question. Defaults to
        ``get_settings().retrieval_candidate_pool``.
    """
    settings = get_settings()
    fn: SearchFn = search_fn if search_fn is not None else web_search_with_fallback
    vfn: VectorSearchFn = vector_search_fn if vector_search_fn is not None else vector_search
    rfn: RerankFn = rerank_fn if rerank_fn is not None else _default_rerank_fn()
    pool = retrieval_candidate_pool if retrieval_candidate_pool is not None else settings.retrieval_candidate_pool
    retriever_node = _retriever_node_factory(fn, vfn, rfn, candidate_pool=pool)

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


@observe(name="research_agent", as_type="agent", capture_input=False, capture_output=False)
def run_research(
    query: str,
    *,
    output_language: Literal["vi", "en"] = "vi",
    max_iterations: int | None = None,
    per_query_cap_usd: float | None = None,
    search_fn: SearchFn | None = None,
    vector_search_fn: VectorSearchFn | None = None,
    rerank_fn: RerankFn | None = None,
    retrieval_candidate_pool: int | None = None,
    flush_langfuse: bool = True,
) -> ResearchState:
    """High-level entry point used by CLI / Streamlit / smoke scripts.

    Wraps the compiled graph inside a single Langfuse ``agent`` trace so
    every node, tool call, and LLM invocation nests under one trace id.
    Without this wrapper each ``@observe`` decorator would start its own
    top-level trace, fragmenting the timeline.

    Parameters mirror :func:`graph.state.new_state` plus injectable
    ``search_fn`` / ``vector_search_fn`` / ``rerank_fn`` for tests. Returns
    the final :class:`ResearchState`.
    """
    settings = get_settings()
    initial = new_state(
        query=query,
        output_language=output_language,
        max_iterations=max_iterations if max_iterations is not None else settings.max_iterations,
        per_query_cap_usd=(
            per_query_cap_usd if per_query_cap_usd is not None else settings.per_query_cap_usd
        ),
    )

    # Capture Langfuse trace identifiers from THIS agent span (we are
    # inside ``@observe(as_type="agent")``). Doing it here rather than
    # inside a specific node sidesteps the fact that LangGraph may run
    # nodes in their own OpenTelemetry context, which would otherwise
    # see ``get_current_trace_id()`` return ``None``.
    trace_id = current_trace_id()
    if trace_id:
        initial["trace_id"] = trace_id
        initial["trace_url"] = current_trace_url()

    graph = build_graph(
        search_fn=search_fn,
        vector_search_fn=vector_search_fn,
        rerank_fn=rerank_fn,
        retrieval_candidate_pool=retrieval_candidate_pool,
    )
    final = cast(ResearchState, graph.invoke(initial))

    # Attach a compact, human-readable trace summary. Avoids dumping the
    # full state (which includes evidence content and drafts) into the
    # Langfuse UI where it would blow past payload limits.
    update_trace_io(
        input={"query": query, "output_language": output_language},
        output={
            "n_sub_questions": len(final.get("plan", [])),
            "n_drafts": len(final.get("drafts", {})),
            "total_cost_usd": round(final.get("total_cost_usd", 0.0), 6),
            "report_chars": len(final.get("final_report") or ""),
            "iterations": final.get("iterations", 0),
        },
    )

    if flush_langfuse:
        # Ensure spans reach the backend before short-lived CLI processes
        # exit. No-op when Langfuse is disabled.
        _lf_flush()

    return final
