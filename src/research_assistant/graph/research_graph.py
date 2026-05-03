from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from research_assistant.agents.compare_sources import compare_sources_node
from research_assistant.agents.critic import critic_node, critic_route_edge
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
from research_assistant.tools.academic_search import academic_search as default_academic_search
from research_assistant.tools.router import (
    plan_for_sub_question,
    retrieval_tool_plan_differs_from_planner,
    sanitize_planner_suggested_tools,
)
from research_assistant.tools.vector_search import VectorSearchError, vector_search
from research_assistant.tools.web_search import WebSearchError, web_search_with_fallback

logger = logging.getLogger(__name__)

SearchFn = Callable[..., list[Any]]
VectorSearchFn = Callable[..., list[SearchHit]]
AcademicSearchFn = Callable[..., list[SearchHit]]
RerankFn = Callable[[str, list[SearchHit]], list[SearchHit]]


def no_cross_encoder_rerank_fn() -> RerankFn:
    def _go(query: str, hits: list[SearchHit]) -> list[SearchHit]:
        _ = query
        k = get_settings().synthesizer_evidence_top_k
        return hits[:k]

    return _go


def _default_rerank_fn() -> RerankFn:
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
) -> tuple[list[SearchHit], dict[str, Any]]:
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
            "n_academic": 0,
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
            "n_academic": 0,
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
        "n_academic": 0,
    }


def _route_then_collect(
    question: str,
    rationale: str,
    *,
    max_results: int,
    vector_fn: VectorSearchFn,
    web_fn: SearchFn,
    academic_fn: AcademicSearchFn,
    max_router_tools: int,
) -> tuple[list[SearchHit], dict[str, Any]]:
    tool_plan = plan_for_sub_question(
        question,
        rationale,
        max_tools=max_router_tools,
    )
    merged: list[SearchHit] = []
    seen_urls: set[str] = set()

    def _append_unique(raw: list[SearchHit]) -> None:
        for h in raw:
            u = str(h.url)
            if u in seen_urls:
                continue
            seen_urls.add(u)
            merged.append(h)
            if len(merged) >= max_results:
                return

    for tool in tool_plan.ordered_tools:
        need = max_results - len(merged)
        if need <= 0:
            break
        chunk: list[SearchHit] = []
        if tool == "vector_search":
            try:
                chunk = list(vector_fn(question, top_k=need))
            except VectorSearchError as exc:
                logger.warning("vector_search not available or invalid query: %s", exc)
            except Exception:
                logger.exception("Unexpected vector_search error during routed retrieval")
        elif tool == "web_search":
            try:
                chunk = list(web_fn(question, max_results=need))
            except WebSearchError as exc:
                logger.warning("web_search failed: %s", exc)
            except Exception:
                logger.exception("Unexpected web_search error during routed retrieval")
        elif tool == "academic_search":
            try:
                chunk = list(
                    academic_fn(question, max_results=min(need, 20)),
                )
            except Exception as exc:
                logger.warning("academic_search failed: %s", exc)
        else:
            logger.warning("Unknown routed tool id %r — skipping.", tool)

        _append_unique(chunk)

    n_corpus = sum(1 for h in merged if h.source == "corpus")
    n_web = sum(1 for h in merged if h.source == "web")
    n_academic = sum(1 for h in merged if h.source == "academic")
    retrieval_path = f"routed:{tool_plan.intent}:{'+'.join(tool_plan.ordered_tools)}"
    return merged[:max_results], {
        "n_corpus": n_corpus,
        "n_web": n_web,
        "n_academic": n_academic,
        "retrieval_path": retrieval_path,
        "router_intent": tool_plan.intent,
        "router_tools": "+".join(tool_plan.ordered_tools),
        "router_ordered_tools": list(tool_plan.ordered_tools),
    }


def _retriever_node_factory(
    search_fn: SearchFn,
    vector_fn: VectorSearchFn,
    academic_fn: AcademicSearchFn,
    rerank_fn: RerankFn,
    *,
    candidate_pool: int,
) -> Callable[[ResearchState], dict[str, Any]]:
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
        settings = get_settings()
        router_flag = state.get("tool_router_enabled_override")
        use_router = settings.tool_router_enabled if router_flag is None else bool(router_flag)
        try:
            if use_router:
                merged, rstats = _route_then_collect(
                    sub_q.question,
                    sub_q.rationale or "",
                    max_results=candidate_pool,
                    vector_fn=vector_fn,
                    web_fn=search_fn,
                    academic_fn=academic_fn,
                    max_router_tools=settings.tool_router_max_tools,
                )
            else:
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
            planner_suggested = sanitize_planner_suggested_tools(sub_q.suggested_tools)
            if use_router:
                router_ord = [str(x) for x in rstats.get("router_ordered_tools", [])]
                router_overrode = retrieval_tool_plan_differs_from_planner(
                    sub_q.suggested_tools,
                    router_ord,
                )
            else:
                router_ord = []
                router_overrode = False
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
                "n_academic": rstats.get("n_academic", 0),
                "retrieval_path": rstats.get("retrieval_path", "unknown"),
                "n_pool": rstats.get("n_pool", 0),
                "n_after_rerank": rstats.get("n_after_rerank", 0),
                "status": status,
                "top_urls": [str(ev.hit.url) for ev in evidence_list[:5]],
                "router_intent": rstats.get("router_intent"),
                "router_tools": rstats.get("router_tools"),
                "planner_suggested_tools": ",".join(planner_suggested),
                "router_ordered_tools": ",".join(router_ord),
                "router_overrode_planner": router_overrode,
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
                        "n_academic": rstats.get("n_academic", 0),
                        "n_pool": rstats.get("n_pool", 0),
                        "n_after_rerank": rstats.get("n_after_rerank", 0),
                        "retrieval_path": str(rstats.get("retrieval_path", "")),
                        "router_intent": str(rstats.get("router_intent") or ""),
                        "router_tools": str(rstats.get("router_tools") or ""),
                        "planner_suggested_tools": ",".join(planner_suggested),
                        "router_ordered_tools": ",".join(router_ord),
                        "router_overrode_planner": router_overrode,
                    },
                ),
            ],
        }

    return retriever_node


def _loop_condition(state: ResearchState) -> Literal["retriever", "reporter"]:
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


def _tick_node(state: ResearchState) -> dict[str, Any]:
    _ = state
    return {
        "trace": [
            StepLog(
                node="tick",
                duration_ms=0.0,
                status="ok",
                details={},
            ),
        ],
    }


def build_graph(
    *,
    search_fn: SearchFn | None = None,
    vector_search_fn: VectorSearchFn | None = None,
    academic_search_fn: AcademicSearchFn | None = None,
    rerank_fn: RerankFn | None = None,
    retrieval_candidate_pool: int | None = None,
) -> Any:

    settings = get_settings()
    fn: SearchFn = search_fn if search_fn is not None else web_search_with_fallback
    vfn: VectorSearchFn = vector_search_fn if vector_search_fn is not None else vector_search
    rfn: RerankFn = rerank_fn if rerank_fn is not None else _default_rerank_fn()
    afn: AcademicSearchFn = (
        academic_search_fn if academic_search_fn is not None else default_academic_search
    )
    pool = (
        retrieval_candidate_pool
        if retrieval_candidate_pool is not None
        else settings.retrieval_candidate_pool
    )
    retriever_node = _retriever_node_factory(fn, vfn, afn, rfn, candidate_pool=pool)

    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)  # type: ignore[arg-type]
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("compare_sources", compare_sources_node)
    builder.add_node("critic", critic_node)
    builder.add_node("tick", _tick_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "synthesizer")
    builder.add_edge("synthesizer", "compare_sources")
    builder.add_edge("compare_sources", "critic")
    builder.add_conditional_edges(
        "critic",
        critic_route_edge,
        {"retriever": "retriever", "tick": "tick"},
    )
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
    academic_search_fn: AcademicSearchFn | None = None,
    rerank_fn: RerankFn | None = None,
    retrieval_candidate_pool: int | None = None,
    critic_enabled_override: bool | None = None,
    tool_router_enabled_override: bool | None = None,
    compare_sources_mode_override: Literal["off", "heuristic", "auto"] | None = None,
    flush_langfuse: bool = True,
) -> ResearchState:

    settings = get_settings()
    initial = new_state(
        query=query,
        output_language=output_language,
        max_iterations=max_iterations if max_iterations is not None else settings.max_iterations,
        per_query_cap_usd=(
            per_query_cap_usd if per_query_cap_usd is not None else settings.per_query_cap_usd
        ),
    )
    if critic_enabled_override is not None:
        initial["critic_enabled_override"] = critic_enabled_override
    if tool_router_enabled_override is not None:
        initial["tool_router_enabled_override"] = tool_router_enabled_override
    if compare_sources_mode_override is not None:
        initial["compare_sources_mode_override"] = compare_sources_mode_override

    trace_id = current_trace_id()
    if trace_id:
        initial["trace_id"] = trace_id
        initial["trace_url"] = current_trace_url()

    graph = build_graph(
        search_fn=search_fn,
        vector_search_fn=vector_search_fn,
        academic_search_fn=academic_search_fn,
        rerank_fn=rerank_fn,
        retrieval_candidate_pool=retrieval_candidate_pool,
    )
    final = cast(ResearchState, graph.invoke(initial))

    update_trace_io(
        input={"query": query, "output_language": output_language},
        output={
            "n_sub_questions": len(final.get("plan", [])),
            "n_drafts": len(final.get("drafts", {})),
            "total_cost_usd": round(final.get("total_cost_usd", 0.0), 6),
            "report_chars": len(final.get("final_report") or ""),
            "iterations": final.get("iterations", 0),
            "max_iterations_reached": bool(final.get("max_iterations_reached", False)),
        },
    )

    if flush_langfuse:
        _lf_flush()

    return final
