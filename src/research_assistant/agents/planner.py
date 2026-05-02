from __future__ import annotations

import logging
import time
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from research_assistant.agents._llm import LLMCallResult, invoke_structured_llm
from research_assistant.config import get_settings, planned_max_iterations
from research_assistant.graph.state import ResearchState, StepLog, SubQuestion
from research_assistant.observability import (
    current_trace_id,
    current_trace_url,
    observe,
    update_span,
)
from research_assistant.prompts.loader import render
from research_assistant.tools.router import sanitize_planner_suggested_tools

logger = logging.getLogger(__name__)


class _PlanItemDraft(BaseModel):
    question: str = Field(
        ...,
        description="Concrete, self-contained sub-question in the output language.",
    )
    rationale: str = Field(
        default="",
        description="One-sentence reason this sub-question is needed.",
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description=(
            "1-3 advisory tool ids: vector_search, web_search, academic_search. "
            "Invalid names are dropped at validation. Execution order always "
            "follows the rule-based router when it disagrees."
        ),
    )
    dependency_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional upstream sub-question ids this one depends on. Leave empty if independent."
        ),
    )


class _PlanDraft(BaseModel):

    sub_questions: list[_PlanItemDraft] = Field(
        ...,
        description="Between 3 and 7 sub-questions that together cover the user query.",
    )


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a valid plan."""


def _drafts_to_plan(drafts: list[_PlanItemDraft]) -> list[SubQuestion]:
    if not drafts:
        raise PlannerError("Planner returned empty sub_questions list.")

    # Synthesize stable ids up front so we can remap dependencies.
    staged: list[SubQuestion] = []
    id_remap: dict[str, str] = {}
    for idx, draft in enumerate(drafts, start=1):
        new_id = f"sq_{idx}"
        try:
            sq = SubQuestion(
                id=new_id,
                question=draft.question.strip(),
                rationale=draft.rationale.strip(),
                suggested_tools=sanitize_planner_suggested_tools(draft.suggested_tools),
                dependency_ids=list(draft.dependency_ids),
            )
        except ValidationError as exc:
            raise PlannerError(f"Plan item {idx} failed validation: {exc}") from exc
        id_remap[new_id] = new_id

        staged.append(sq)

    final_plan = [
        sq.model_copy(
            update={
                "dependency_ids": [d for d in sq.dependency_ids if d in id_remap],
            },
        )
        for sq in staged
    ]

    if not 3 <= len(final_plan) <= 7:
        logger.warning(
            "Planner returned %d sub-questions (expected 3-7). Keeping as-is.",
            len(final_plan),
        )
    return final_plan


def _fallback_plan(query: str) -> list[SubQuestion]:
    return [
        SubQuestion(
            id="sq_1",
            question=query,
            rationale="Fallback plan — Planner failed; answering the query directly.",
            suggested_tools=["web_search"],
            dependency_ids=[],
        ),
    ]


@observe(name="planner", as_type="span", capture_input=False, capture_output=False)
def planner_node(state: ResearchState) -> dict[str, Any]:
    started = time.perf_counter()
    settings = get_settings()
    model = settings.anthropic_planner_model
    query = state["query"]
    output_language = state.get("output_language", "vi")
    current_cost = state.get("total_cost_usd", 0.0)

    prompt = render("planner_v1.jinja", query=query, output_language=output_language)

    try:
        draft, result = invoke_structured_llm(
            model=model,
            prompt=prompt,
            schema=_PlanDraft,
            temperature=0.2,
            max_tokens=1024,
            current_cost_usd=current_cost,
            per_query_cap_usd=state.get("per_query_cap_usd"),
        )
        plan = _drafts_to_plan(draft.sub_questions)
        status: str = "ok"
        details: dict[str, Any] = {
            "model": result.model,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "cost_usd": round(result.cost_usd, 6),
            "n_sub_questions": len(plan),
        }
        cost_delta = result.cost_usd
    except Exception as exc:
        logger.warning("Planner structured output failed (%s); using fallback.", exc)
        plan = _fallback_plan(query)
        status = "error"
        details = {"error": str(exc)}
        cost_delta = 0.0
        result = cast(LLMCallResult, None)  # for type narrowing below

    crit_override = state.get("critic_enabled_override")
    attempts_per_sq = (
        1 if crit_override is False else int(settings.critic_max_attempts_per_sub_question)
    )
    planned = planned_max_iterations(len(plan), attempts_per_sq)
    prior_cap = state.get("max_iterations", planned)
    iter_cap = max(planned, prior_cap)
    details["planned_max_iterations"] = planned
    details["max_iterations"] = iter_cap

    elapsed_ms = (time.perf_counter() - started) * 1000
    update: dict[str, Any] = {
        "plan": plan,
        "max_iterations": iter_cap,
        "trace": [
            StepLog(
                node="planner",
                duration_ms=elapsed_ms,
                status=cast(Any, status),
                details=details,
            ),
        ],
    }
    if cost_delta:
        update["total_cost_usd"] = current_cost + cost_delta
    if not state.get("trace_id"):
        trace_id = current_trace_id()
        if trace_id:
            update["trace_id"] = trace_id
            trace_url = current_trace_url()
            if trace_url:
                update["trace_url"] = trace_url

    update_span(
        input={"query": query, "output_language": output_language},
        output={
            "n_sub_questions": len(plan),
            "sub_questions": [sq.question for sq in plan],
            "status": status,
            "max_iterations": iter_cap,
            "planned_max_iterations": planned,
        },
        metadata={
            "model": model,
            "cost_usd": cost_delta,
        },
    )
    return update
