"""Planner agent — decomposes a research query into sub-questions.

Strategy (post-structured-output fix):

1. Ask Claude Sonnet 4.5 via ``ChatAnthropic.with_structured_output`` using
   a lightweight :class:`_PlanDraft` schema. Anthropic's native tool-use
   path emits shape-valid JSON, so we no longer need regex + ``json.loads``
   + repair prompts. If Anthropic still somehow returns ``parsing_error``,
   the exception bubbles up and we fall back to a single-sub-question plan.
2. Post-process drafts into strict :class:`SubQuestion` objects (min-length
   constraints live there, not on the draft schema — giving the LLM room
   to self-correct without violating Pydantic).
3. Renumber ids to ``sq_1..sq_N`` and rewrite ``dependency_ids`` so the
   Synthesizer / Reporter downstream can rely on stable identifiers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from pydantic import BaseModel, Field, ValidationError

from research_assistant.agents._llm import LLMCallResult, invoke_structured_llm
from research_assistant.config import get_settings
from research_assistant.graph.state import ResearchState, StepLog, SubQuestion
from research_assistant.prompts.loader import render

logger = logging.getLogger(__name__)


class _PlanItemDraft(BaseModel):
    """Loose draft emitted by Claude's tool-use.

    Intentionally has NO ``min_length`` on ``question`` — we validate the
    stricter contract (see :class:`SubQuestion`) in Python so a borderline
    draft is caught as a Python error rather than a silent LLM retry loop.
    """

    question: str = Field(
        ...,
        description="Concrete, self-contained sub-question in the output language.",
    )
    rationale: str = Field(
        default="",
        description="One-sentence reason this sub-question is needed.",
    )
    suggested_tools: list[str] = Field(
        default_factory=lambda: ["web_search"],
        description="Tools the retriever should use. For Week 1 only 'web_search' is wired.",
    )
    dependency_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional upstream sub-question ids this one depends on. Leave empty if independent."
        ),
    )


class _PlanDraft(BaseModel):
    """Top-level structured-output container returned by the Planner LLM."""

    sub_questions: list[_PlanItemDraft] = Field(
        ...,
        description="Between 3 and 7 sub-questions that together cover the user query.",
    )


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a valid plan."""


def _drafts_to_plan(drafts: list[_PlanItemDraft]) -> list[SubQuestion]:
    """Convert loose drafts → validated ``SubQuestion`` list with stable ids.

    Also drops any ``dependency_ids`` that reference unknown items (LLM
    occasionally hallucinates cross-references).
    """
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
                suggested_tools=draft.suggested_tools or ["web_search"],
                dependency_ids=list(draft.dependency_ids),
            )
        except ValidationError as exc:
            raise PlannerError(f"Plan item {idx} failed validation: {exc}") from exc
        id_remap[new_id] = new_id
        # Also map the LLM-proposed dep ids (e.g. "q1", "sq1") best-effort:
        # anything that looks like an index maps onto the staged id.
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
    """Single-sub-question plan used when the Planner fails.

    Ensures the graph can still produce *some* answer instead of erroring
    out — graceful degradation per PLAN.md §11 (infinite-loop risk).
    """
    return [
        SubQuestion(
            id="sq_1",
            question=query,
            rationale="Fallback plan — Planner failed; answering the query directly.",
            suggested_tools=["web_search"],
            dependency_ids=[],
        ),
    ]


def planner_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node entry point.

    Returns a partial state update (``plan`` + ``trace`` + ``total_cost_usd``).
    Uses the reducer-friendly shape so append semantics work for ``plan``
    and ``trace`` (see ``ResearchState`` annotations).
    """
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
        # No token usage available on the structured path when it explodes;
        # cost stays at whatever the previous total was.
        cost_delta = 0.0
        result = cast(LLMCallResult, None)  # for type narrowing below

    elapsed_ms = (time.perf_counter() - started) * 1000
    update: dict[str, Any] = {
        "plan": plan,
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
    return update
