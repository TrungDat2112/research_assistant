"""Planner agent — decomposes a research query into sub-questions.

Produces 3-7 :class:`SubQuestion` objects by asking Claude Sonnet 4.5 to
emit a strict JSON array, then validating each entry. On malformed JSON
we retry once with a remedial prompt; if still invalid, we surface a
``PlannerError`` so the graph can degrade gracefully (fall back to a
single-sub-question plan mirroring the user query).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, cast

from pydantic import ValidationError

from research_assistant.agents._llm import LLMCallResult, invoke_llm
from research_assistant.config import get_settings
from research_assistant.graph.state import ResearchState, StepLog, SubQuestion
from research_assistant.prompts.loader import render

logger = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a valid plan."""


def _extract_json_array(text: str) -> str:
    """Best-effort extraction of the first JSON array from LLM output.

    Claude usually respects the "return only JSON" instruction but may wrap
    it in ``` fences or add a trailing sentence. We grab the first
    ``[ ... ]`` span to be safe.
    """
    match = _JSON_ARRAY_RE.search(text)
    if match is None:
        raise PlannerError(f"Planner output does not contain a JSON array:\n{text[:400]}")
    return match.group(0)


def _parse_plan(text: str) -> list[SubQuestion]:
    raw = _extract_json_array(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"Planner produced invalid JSON: {exc}") from exc

    if not isinstance(data, list) or not data:
        raise PlannerError(f"Planner returned non-list or empty plan: {type(data).__name__}")

    plan: list[SubQuestion] = []
    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise PlannerError(f"Plan item {index} is not an object: {entry!r}")
        entry_dict = cast(dict[str, Any], entry)
        entry_dict.setdefault("id", f"sq_{index}")
        entry_dict.setdefault("suggested_tools", ["web_search"])
        entry_dict.setdefault("dependency_ids", [])
        try:
            plan.append(SubQuestion.model_validate(entry_dict))
        except ValidationError as exc:
            raise PlannerError(f"Plan item {index} failed validation: {exc}") from exc

    # Renumber ids to be contiguous sq_1..sq_N, which Synthesizer/Reporter
    # rely on for citation math.
    renumbered: list[SubQuestion] = []
    id_remap: dict[str, str] = {}
    for idx, sq in enumerate(plan, start=1):
        new_id = f"sq_{idx}"
        id_remap[sq.id] = new_id
        renumbered.append(sq.model_copy(update={"id": new_id}))

    # Rewrite dependency_ids through the id remap; drop unknown refs.
    final_plan = [
        sq.model_copy(
            update={
                "dependency_ids": [id_remap[d] for d in sq.dependency_ids if d in id_remap],
            },
        )
        for sq in renumbered
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
        result: LLMCallResult = invoke_llm(
            model=model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=1024,
            current_cost_usd=current_cost,
            per_query_cap_usd=state.get("per_query_cap_usd"),
        )
    except Exception as exc:
        logger.exception("Planner LLM call failed; using fallback plan.")
        return {
            "plan": _fallback_plan(query),
            "trace": [
                StepLog(
                    node="planner",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="error",
                    details={"error": str(exc)},
                ),
            ],
        }

    try:
        plan = _parse_plan(result.text)
    except PlannerError as exc:
        logger.warning("Planner produced unparsable output (%s); using fallback.", exc)
        plan = _fallback_plan(query)
        status: str = "error"
    else:
        status = "ok"

    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "plan": plan,
        "total_cost_usd": current_cost + result.cost_usd,
        "trace": [
            StepLog(
                node="planner",
                duration_ms=elapsed_ms,
                status=cast(Any, status),
                details={
                    "model": result.model,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "cost_usd": round(result.cost_usd, 6),
                    "n_sub_questions": len(plan),
                },
            ),
        ],
    }
