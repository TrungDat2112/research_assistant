"""Critic agent (draft) — judges a single sub-question draft before advancing.

Combines:
  * **Deterministic citation coverage** on paragraph granularity (ADR-005: ≥90%).
  * **Structured Sonnet output** for whether the draft answers the sub-question
    and overall quality (PLAN §6.3).

Routes the LangGraph: retry retrieval + synthesis when the critique fails and
attempt budget remains; otherwise force-pass and advance (issues recorded).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_assistant.agents._llm import invoke_structured_llm
from research_assistant.config import get_settings
from research_assistant.graph.state import (
    Critique,
    ResearchState,
    StepLog,
)
from research_assistant.observability import observe, update_span
from research_assistant.prompts.loader import render

logger = logging.getLogger(__name__)

_CITATION_MARKER_RE = re.compile(r"\[\^\d+\]")


def _is_insufficient_fallback(text: str) -> bool:
    """True when the Synthesizer used the explicit no-evidence template."""
    t = text.lower()
    return (
        "insufficient evidence" in t
        or "chưa đủ dữ liệu" in t
        or "không đủ dữ liệu để kết luận" in t
    )


def paragraph_citation_stats(
    text: str,
    *,
    skip_paragraph: Callable[[str], bool] | None = None,
    apply_full_body_insufficient_guard: bool = True,
) -> tuple[float, int, int]:
    """Share of substantive paragraphs with ``[^N]`` plus counts.

    Returns ``(1.0, 0, 0)`` for empty text, explicit insufficient-evidence
    bodies, or when no substantive paragraphs remain after filtering.

    ``skip_paragraph`` may exclude paragraphs (e.g. Markdown headings) from
    both numerator and denominator.

    When ``apply_full_body_insufficient_guard`` is False, the draft is never
    short-circuited as insufficient-only (used for stitched multi-section
    reports where some sections disclaim lack of evidence).
    """
    raw = text.strip()
    if not raw:
        return (1.0, 0, 0)
    if apply_full_body_insufficient_guard and _is_insufficient_fallback(raw):
        return (1.0, 0, 0)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paragraphs:
        paragraphs = [raw]

    cited = 0
    total = 0
    for p in paragraphs:
        if skip_paragraph is not None and skip_paragraph(p):
            continue
        if len(p) < 24:
            continue
        total += 1
        if _CITATION_MARKER_RE.search(p):
            cited += 1

    if total == 0:
        return (1.0, 0, 0)
    return (cited / total, cited, total)


def paragraph_citation_coverage(text: str) -> float:
    """Share of substantive paragraphs that contain at least one ``[^N]`` marker.

    Returns ``1.0`` for empty text, explicit insufficient-evidence bodies, or
    when no substantive paragraphs are detected (short answers).
    """
    return paragraph_citation_stats(text)[0]


class _CritiqueDraft(BaseModel):
    """Loose structured output from the Critic LLM."""

    addresses_sub_question: bool = Field(
        ...,
        description="True if the draft directly answers the sub-question using the evidence.",
    )
    overall_score: int = Field(..., ge=1, le=5, description="1-5 holistic quality.")
    should_pass: bool = Field(
        ...,
        description="True if the draft is acceptable to advance without another retrieval round.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems (missing aspects, weak grounding, etc.).",
    )
    suggested_fixes: list[str] = Field(
        default_factory=list,
        description="Actionable fixes for the Synthesizer on retry (bullet phrases).",
    )


def _merge_pass(
    *,
    det_cov: float,
    llm: _CritiqueDraft,
    threshold: float,
) -> tuple[bool, list[str]]:
    """Combine deterministic coverage with LLM judgment."""
    reasons: list[str] = []
    cov_ok = det_cov + 1e-9 >= threshold
    if not cov_ok:
        reasons.append(
            f"citation_coverage_below_{int(threshold * 100)}_pct (paragraph_metric={det_cov:.2f})",
        )
    if not llm.addresses_sub_question:
        reasons.append("does_not_address_sub_question")
    if llm.overall_score < 4:
        reasons.append(f"overall_score_below_4 (score={llm.overall_score})")
    if not llm.should_pass:
        reasons.append("llm_should_pass_false")

    merged = cov_ok and llm.addresses_sub_question and llm.overall_score >= 4 and llm.should_pass
    return merged, reasons


@observe(name="critic", as_type="span", capture_input=False, capture_output=False)
def critic_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: critique the draft for ``current_sub_question_index``."""
    started = time.perf_counter()
    plan = state.get("plan", [])
    idx = state.get("current_sub_question_index", 0)
    settings = get_settings()
    prior_iter = state.get("iterations", 0)

    if idx >= len(plan):
        return {
            "critic_route_next": "tick",
            "trace": [
                StepLog(
                    node="critic",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="skipped",
                    details={"reason": "index_past_plan_end"},
                ),
            ],
        }

    sub_q = plan[idx]
    draft = state.get("drafts", {}).get(sub_q.id)
    evidence_list = list(state.get("evidence", {}).get(sub_q.id, []))

    override = state.get("critic_enabled_override")
    critic_on = settings.critic_enabled if override is None else override

    if draft is None:
        return {
            "critic_route_next": "tick",
            "current_sub_question_index": idx + 1,
            "synth_critic_feedback": None,
            "iterations": prior_iter + 1,
            "trace": [
                StepLog(
                    node="critic",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="error",
                    details={"sub_question_id": sub_q.id, "reason": "missing_draft"},
                ),
            ],
        }

    if not critic_on:
        critique = Critique(
            sub_question_id=sub_q.id,
            passed=True,
            forced_pass=False,
            overall_score=5,
            paragraph_citation_coverage=paragraph_citation_coverage(draft.content),
            addresses_sub_question=True,
            issues=[],
            suggested_fixes=[],
            model="(disabled)",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )
        update_span(
            input={"sub_question_id": sub_q.id, "mode": "disabled"},
            output={"passed": True},
        )
        return {
            "critiques": {sub_q.id: critique},
            "current_sub_question_index": idx + 1,
            "synth_critic_feedback": None,
            "critic_route_next": "tick",
            "iterations": prior_iter + 1,
            "trace": [
                StepLog(
                    node="critic",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="ok",
                    details={
                        "sub_question_id": sub_q.id,
                        "passed": True,
                        "mode": "disabled",
                    },
                ),
            ],
        }

    det_cov = paragraph_citation_coverage(draft.content)
    attempt = int(state.get("critic_attempts", {}).get(sub_q.id, 0))
    max_attempts = settings.critic_max_attempts_per_sub_question
    # attempt is 0 on first critique after first synthesis; retry bumps it in-router.
    tries_left = max(0, max_attempts - attempt - 1)

    cost_before = state.get("total_cost_usd", 0.0)
    cap = state.get("per_query_cap_usd")

    try:
        system = render("critic_system_v1.jinja")
        cacheable_prefix = render(
            "critic_user_prefix_v1.jinja",
            user_query=state.get("query", ""),
        )
        user_rest = render(
            "critic_user_rest_v1.jinja",
            sub_question=sub_q,
            draft=draft,
            evidence=evidence_list,
            paragraph_citation_coverage=round(det_cov, 4),
        )
        llm_draft, result = invoke_structured_llm(
            model=settings.anthropic_planner_model,
            prompt=user_rest,
            system=system,
            cacheable_user_prefix=cacheable_prefix,
            schema=_CritiqueDraft,
            temperature=0.0,
            max_tokens=768,
            current_cost_usd=cost_before,
            per_query_cap_usd=cap,
        )
    except Exception as exc:
        logger.exception("Critic LLM failed for %s — force pass", sub_q.id)
        critique = Critique(
            sub_question_id=sub_q.id,
            passed=True,
            forced_pass=True,
            overall_score=3,
            paragraph_citation_coverage=det_cov,
            addresses_sub_question=True,
            issues=[f"critic_error: {exc}"],
            suggested_fixes=[],
            model="(error)",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )
        return {
            "critiques": {sub_q.id: critique},
            "current_sub_question_index": idx + 1,
            "synth_critic_feedback": None,
            "critic_route_next": "tick",
            "iterations": prior_iter + 1,
            "trace": [
                StepLog(
                    node="critic",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="error",
                    details={"sub_question_id": sub_q.id, "error": str(exc), "forced_pass": True},
                ),
            ],
        }

    merged_pass, merge_reasons = _merge_pass(
        det_cov=det_cov,
        llm=llm_draft,
        threshold=settings.critic_min_paragraph_citation_coverage,
    )
    forced = False
    retry = False

    if merged_pass:
        passed = True
    elif tries_left > 0:
        passed = False
        retry = True
    else:
        passed = True
        forced = True

    critique = Critique(
        sub_question_id=sub_q.id,
        passed=passed,
        forced_pass=forced,
        overall_score=llm_draft.overall_score,
        paragraph_citation_coverage=det_cov,
        addresses_sub_question=llm_draft.addresses_sub_question,
        issues=[*llm_draft.issues, *([] if merged_pass else merge_reasons)],
        suggested_fixes=list(llm_draft.suggested_fixes),
        model=result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
    )

    update_span(
        input={
            "sub_question_id": sub_q.id,
            "attempt": attempt,
            "tries_left": tries_left,
        },
        output={
            "passed": passed,
            "forced_pass": forced,
            "retry": retry,
            "paragraph_citation_coverage": det_cov,
            "overall_score": llm_draft.overall_score,
        },
        metadata={
            "model": result.model,
            "cost_usd": round(result.cost_usd, 6),
        },
    )

    if retry:
        fixes = llm_draft.suggested_fixes or llm_draft.issues or merge_reasons
        feedback = (
            "; ".join(fixes[:6]) if fixes else "Improve citations and answer the sub-question."
        )
        return {
            "critiques": {sub_q.id: critique},
            "critic_attempts": {sub_q.id: attempt + 1},
            "synth_critic_feedback": feedback,
            "critic_route_next": "retriever",
            "total_cost_usd": cost_before + result.cost_usd,
            "iterations": prior_iter + 1,
            "trace": [
                StepLog(
                    node="critic",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="ok",
                    details={
                        "sub_question_id": sub_q.id,
                        "passed": False,
                        "retry": True,
                        "paragraph_citation_coverage": round(det_cov, 4),
                        "overall_score": llm_draft.overall_score,
                    },
                ),
            ],
        }

    return {
        "critiques": {sub_q.id: critique},
        "current_sub_question_index": idx + 1,
        "critic_attempts": {sub_q.id: 0},
        "synth_critic_feedback": None,
        "critic_route_next": "tick",
        "total_cost_usd": cost_before + result.cost_usd,
        "iterations": prior_iter + 1,
        "trace": [
            StepLog(
                node="critic",
                duration_ms=(time.perf_counter() - started) * 1000,
                status="ok",
                details={
                    "sub_question_id": sub_q.id,
                    "passed": True,
                    "forced_pass": forced,
                    "paragraph_citation_coverage": round(det_cov, 4),
                    "overall_score": llm_draft.overall_score,
                },
            ),
        ],
    }


def critic_route_edge(state: ResearchState) -> Literal["retriever", "tick"]:
    """Conditional edge target after the Critic node."""
    nxt = state.get("critic_route_next", "tick")
    if nxt == "retriever":
        return "retriever"
    return "tick"
