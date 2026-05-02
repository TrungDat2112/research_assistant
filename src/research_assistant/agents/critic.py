from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from research_assistant.agents._llm import invoke_structured_llm
from research_assistant.config import get_settings
from research_assistant.graph.state import (
    ConflictItem,
    Critique,
    ResearchState,
    StepLog,
)
from research_assistant.observability import observe, update_span
from research_assistant.prompts.loader import render

logger = logging.getLogger(__name__)

_CITATION_MARKER_RE = re.compile(r"\[\^\d+\]")


def _is_insufficient_fallback(text: str) -> bool:
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

    return paragraph_citation_stats(text)[0]


def consistency_score_from_conflicts(items: Sequence[ConflictItem]) -> int:
    
    if not items:
        return 5
    has_high = any(x.severity == "high" for x in items)
    has_medium = any(x.severity == "medium" for x in items)
    if has_high:
        return 2
    if has_medium:
        return 3
    return 4


def _axis_meets_bar(score: int, minimum: float) -> bool:
    return score + 1e-9 >= minimum


class _CritiqueDraft(BaseModel):
    addresses_sub_question: bool = Field(
        ...,
        description="True if the draft fully addresses the sub-question scope.",
    )
    faithfulness_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="1-5: claims are supported by the cited evidence, no overreach.",
    )
    completeness_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="1-5: all important aspects of the sub-question are covered.",
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
    citation_threshold: float,
    consistency_score: int,
    min_faithfulness: float,
    min_completeness: float,
    min_consistency: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    cov_ok = det_cov + 1e-9 >= citation_threshold
    if not cov_ok:
        reasons.append(
            f"citation_coverage_below_{int(citation_threshold * 100)}_pct (paragraph_metric={det_cov:.2f})",
        )
    if not llm.addresses_sub_question:
        reasons.append("does_not_address_sub_question")
    if not _axis_meets_bar(llm.faithfulness_score, min_faithfulness):
        reasons.append(
            f"faithfulness_below_min (score={llm.faithfulness_score}, min={min_faithfulness:g})",
        )
    if not _axis_meets_bar(llm.completeness_score, min_completeness):
        reasons.append(
            f"completeness_below_min (score={llm.completeness_score}, min={min_completeness:g})",
        )
    if not _axis_meets_bar(consistency_score, min_consistency):
        reasons.append(
            f"consistency_below_min (score={consistency_score}, min={min_consistency:g})",
        )
    if llm.overall_score < 4:
        reasons.append(f"overall_score_below_4 (score={llm.overall_score})")
    if not llm.should_pass:
        reasons.append("llm_should_pass_false")

    merged = (
        cov_ok
        and llm.addresses_sub_question
        and _axis_meets_bar(llm.faithfulness_score, min_faithfulness)
        and _axis_meets_bar(llm.completeness_score, min_completeness)
        and _axis_meets_bar(consistency_score, min_consistency)
        and llm.overall_score >= 4
        and llm.should_pass
    )
    return merged, reasons


def _forced_pass_conflict_block(conflicts: Sequence[ConflictItem]) -> list[str]:
    if not conflicts:
        return []
    lines = [
        "forced_pass_with_active_conflicts: reconcile or explicitly disclose source disagreements.",
    ]
    for c in conflicts:
        if c.severity in ("high", "medium"):
            lines.append(f"conflict_{c.severity}: {c.summary}")
    return lines


def _conflicts_for_subq(state: ResearchState, sub_q_id: str) -> list[ConflictItem]:
    rep = state.get("conflict_reports", {}).get(sub_q_id)
    return list(rep.items) if rep else []


@observe(name="critic", as_type="span", capture_input=False, capture_output=False)
def critic_node(state: ResearchState) -> dict[str, Any]:
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

    conflict_rows = _conflicts_for_subq(state, sub_q.id)
    consistency = consistency_score_from_conflicts(conflict_rows)

    if not critic_on:
        critique = Critique(
            sub_question_id=sub_q.id,
            passed=True,
            forced_pass=False,
            overall_score=5,
            faithfulness_score=5,
            completeness_score=5,
            consistency_score=consistency,
            paragraph_citation_coverage=paragraph_citation_coverage(draft.content),
            addresses_sub_question=True,
            issues=[],
            suggested_fixes=[],
            conflicts=conflict_rows,
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
            conflict_report=state.get("conflict_reports", {}).get(sub_q.id),
            consistency_score=consistency,
        )
        llm_draft, result = invoke_structured_llm(
            model=settings.anthropic_planner_model,
            prompt=user_rest,
            system=system,
            cacheable_user_prefix=cacheable_prefix,
            schema=_CritiqueDraft,
            temperature=0.0,
            max_tokens=896,
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
            faithfulness_score=3,
            completeness_score=3,
            consistency_score=consistency,
            paragraph_citation_coverage=det_cov,
            addresses_sub_question=True,
            issues=[f"critic_error: {exc}", *_forced_pass_conflict_block(conflict_rows)],
            suggested_fixes=[],
            conflicts=conflict_rows,
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
        citation_threshold=settings.critic_min_paragraph_citation_coverage,
        consistency_score=consistency,
        min_faithfulness=settings.critic_min_faithfulness,
        min_completeness=settings.critic_min_completeness,
        min_consistency=settings.critic_min_consistency,
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

    issue_tail: list[str] = []
    if not merged_pass:
        issue_tail = list(merge_reasons)
    if forced:
        issue_tail = [*issue_tail, *_forced_pass_conflict_block(conflict_rows)]

    critique = Critique(
        sub_question_id=sub_q.id,
        passed=passed,
        forced_pass=forced,
        overall_score=llm_draft.overall_score,
        faithfulness_score=llm_draft.faithfulness_score,
        completeness_score=llm_draft.completeness_score,
        consistency_score=consistency,
        paragraph_citation_coverage=det_cov,
        addresses_sub_question=llm_draft.addresses_sub_question,
        issues=[*llm_draft.issues, *issue_tail],
        suggested_fixes=list(llm_draft.suggested_fixes),
        conflicts=conflict_rows,
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
            "faithfulness_score": llm_draft.faithfulness_score,
            "completeness_score": llm_draft.completeness_score,
            "consistency_score": consistency,
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
                        "faithfulness_score": llm_draft.faithfulness_score,
                        "completeness_score": llm_draft.completeness_score,
                        "consistency_score": consistency,
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
                    "faithfulness_score": llm_draft.faithfulness_score,
                    "completeness_score": llm_draft.completeness_score,
                    "consistency_score": consistency,
                    "overall_score": llm_draft.overall_score,
                },
            ),
        ],
    }


def critic_route_edge(state: ResearchState) -> Literal["retriever", "tick"]:
    nxt = state.get("critic_route_next", "tick")
    if nxt == "retriever":
        return "retriever"
    return "tick"
