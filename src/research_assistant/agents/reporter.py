from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, cast

from research_assistant.graph.state import (
    Critique,
    Draft,
    Evidence,
    ResearchState,
    StepLog,
    SubQuestion,
)
from research_assistant.observability import observe, update_span
from research_assistant.prompts.loader import render

logger = logging.getLogger(__name__)

_LOCAL_MARKER_RE = re.compile(r"\[\^(\d+)\]")


def _conflicts_noted_rows(
    plan: list[SubQuestion],
    critiques: dict[str, Critique],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sq in plan:
        cr = critiques.get(sq.id)
        if cr is None:
            continue
        for it in cr.conflicts:
            if it.severity not in ("medium", "high"):
                continue
            refs = ", ".join(it.involved_ref_labels) if it.involved_ref_labels else ""
            rows.append(
                {
                    "sub_question_id": sq.id,
                    "sub_question": sq.question.replace("\n", " ").strip(),
                    "severity": it.severity,
                    "summary": it.summary.replace("\n", " ").strip(),
                    "refs": refs,
                },
            )
    return rows


def _renumber_draft_content(
    draft: Draft,
    evidence_for_q: list[Evidence],
    global_offset: int,
) -> str:


    def _sub(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(evidence_for_q):
            return f"[^{n + global_offset}]"
        logger.warning(
            "Draft %s contains out-of-range citation [^%d] (evidence=%d)",
            draft.sub_question_id,
            n,
            len(evidence_for_q),
        )
        return match.group(0)

    return _LOCAL_MARKER_RE.sub(_sub, draft.content)


def build_report(
    *,
    query: str,
    output_language: str,
    plan: list[SubQuestion],
    drafts: dict[str, Draft],
    evidence: dict[str, list[Evidence]],
    total_cost_usd: float,
    generated_at: datetime | None = None,
    trace_url: str | None = None,
    critiques: dict[str, Critique] | None = None,
) -> str:

    generated_at = generated_at or datetime.now(tz=UTC)

    offset = 0
    renumbered_drafts: dict[str, Draft] = {}
    for sq in plan:
        evs = evidence.get(sq.id, [])
        draft = drafts.get(sq.id)
        if draft is not None:
            rewritten = _renumber_draft_content(draft, evs, offset)
            renumbered_drafts[sq.id] = draft.model_copy(update={"content": rewritten})
        offset += len(evs)

    crit_map = critiques or {}
    conflicts_noted = _conflicts_noted_rows(plan, crit_map)

    return render(
        "reporter_v1.jinja",
        query=query,
        output_language=output_language,
        plan=plan,
        drafts=renumbered_drafts,
        evidence=evidence,
        generated_at_iso=generated_at.isoformat(timespec="seconds"),
        total_cost_usd=total_cost_usd,
        trace_url=trace_url,
        conflicts_noted=conflicts_noted,
    )


@observe(name="reporter", as_type="span", capture_input=False, capture_output=False)
def reporter_node(state: ResearchState) -> dict[str, Any]:
    started = time.perf_counter()
    plan = list(state.get("plan", []))
    idx = int(state.get("current_sub_question_index", 0))
    iters = int(state.get("iterations", 0))
    max_i = int(state.get("max_iterations", 8))
    max_iterations_reached = iters >= max_i and idx < len(plan)

    try:
        report = build_report(
            query=state["query"],
            output_language=state.get("output_language", "vi"),
            plan=plan,
            drafts=dict(state.get("drafts", {})),
            evidence=dict(state.get("evidence", {})),
            total_cost_usd=state.get("total_cost_usd", 0.0),
            trace_url=state.get("trace_url"),
            critiques=dict(state.get("critiques", {})),
        )
    except Exception as exc:
        logger.exception("Reporter failed; emitting minimal fallback report.")
        report = f"# {state.get('query', '(no query)')}\n\n_(Report generation failed: {exc})_\n"
        status: str = "error"
    else:
        status = "ok"

    update_span(
        input={
            "n_plan": len(plan),
            "n_drafts": len(state.get("drafts", {})),
        },
        output={
            "length_chars": len(report),
            "status": status,
            "max_iterations_reached": max_iterations_reached,
        },
        metadata={"total_cost_usd": round(state.get("total_cost_usd", 0.0), 6)},
    )

    return {
        "final_report": report,
        "max_iterations_reached": max_iterations_reached,
        "trace": [
            StepLog(
                node="reporter",
                duration_ms=(time.perf_counter() - started) * 1000,
                status=cast(Any, status),
                details={
                    "length_chars": len(report),
                    "n_drafts": len(state.get("drafts", {})),
                    "max_iterations_reached": max_iterations_reached,
                },
            ),
        ],
    }
