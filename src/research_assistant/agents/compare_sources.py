from __future__ import annotations

import time
from typing import Any

from research_assistant.config import get_settings
from research_assistant.graph.state import ResearchState, StepLog
from research_assistant.observability import observe, update_span
from research_assistant.tools.compare_sources import CompareSourcesSetting, build_conflict_report


@observe(name="compare_sources", as_type="span", capture_input=False, capture_output=False)
def compare_sources_node(state: ResearchState) -> dict[str, Any]:
    started = time.perf_counter()
    plan = state.get("plan", [])
    idx = state.get("current_sub_question_index", 0)
    settings = get_settings()

    if idx >= len(plan):
        return {
            "trace": [
                StepLog(
                    node="compare_sources",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="skipped",
                    details={"reason": "index_past_plan_end"},
                ),
            ],
        }

    sub_q = plan[idx]
    draft = state.get("drafts", {}).get(sub_q.id)
    evidence_list = list(state.get("evidence", {}).get(sub_q.id, []))

    raw_mode = state.get("compare_sources_mode_override")
    mode_override: CompareSourcesSetting | None = (
        raw_mode if raw_mode in ("off", "heuristic", "auto") else None
    )
    effective_mode: CompareSourcesSetting = (
        mode_override if mode_override is not None else settings.compare_sources_mode
    )

    if draft is None:
        return {
            "trace": [
                StepLog(
                    node="compare_sources",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="error",
                    details={"sub_question_id": sub_q.id, "reason": "missing_draft"},
                ),
            ],
        }

    cost_before = float(state.get("total_cost_usd", 0.0))
    cap = state.get("per_query_cap_usd")

    report, extra_cost = build_conflict_report(
        sub_q=sub_q,
        evidence=evidence_list,
        draft=draft,
        mode=mode_override,
        cost_before=cost_before,
        per_query_cap_usd=cap,
    )

    update_span(
        input={
            "sub_question_id": sub_q.id,
            "mode": effective_mode,
            "n_evidence": len(evidence_list),
        },
        output={
            "mode_used": report.mode_used,
            "n_conflicts": len(report.items),
        },
    )

    out: dict[str, Any] = {
        "conflict_reports": {sub_q.id: report},
        "trace": [
            StepLog(
                node="compare_sources",
                duration_ms=(time.perf_counter() - started) * 1000,
                status="ok",
                details={
                    "sub_question_id": sub_q.id,
                    "mode_used": report.mode_used,
                    "n_conflicts": len(report.items),
                },
            ),
        ],
    }
    if extra_cost > 0.0:
        out["total_cost_usd"] = cost_before + extra_cost
    return out
