"""Reporter agent — assembles the final Markdown report.

Deterministic (no LLM call) by design — per ADR-005 the Synthesizer is the
only component allowed to generate prose, so the Reporter's job is purely:

  1. Renumber per-sub-question ``[^N]`` citation markers into a single
     globally consistent ``[^K]`` sequence across the whole report.
  2. Render :mod:`prompts.reporter_v1.jinja` with plan, drafts, evidence,
     and metadata.

Keeping this deterministic means zero hallucination risk at the final step
and zero token cost — the Reporter is free to run in CI / regression tests.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, cast

from research_assistant.graph.state import (
    Draft,
    Evidence,
    ResearchState,
    StepLog,
    SubQuestion,
)
from research_assistant.prompts.loader import render

logger = logging.getLogger(__name__)

_LOCAL_MARKER_RE = re.compile(r"\[\^(\d+)\]")


def _renumber_draft_content(
    draft: Draft,
    evidence_for_q: list[Evidence],
    global_offset: int,
) -> str:
    """Rewrite ``[^N]`` markers in ``draft.content`` to their global index.

    ``N`` is the 1-based position of an evidence within ``evidence_for_q``;
    the global marker becomes ``N + global_offset``. Out-of-range markers
    are left intact (so their presence is visible for debugging) but logged.
    """

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
) -> str:
    """Pure-function variant — used by tests and the graph node alike.

    Produces a Markdown string. Safe to call with a partial ``drafts`` dict
    (sub-questions with no draft are rendered as a "no answer" placeholder).
    """
    generated_at = generated_at or datetime.now(tz=UTC)

    # Renumber citations globally. Order follows ``plan`` so the References
    # list and inline markers agree. We materialise *new* Draft objects so
    # the original state is not mutated.
    offset = 0
    renumbered_drafts: dict[str, Draft] = {}
    for sq in plan:
        evs = evidence.get(sq.id, [])
        draft = drafts.get(sq.id)
        if draft is not None:
            rewritten = _renumber_draft_content(draft, evs, offset)
            renumbered_drafts[sq.id] = draft.model_copy(update={"content": rewritten})
        offset += len(evs)

    return render(
        "reporter_v1.jinja",
        query=query,
        output_language=output_language,
        plan=plan,
        drafts=renumbered_drafts,
        evidence=evidence,
        generated_at_iso=generated_at.isoformat(timespec="seconds"),
        total_cost_usd=total_cost_usd,
    )


def reporter_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph terminal node — writes ``final_report`` into state."""
    started = time.perf_counter()
    try:
        report = build_report(
            query=state["query"],
            output_language=state.get("output_language", "vi"),
            plan=list(state.get("plan", [])),
            drafts=dict(state.get("drafts", {})),
            evidence=dict(state.get("evidence", {})),
            total_cost_usd=state.get("total_cost_usd", 0.0),
        )
    except Exception as exc:
        logger.exception("Reporter failed; emitting minimal fallback report.")
        report = f"# {state.get('query', '(no query)')}\n\n_(Report generation failed: {exc})_\n"
        status: str = "error"
    else:
        status = "ok"

    return {
        "final_report": report,
        "trace": [
            StepLog(
                node="reporter",
                duration_ms=(time.perf_counter() - started) * 1000,
                status=cast(Any, status),
                details={
                    "length_chars": len(report),
                    "n_drafts": len(state.get("drafts", {})),
                },
            ),
        ],
    }
