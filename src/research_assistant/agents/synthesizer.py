"""Synthesizer agent — writes the answer for a single sub-question.

Uses Claude Haiku (cheap, fast) and enforces citation discipline:
  * Evidence is numbered ``[1], [2], ...`` in the prompt.
  * The LLM is instructed to emit ``[^N]`` markers matching those numbers.
  * We parse the markers out and rebuild them into :class:`Citation`
    objects so the Reporter (and eventual Critic) can verify coverage.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, cast

from research_assistant.agents._llm import invoke_llm
from research_assistant.config import get_settings
from research_assistant.graph.state import (
    Citation,
    Draft,
    Evidence,
    ResearchState,
    StepLog,
    SubQuestion,
)
from research_assistant.observability import observe, update_span
from research_assistant.prompts.loader import render

logger = logging.getLogger(__name__)

_CITATION_MARKER_RE = re.compile(r"\[\^(\d+)\]")


def _extract_citations(content: str, evidence: list[Evidence]) -> list[Citation]:
    """Map ``[^N]`` markers in ``content`` back to evidence ref labels.

    Markers are 1-indexed to match the prompt numbering. Out-of-range
    markers are dropped with a warning — the Critic may penalise
    drafts that leave them unresolved.
    """
    seen: set[int] = set()
    citations: list[Citation] = []
    for match in _CITATION_MARKER_RE.finditer(content):
        n = int(match.group(1))
        if n in seen:
            continue
        if 1 <= n <= len(evidence):
            seen.add(n)
            citations.append(Citation(marker=n, ref_label=evidence[n - 1].ref_label))
        else:
            logger.warning(
                "Synthesizer emitted out-of-range citation [^%d] (evidence list has %d items)",
                n,
                len(evidence),
            )
    return citations


def synthesize_one(
    sub_question: SubQuestion,
    evidence: list[Evidence],
    *,
    user_query: str = "",
    output_language: str = "vi",
    critic_feedback: str | None = None,
    current_cost_usd: float = 0.0,
    per_query_cap_usd: float | None = None,
) -> Draft:
    """Generate a single :class:`Draft` for ``sub_question``.

    Public so the graph node and tests can both call it directly.
    """
    settings = get_settings()
    model = settings.anthropic_synthesizer_model

    system = render(
        "synthesizer_system_v1.jinja",
        user_query=user_query,
        output_language=output_language,
    )
    user_prompt = render(
        "synthesizer_user_v1.jinja",
        sub_question=sub_question.question,
        evidence=evidence,
        critic_feedback=critic_feedback,
    )

    result = invoke_llm(
        model=model,
        prompt=user_prompt,
        system=system,
        temperature=0.1,
        max_tokens=1024,
        current_cost_usd=current_cost_usd,
        per_query_cap_usd=per_query_cap_usd,
    )

    citations = _extract_citations(result.text, evidence)

    # Mark cited evidence as "used" so downstream nodes can detect orphans.
    cited_labels = {c.ref_label for c in citations}
    for ev in evidence:
        if ev.ref_label in cited_labels:
            ev.used = True

    return Draft(
        sub_question_id=sub_question.id,
        content=result.text.strip(),
        citations=citations,
        model=result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
    )


@observe(name="synthesizer", as_type="span", capture_input=False, capture_output=False)
def synthesizer_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: synthesize the sub-question at ``current_sub_question_index``.

    Reads the plan and evidence from ``state``; writes the resulting
    :class:`Draft` into ``drafts`` keyed by ``sub_question_id`` and advances
    the loop index. Designed to be called in a loop by the graph.
    """
    started = time.perf_counter()
    plan = state.get("plan", [])
    idx = state.get("current_sub_question_index", 0)
    drafts_update: dict[str, Draft] = {}
    current_cost = state.get("total_cost_usd", 0.0)

    if idx >= len(plan):
        logger.debug("Synthesizer called with idx=%d but plan has %d items; noop.", idx, len(plan))
        return {
            "trace": [
                StepLog(
                    node="synthesizer",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="skipped",
                    details={"reason": "index_past_plan_end"},
                ),
            ],
        }

    sub_q = plan[idx]
    evidence_for_q = list(state.get("evidence", {}).get(sub_q.id, []))

    if not evidence_for_q:
        fallback_text = (
            "Chưa đủ dữ liệu để kết luận câu hỏi này từ các nguồn web công khai."
            if state.get("output_language", "vi") == "vi"
            else "Insufficient evidence to answer this sub-question from public web sources."
        )
        draft = Draft(
            sub_question_id=sub_q.id,
            content=fallback_text,
            citations=[],
            model="(none)",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )
        drafts_update[sub_q.id] = draft
        return {
            "drafts": drafts_update,
            "trace": [
                StepLog(
                    node="synthesizer",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="skipped",
                    details={"sub_question_id": sub_q.id, "reason": "no_evidence"},
                ),
            ],
        }

    try:
        draft = synthesize_one(
            sub_q,
            evidence_for_q,
            user_query=state.get("query", ""),
            output_language=state.get("output_language", "vi"),
            critic_feedback=state.get("synth_critic_feedback"),
            current_cost_usd=current_cost,
            per_query_cap_usd=state.get("per_query_cap_usd"),
        )
    except Exception as exc:
        logger.exception("Synthesizer failed for %s", sub_q.id)
        return {
            "trace": [
                StepLog(
                    node="synthesizer",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    status="error",
                    details={"sub_question_id": sub_q.id, "error": str(exc)},
                ),
            ],
        }

    drafts_update[sub_q.id] = draft
    update_span(
        input={
            "sub_question_id": sub_q.id,
            "sub_question": sub_q.question,
            "n_evidence": len(evidence_for_q),
        },
        output={
            "draft_chars": len(draft.content),
            "n_citations": len(draft.citations),
        },
        metadata={
            "model": draft.model,
            "cost_usd": round(draft.cost_usd, 6),
            "tokens_in": draft.tokens_in,
            "tokens_out": draft.tokens_out,
        },
    )
    return {
        "drafts": drafts_update,
        "total_cost_usd": current_cost + draft.cost_usd,
        "trace": [
            StepLog(
                node="synthesizer",
                duration_ms=(time.perf_counter() - started) * 1000,
                status=cast(Any, "ok"),
                details={
                    "sub_question_id": sub_q.id,
                    "model": draft.model,
                    "tokens_in": draft.tokens_in,
                    "tokens_out": draft.tokens_out,
                    "cost_usd": round(draft.cost_usd, 6),
                    "n_citations": len(draft.citations),
                    "n_evidence": len(evidence_for_q),
                },
            ),
        ],
    }
