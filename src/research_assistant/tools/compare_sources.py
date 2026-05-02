from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from research_assistant.agents._llm import invoke_structured_llm
from research_assistant.config import get_settings
from research_assistant.graph.state import (
    ConflictItem,
    ConflictReport,
    Draft,
    Evidence,
    SubQuestion,
)
from research_assistant.observability import update_span
from research_assistant.prompts.loader import render
from research_assistant.tools.router import classify_intent

logger = logging.getLogger(__name__)

CompareSourcesSetting = Literal["off", "heuristic", "auto"]

_QTY_RE = re.compile(
    r"\b(?P<num>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|percent|million|billion|trillion|thousand|"
    r"parameters?|params?|tokens?|layers?|"
    r"GB|MB|TB|"
    r"ms|mins?|minutes?|hours?|hrs?|"
    r"(?<![A-Za-z])[Kk](?![A-Za-z])|"
    r"(?<![A-Za-z])[Mm](?![A-Za-z])|"
    r"(?<![A-Za-z])[Bb](?![A-Za-z])"
    r")?",
    re.IGNORECASE,
)


def _parse_float(num: str) -> float:
    return float(num.replace(",", "."))


def _normalize_unit(raw: str | None) -> str:
    if not raw:
        return "_unitless"
    u = raw.lower()
    if u == "%" or u == "percent":
        return "pct"
    if u in ("k",):
        return "thousand"
    if u in ("m",):
        return "million"
    if u in ("b",):
        return "billion"
    if u in ("million", "billion", "trillion", "thousand"):
        return u
    if u in ("parameter", "parameters", "param", "params"):
        return "parameters"
    if u in ("token", "tokens"):
        return "tokens"
    if u in ("layer", "layers"):
        return "layers"
    if u in ("gb", "mb", "tb"):
        return u.upper()
    if u in ("ms", "min", "mins", "minute", "minutes", "hr", "hrs", "hour", "hours"):
        return u
    return u


def _evidence_body(ev: Evidence) -> str:
    parts = [ev.hit.snippet or ""]
    if ev.hit.raw_content:
        parts.append(ev.hit.raw_content[:4000])
    return "\n".join(p for p in parts if p).strip()


def extract_quantities(text: str) -> list[tuple[float, str, str]]:
    out: list[tuple[float, str, str]] = []
    for m in _QTY_RE.finditer(text):
        num_s = m.group("num")
        unit_raw = m.group("unit")
        try:
            val = _parse_float(num_s)
        except ValueError:
            continue
        if not math.isfinite(val):
            continue
        unit_key = _normalize_unit(unit_raw)
        start, end = m.span()
        ctx = text[max(0, start - 48) : min(len(text), end + 48)].replace("\n", " ")
        out.append((val, unit_key, ctx.strip()))
    return out


def _numeric_mismatch(val_a: float, val_b: float, unit_key: str) -> bool:
    if unit_key == "pct":
        return abs(val_a - val_b) > 2.0 + 1e-9
    if unit_key in ("million", "billion", "trillion", "thousand"):
        base = max(abs(val_a), abs(val_b), 1e-9)
        return abs(val_a - val_b) / base > 0.05 + 1e-9
    if unit_key == "_unitless":
        top = max(abs(val_a), abs(val_b))
        if top >= 100:
            base = max(top, 1e-9)
            return abs(val_a - val_b) / base > 0.15 + 1e-9
        return abs(val_a - val_b) >= 1.0 + 1e-9
    base = max(abs(val_a), abs(val_b), 1e-9)
    return abs(val_a - val_b) / base > 0.10 + 1e-9


def _severity_for(val_a: float, val_b: float, unit_key: str) -> Literal["low", "medium", "high"]:
    rel = abs(val_a - val_b) / max(abs(val_a), abs(val_b), 1e-9)
    if unit_key == "pct":
        if abs(val_a - val_b) >= 10:
            return "high"
        if abs(val_a - val_b) >= 5:
            return "medium"
        return "low"
    if rel >= 0.5:
        return "high"
    if rel >= 0.2:
        return "medium"
    return "low"


def heuristic_compare_sources(evidence: Sequence[Evidence]) -> list[ConflictItem]:
    by_label: dict[str, list[tuple[float, str, str]]] = {}
    urls: dict[str, str] = {}
    for ev in evidence:
        body = _evidence_body(ev)
        if not body:
            continue
        by_label[ev.ref_label] = extract_quantities(body)
        urls[ev.ref_label] = str(ev.hit.url)

    seen_pairs: set[frozenset[str]] = set()
    items: list[ConflictItem] = []

    labels = list(by_label.keys())
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            if urls.get(la) == urls.get(lb):
                continue
            for va, ua, ca in by_label[la]:
                for vb, ub, cb in by_label[lb]:
                    if ua != ub:
                        continue
                    if not _numeric_mismatch(va, vb, ua):
                        continue
                    key = frozenset({la, lb})
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    sev = _severity_for(va, vb, ua)
                    summary = (
                        f"Numeric mismatch ({ua}) between sources: "
                        f"{va:g} vs {vb:g} (refs {la}, {lb})"
                    )
                    detail = f"A≈{ca[:120]} | B≈{cb[:120]}"
                    items.append(
                        ConflictItem(
                            summary=summary,
                            severity=sev,
                            involved_ref_labels=sorted({la, lb}),
                            detection="heuristic",
                            detail=detail,
                        ),
                    )
    return items


class _ConflictItemDraft(BaseModel):
    summary: str = Field(..., min_length=3)
    severity: Literal["low", "medium", "high"] = Field(...)
    involved_ref_labels: list[str] = Field(default_factory=list)
    detail: str = ""


class _CompareSourcesDraft(BaseModel):
    conflicts: list[_ConflictItemDraft] = Field(
        default_factory=list,
        description="Validated cross-source conflicts (may confirm or drop heuristics).",
    )


def _should_run_llm(
    *,
    mode: CompareSourcesSetting,
    heuristic_items: Sequence[ConflictItem],
    sub_q: SubQuestion,
) -> bool:
    if mode != "auto":
        return False
    if heuristic_items:
        return True
    intent = classify_intent(sub_q.question, sub_q.rationale)
    return intent == "comparative"


def _llm_refine_conflicts(
    *,
    sub_q: SubQuestion,
    draft: Draft,
    evidence: Sequence[Evidence],
    heuristic_items: Sequence[ConflictItem],
    cost_before: float,
    cap: float | None,
) -> tuple[list[ConflictItem], float, str, int, int]:
    system = render("compare_sources_system_v1.jinja")
    user = render(
        "compare_sources_user_v1.jinja",
        sub_question=sub_q,
        draft=draft,
        evidence=evidence,
        heuristic_items=heuristic_items,
    )
    settings = get_settings()
    parsed, result = invoke_structured_llm(
        model=settings.anthropic_planner_model,
        prompt=user,
        system=system,
        schema=_CompareSourcesDraft,
        temperature=0.0,
        max_tokens=1024,
        current_cost_usd=cost_before,
        per_query_cap_usd=cap,
    )
    items: list[ConflictItem] = []
    for row in parsed.conflicts:
        items.append(
            ConflictItem(
                summary=row.summary,
                severity=row.severity,
                involved_ref_labels=list(row.involved_ref_labels),
                detection="llm",
                detail=row.detail,
            ),
        )
    return items, result.cost_usd, result.model, result.tokens_in, result.tokens_out


def build_conflict_report(
    *,
    sub_q: SubQuestion,
    evidence: Sequence[Evidence],
    draft: Draft,
    mode: CompareSourcesSetting | None = None,
    cost_before: float = 0.0,
    per_query_cap_usd: float | None = None,
) -> tuple[ConflictReport, float]:

    settings = get_settings()
    resolved = mode if mode is not None else settings.compare_sources_mode
    cap = per_query_cap_usd if per_query_cap_usd is not None else None

    if resolved == "off":
        return (
            ConflictReport(sub_question_id=sub_q.id, mode_used="off", items=[]),
            0.0,
        )

    heur = heuristic_compare_sources(evidence)
    cost_delta = 0.0

    if resolved == "heuristic":
        report = ConflictReport(sub_question_id=sub_q.id, mode_used="heuristic", items=list(heur))
        update_span(
            input={"sub_question_id": sub_q.id, "mode": resolved},
            output={"n_heuristic": len(report.items)},
        )
        return report, cost_delta

    assert resolved == "auto"
    if not _should_run_llm(mode="auto", heuristic_items=heur, sub_q=sub_q):
        report = ConflictReport(sub_question_id=sub_q.id, mode_used="heuristic", items=list(heur))
        update_span(
            input={"sub_question_id": sub_q.id, "mode": resolved, "llm": False},
            output={"n_heuristic": len(heur)},
        )
        return report, cost_delta

    try:
        llm_items, extra, _model, _tin, _tout = _llm_refine_conflicts(
            sub_q=sub_q,
            draft=draft,
            evidence=evidence,
            heuristic_items=heur,
            cost_before=cost_before,
            cap=cap,
        )
        cost_delta += extra
        merged = llm_items or list(heur)
        auto_mode: Literal["heuristic+llm", "llm"] = "heuristic+llm" if heur else "llm"
        report = ConflictReport(sub_question_id=sub_q.id, mode_used=auto_mode, items=merged)
        update_span(
            input={"sub_question_id": sub_q.id, "mode": resolved, "llm": True},
            output={"n_items": len(merged), "n_heuristic": len(heur)},
        )
        return report, cost_delta
    except Exception:
        logger.exception("compare_sources LLM failed — falling back to heuristic only")
        report = ConflictReport(
            sub_question_id=sub_q.id,
            mode_used="heuristic",
            items=list(heur),
        )
        update_span(
            input={"sub_question_id": sub_q.id, "mode": resolved, "llm": True},
            output={"error": True, "n_heuristic": len(heur)},
        )
        return report, cost_delta
