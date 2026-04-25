"""LLM-as-judge rubric for VI/EN report quality (Tuần 3).

Four axes: accuracy, fluency, terminology, citation - each scored 1-5.
Used by :mod:`scripts.run_language_quality_eval`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from research_assistant.agents._llm import LLMCallResult, invoke_structured_llm
from research_assistant.config import get_settings
from research_assistant.prompts.loader import render

AXES: tuple[str, ...] = ("accuracy", "fluency", "terminology", "citation")

_MAX_REPORT_CHARS = 120_000

DEFAULT_QUERIES_VI: tuple[str, ...] = (
    "So sánh LoRA và QLoRA cho fine-tuning LLM năm 2026",
    "Retrieval-Augmented Generation là gì, khi nào nên dùng thay vì fine-tuning?",
    "GraphRAG khác gì so với RAG vector truyền thống trong các hệ Q&A đa bước?",
    "Các kỹ thuật giảm chi phí inference cho LLM (quantization, distillation) năm 2025-2026?",
    "Tác nhân AI (agent) sử dụng LangGraph: ưu điểm và hạn chế so với chuỗi prompt đơn?",
)

DEFAULT_QUERIES_EN: tuple[str, ...] = (
    "What are the latest advances in reasoning models like OpenAI o3 and DeepSeek R1 in 2026?",
    "Compare vector databases: Qdrant vs Weaviate vs Milvus for production RAG",
    "What is Agentic RAG and how does it differ from traditional RAG?",
    "How does Anthropic's contextual retrieval approach improve RAG pipelines?",
    "What are design patterns for tool use in LLM agents (ReAct, planner-executor)?",
)


class LanguageQualityScores(BaseModel):
    """Structured verdict from the judge model."""

    accuracy: int = Field(..., ge=1, le=5)
    fluency: int = Field(..., ge=1, le=5)
    terminology: int = Field(..., ge=1, le=5)
    citation: int = Field(..., ge=1, le=5)
    rationale_brief: str = Field(
        default="",
        description="Short English summary of why these scores were assigned.",
    )


def clip_report_for_judge(report_markdown: str, *, max_chars: int = _MAX_REPORT_CHARS) -> str:
    """Bound context size for the judge call."""
    if len(report_markdown) <= max_chars:
        return report_markdown
    return report_markdown[:max_chars] + "\n\n[... report truncated for judge context ...]\n"


def judge_language_quality(
    *,
    query: str,
    output_language: Literal["vi", "en"],
    report_markdown: str,
    model: str | None = None,
    current_cost_usd: float = 0.0,
    per_query_cap_usd: float | None = None,
    use_prompt_cache: bool | None = False,
) -> tuple[LanguageQualityScores, LLMCallResult]:
    """Score one report on the four rubric axes (Sonnet structured output)."""
    settings = get_settings()
    m = model or settings.anthropic_planner_model
    system = render(
        "language_quality_judge_system_v1.jinja",
        output_language=output_language,
    )
    user = render(
        "language_quality_judge_user_v1.jinja",
        query=query,
        output_language=output_language,
        report_markdown=clip_report_for_judge(report_markdown),
    )
    return invoke_structured_llm(
        model=m,
        prompt=user,
        system=system,
        schema=LanguageQualityScores,
        temperature=0.0,
        max_tokens=1024,
        current_cost_usd=current_cost_usd,
        per_query_cap_usd=per_query_cap_usd,
        use_prompt_cache=use_prompt_cache,
    )


def mean_over_axes(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Macro mean per axis from query result dicts with ``scores`` nested dict."""
    if not rows:
        return dict.fromkeys(AXES, 0.0)
    sums = dict.fromkeys(AXES, 0.0)
    n = 0
    for row in rows:
        sc = row.get("scores")
        if not isinstance(sc, dict):
            continue
        n += 1
        for a in AXES:
            v = sc.get(a)
            if isinstance(v, (int, float)):
                sums[a] += float(v)
    if n == 0:
        return dict.fromkeys(AXES, 0.0)
    return {a: round(sums[a] / n, 4) for a in AXES}


def overall_mean(mean_by_axis: dict[str, float]) -> float:
    if not mean_by_axis:
        return 0.0
    return round(sum(mean_by_axis.values()) / len(mean_by_axis), 4)


@dataclass(frozen=True)
class PreparedReportItem:
    query: str
    language: Literal["vi", "en"]
    report: str


def load_reports_json(path: Path) -> list[PreparedReportItem]:
    """Load ``--reports-json`` file: ``{ "queries": [ { "query", "language", "report" } ] }``."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("queries")
    if not isinstance(items, list):
        raise ValueError("reports JSON must contain a 'queries' array")
    out: list[PreparedReportItem] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"queries[{i}] must be an object")
        q = it.get("query")
        lang = it.get("language")
        rep = it.get("report")
        if not isinstance(q, str) or not isinstance(rep, str):
            raise ValueError(f"queries[{i}] needs string 'query' and 'report'")
        if lang not in ("vi", "en"):
            raise ValueError(f"queries[{i}].language must be 'vi' or 'en'")
        out.append(
            PreparedReportItem(
                query=q,
                language=cast(Literal["vi", "en"], lang),
                report=rep,
            ),
        )
    return out


def build_eval_payload(
    *,
    query_rows: list[dict[str, Any]],
    judge_model: str,
    mean_by_axis: dict[str, float],
    total_research_cost_usd: float,
    total_judge_cost_usd: float,
    total_wallclock_research_s: float,
    total_wallclock_judge_s: float,
    baseline_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top-level JSON shape written by the script."""
    return {
        "version": 1,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "axes": list(AXES),
        "judge_model": judge_model,
        "mean_by_axis": mean_by_axis,
        "mean_overall": overall_mean(mean_by_axis),
        "n_queries": len(query_rows),
        "total_research_cost_usd": round(total_research_cost_usd, 6),
        "total_judge_cost_usd": round(total_judge_cost_usd, 6),
        "total_cost_usd": round(total_research_cost_usd + total_judge_cost_usd, 6),
        "total_wallclock_research_s": round(total_wallclock_research_s, 2),
        "total_wallclock_judge_s": round(total_wallclock_judge_s, 2),
        "queries": query_rows,
        **({"baseline_comparison": baseline_comparison} if baseline_comparison else {}),
    }


def compare_to_baseline(
    current: dict[str, Any],
    previous_path: Path,
) -> dict[str, Any]:
    """Compute simple deltas vs a prior ``language_quality.json`` run."""
    prev = json.loads(previous_path.read_text(encoding="utf-8"))
    p_mean = prev.get("mean_by_axis")
    c_mean = current.get("mean_by_axis")
    if not isinstance(p_mean, dict) or not isinstance(c_mean, dict):
        raise ValueError("baseline file missing mean_by_axis")
    delta_axes = {a: round(float(c_mean.get(a, 0)) - float(p_mean.get(a, 0)), 4) for a in AXES}
    p_o = float(prev.get("mean_overall", 0))
    c_o = float(current.get("mean_overall", 0))
    return {
        "previous_path": str(previous_path),
        "previous_created_at": prev.get("created_at"),
        "mean_overall_delta": round(c_o - p_o, 4),
        "mean_by_axis_delta": delta_axes,
    }
