from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, TypeAdapter

from research_assistant.agents._llm import LLMCallResult, invoke_structured_llm
from research_assistant.config import get_settings
from research_assistant.prompts.loader import render

FactualityVerdict = Literal["supported", "contradicted", "unsupported"]

_MIN_CLAIMS = 3
_MAX_CLAIMS = 5
_MAX_REPORT_CHARS = 120_000


class FactualityEvalItem(BaseModel):
    id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=8)
    language: Literal["en", "vi"] = "en"
    gold_claims: list[str] = Field(..., min_length=_MIN_CLAIMS, max_length=_MAX_CLAIMS)


class _FactualityEvalFile(BaseModel):
    version: int = 1
    description: str = ""
    items: list[FactualityEvalItem]


class FactualityClaimJudgment(BaseModel):
    claim_index: int = Field(..., ge=0)
    verdict: FactualityVerdict
    rationale_brief: str = Field(
        default="",
        description="One short English sentence explaining the verdict.",
    )


class FactualityJudgeOutput(BaseModel):
    judgments: list[FactualityClaimJudgment]


def clip_report_for_judge(report_markdown: str, *, max_chars: int = _MAX_REPORT_CHARS) -> str:
    if len(report_markdown) <= max_chars:
        return report_markdown
    return report_markdown[:max_chars] + "\n\n[... report truncated for judge context ...]\n"


def load_factuality_eval(path: Path) -> list[FactualityEvalItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bundle = _FactualityEvalFile.model_validate(raw)
    for it in bundle.items:
        for i, c in enumerate(it.gold_claims):
            t = c.strip()
            if len(t) < 8:
                raise ValueError(f"{it.id}: gold_claims[{i}] too short")
    return bundle.items


def normalize_judgments(
    gold_claims: list[str],
    parsed: FactualityJudgeOutput,
) -> list[dict[str, Any]]:
    n = len(gold_claims)
    by_idx: dict[int, FactualityClaimJudgment] = {}
    for j in parsed.judgments:
        if 0 <= j.claim_index < n:
            by_idx[j.claim_index] = j
    out: list[dict[str, Any]] = []
    for i in range(n):
        if i in by_idx:
            row = by_idx[i]
            out.append(
                {
                    "claim_index": i,
                    "claim": gold_claims[i],
                    "verdict": row.verdict,
                    "rationale_brief": row.rationale_brief.strip(),
                },
            )
        else:
            out.append(
                {
                    "claim_index": i,
                    "claim": gold_claims[i],
                    "verdict": "unsupported",
                    "rationale_brief": "missing_judgment",
                },
            )
    return out


def per_query_supported_ratio(judgment_rows: list[dict[str, Any]]) -> float:
    if not judgment_rows:
        return 0.0
    ok = sum(1 for r in judgment_rows if r.get("verdict") == "supported")
    return round(ok / len(judgment_rows), 6)


def macro_mean_supported_ratio(query_results: list[dict[str, Any]]) -> float:
    ratios: list[float] = []
    for row in query_results:
        if row.get("status") != "ok":
            continue
        judgments = row.get("judgments")
        if not isinstance(judgments, list):
            continue
        ratios.append(per_query_supported_ratio(cast(list[dict[str, Any]], judgments)))
    if not ratios:
        return 0.0
    return round(sum(ratios) / len(ratios), 6)


def judge_factuality(
    *,
    query: str,
    output_language: Literal["vi", "en"],
    report_markdown: str,
    gold_claims: list[str],
    model: str | None = None,
    current_cost_usd: float = 0.0,
    per_query_cap_usd: float | None = None,
    use_prompt_cache: bool | None = False,
) -> tuple[list[dict[str, Any]], LLMCallResult]:

    settings = get_settings()
    m = model or settings.anthropic_planner_model
    claims_block = "\n".join(f"{i}. {c}" for i, c in enumerate(gold_claims))
    system = render("factuality_judge_system_v1.jinja")
    user = render(
        "factuality_judge_user_v1.jinja",
        query=query,
        output_language=output_language,
        gold_claims_block=claims_block,
        report_markdown=clip_report_for_judge(report_markdown),
    )
    parsed, result = invoke_structured_llm(
        model=m,
        prompt=user,
        system=system,
        schema=FactualityJudgeOutput,
        temperature=0.0,
        max_tokens=4096,
        current_cost_usd=current_cost_usd,
        per_query_cap_usd=per_query_cap_usd,
        use_prompt_cache=use_prompt_cache,
    )
    return normalize_judgments(gold_claims, parsed), result


def validate_eval_counts(items: list[FactualityEvalItem]) -> None:
    if len(items) != 20:
        raise ValueError(f"expected 20 items, got {len(items)}")
    n_en = sum(1 for x in items if x.language == "en")
    n_vi = sum(1 for x in items if x.language == "vi")
    if n_en != 15 or n_vi != 5:
        raise ValueError(f"expected 15 EN + 5 VI, got {n_en} EN + {n_vi} VI")


@dataclass(frozen=True)
class PreparedFactualityReportItem:
    query: str
    language: Literal["vi", "en"]
    gold_claims: list[str]
    report: str


def load_factuality_reports_json(path: Path) -> list[PreparedFactualityReportItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arr = raw.get("queries")
    if not isinstance(arr, list):
        raise ValueError("reports JSON must contain a 'queries' array")
    adapter = TypeAdapter(list[str])
    out: list[PreparedFactualityReportItem] = []
    for i, it in enumerate(arr):
        if not isinstance(it, dict):
            raise ValueError(f"queries[{i}] must be an object")
        q = it.get("query")
        lang = it.get("language")
        rep = it.get("report")
        gc = it.get("gold_claims")
        if not isinstance(q, str) or not isinstance(rep, str):
            raise ValueError(f"queries[{i}] needs string 'query' and 'report'")
        if lang not in ("vi", "en"):
            raise ValueError(f"queries[{i}].language must be 'vi' or 'en'")
        claims = adapter.validate_python(gc)
        if not _MIN_CLAIMS <= len(claims) <= _MAX_CLAIMS:
            raise ValueError(
                f"queries[{i}].gold_claims must have {_MIN_CLAIMS}..{_MAX_CLAIMS} strings",
            )
        out.append(
            PreparedFactualityReportItem(
                query=q,
                language=cast(Literal["vi", "en"], lang),
                gold_claims=claims,
                report=rep,
            ),
        )
    return out


def build_eval_payload(
    *,
    eval_path: str,
    query_rows: list[dict[str, Any]],
    judge_model: str,
    mean_supported_ratio: float,
    total_research_cost_usd: float,
    total_judge_cost_usd: float,
    total_wallclock_research_s: float,
    total_wallclock_judge_s: float,
) -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "eval_file": eval_path,
        "judge_model": judge_model,
        "mean_supported_ratio": mean_supported_ratio,
        "n_queries": len(query_rows),
        "total_research_cost_usd": round(total_research_cost_usd, 6),
        "total_judge_cost_usd": round(total_judge_cost_usd, 6),
        "total_cost_usd": round(total_research_cost_usd + total_judge_cost_usd, 6),
        "total_wallclock_research_s": round(total_wallclock_research_s, 2),
        "total_wallclock_judge_s": round(total_wallclock_judge_s, 2),
        "queries": query_rows,
    }
