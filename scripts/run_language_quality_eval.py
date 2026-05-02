from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

from research_assistant.config import get_settings
from research_assistant.eval.language_quality import (
    AXES,
    DEFAULT_QUERIES_EN,
    DEFAULT_QUERIES_VI,
    build_eval_payload,
    compare_to_baseline,
    judge_language_quality,
    load_reports_json,
    mean_over_axes,
)
from research_assistant.graph.research_graph import no_cross_encoder_rerank_fn, run_research

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "data" / "eval" / "language_quality.json"


def _seed_queries() -> list[tuple[str, Literal["vi", "en"]]]:
    rows: list[tuple[str, Literal["vi", "en"]]] = []
    for q in DEFAULT_QUERIES_VI:
        rows.append((q, "vi"))
    for q in DEFAULT_QUERIES_EN:
        rows.append((q, "en"))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="Language quality rubric (4 axes) over 5 VI + 5 EN queries.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="JSON output (default: data/eval/language_quality.json).",
    )
    p.add_argument(
        "--reports-json",
        type=Path,
        default=None,
        help='Skip research; judge only. File shape: { "queries": [ { "query", "language", "report" } ] }.',
    )
    p.add_argument(
        "--compare-previous",
        type=Path,
        default=None,
        help="Optional prior language_quality.json; adds baseline_comparison deltas to output.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N queries (debug).",
    )
    p.add_argument(
        "--per-query-cap-usd",
        type=float,
        default=None,
        help="Budget cap per research+judge cycle (default: Settings.per_query_cap_usd).",
    )
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-critic", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    if not settings.has_llm_credentials:
        logger.error("ANTHROPIC_API_KEY missing")
        return 2
    cap = (
        args.per_query_cap_usd if args.per_query_cap_usd is not None else settings.per_query_cap_usd
    )

    if args.reports_json is not None:
        if not args.reports_json.is_file():
            logger.error("reports-json not found: %s", args.reports_json)
            return 2
        prepared = load_reports_json(args.reports_json)
        work: list[tuple[str, Literal["vi", "en"], str | None]] = [
            (it.query, it.language, it.report) for it in prepared
        ]
    else:
        if not settings.has_search_credentials:
            logger.error("TAVILY_API_KEY missing (required unless --reports-json)")
            return 2
        work = [(q, lang, None) for q, lang in _seed_queries()]

    if args.limit is not None:
        work = work[: max(0, args.limit)]

    rerank_fn = no_cross_encoder_rerank_fn() if args.no_rerank else None
    critic_override = False if args.no_critic else None
    max_iters = args.max_iterations if args.max_iterations is not None else settings.max_iterations

    query_rows: list[dict[str, Any]] = []
    total_research_cost = 0.0
    total_judge_cost = 0.0
    wall_research = 0.0
    wall_judge = 0.0

    for idx, (query, lang, pre_report) in enumerate(work, start=1):
        logger.info("[%d/%d] %s — %s", idx, len(work), lang, query[:72])
        report: str
        research_cost = 0.0
        trace_id = None
        trace_url = None
        max_iter_hit = False
        wall_r = 0.0

        if pre_report is not None:
            report = pre_report
        else:
            t0 = time.perf_counter()
            try:
                final = run_research(
                    query=query,
                    output_language=lang,
                    max_iterations=max_iters,
                    per_query_cap_usd=cap,
                    rerank_fn=rerank_fn,
                    critic_enabled_override=critic_override,
                )
            except Exception:
                logger.exception("research failed for query %d", idx)
                query_rows.append(
                    {
                        "index": idx,
                        "language": lang,
                        "query": query,
                        "status": "research_error",
                        "scores": None,
                    },
                )
                continue
            wall_r = time.perf_counter() - t0
            wall_research += wall_r
            research_cost = float(final.get("total_cost_usd", 0.0))
            total_research_cost += research_cost
            report = str(final.get("final_report") or "")
            trace_id = final.get("trace_id")
            trace_url = final.get("trace_url")
            max_iter_hit = bool(final.get("max_iterations_reached", False))

        t1 = time.perf_counter()
        try:
            scores, jres = judge_language_quality(
                query=query,
                output_language=lang,
                report_markdown=report,
                current_cost_usd=research_cost,
                per_query_cap_usd=cap,
            )
        except Exception:
            logger.exception("judge failed for query %d", idx)
            query_rows.append(
                {
                    "index": idx,
                    "language": lang,
                    "query": query,
                    "status": "judge_error",
                    "research_cost_usd": round(research_cost, 6) if pre_report is None else None,
                    "wallclock_research_s": round(wall_r, 2) if pre_report is None else None,
                    "langfuse_trace_id": trace_id,
                    "langfuse_trace_url": trace_url,
                    "max_iterations_reached": max_iter_hit if pre_report is None else None,
                    "scores": None,
                },
            )
            continue
        wall_j = time.perf_counter() - t1
        wall_judge += wall_j

        total_judge_cost += jres.cost_usd
        row: dict[str, Any] = {
            "index": idx,
            "language": lang,
            "query": query,
            "status": "ok",
            "scores": {
                "accuracy": scores.accuracy,
                "fluency": scores.fluency,
                "terminology": scores.terminology,
                "citation": scores.citation,
            },
            "rationale_brief": scores.rationale_brief,
            "judge_cost_usd": round(jres.cost_usd, 6),
            "judge_model": jres.model,
            "wallclock_judge_s": round(wall_j, 2),
        }
        if pre_report is None:
            row["research_cost_usd"] = round(research_cost, 6)
            row["wallclock_research_s"] = round(wall_r, 2)
            row["langfuse_trace_id"] = trace_id
            row["langfuse_trace_url"] = trace_url
            row["max_iterations_reached"] = max_iter_hit
        query_rows.append(row)
        logger.info(
            "  ↳ scores %s judge=$%.4f",
            {a: row["scores"][a] for a in AXES},
            jres.cost_usd,
        )

    mean_axes = mean_over_axes([r for r in query_rows if r.get("status") == "ok"])
    judge_model = settings.anthropic_planner_model

    payload = build_eval_payload(
        query_rows=query_rows,
        judge_model=judge_model,
        mean_by_axis=mean_axes,
        total_research_cost_usd=total_research_cost,
        total_judge_cost_usd=total_judge_cost,
        total_wallclock_research_s=wall_research,
        total_wallclock_judge_s=wall_judge,
        baseline_comparison=None,
    )

    if args.compare_previous is not None:
        if not args.compare_previous.is_file():
            logger.error("compare-previous not found: %s", args.compare_previous)
            return 2
        payload["baseline_comparison"] = compare_to_baseline(payload, args.compare_previous)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "wrote %s — ok=%d mean_overall=%.4f mean_axes=%s",
        args.out,
        sum(1 for r in query_rows if r.get("status") == "ok"),
        payload["mean_overall"],
        mean_axes,
    )
    return 0 if all(r.get("status") == "ok" for r in query_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
