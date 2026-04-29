"""Factuality eval: 15 EN + 5 VI queries with atomic gold claims.

Runs the research graph per query (unless ``--reports-json``), then Sonnet judge
labels each claim **supported** / **contradicted** / **unsupported** vs the final
report. Writes ``data/eval/factuality.json`` by default.

Usage::

    uv run python scripts/run_factuality_eval.py
    uv run python scripts/run_factuality_eval.py --eval-json data/eval/factuality_eval_20.json
    uv run python scripts/run_factuality_eval.py --reports-json path.json --limit 2
    uv run python scripts/run_factuality_eval.py --target-mean-supported 0.8 --strict-exit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

from research_assistant.config import get_settings
from research_assistant.eval.factuality import (
    build_eval_payload,
    judge_factuality,
    load_factuality_eval,
    load_factuality_reports_json,
    macro_mean_supported_ratio,
    per_query_supported_ratio,
)
from research_assistant.graph.research_graph import no_cross_encoder_rerank_fn, run_research

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_EVAL = _REPO / "data" / "eval" / "factuality_eval_20.json"
_DEFAULT_OUT = _REPO / "data" / "eval" / "factuality.json"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Factuality judge (supported/contradicted/unsupported) on gold claims.",
    )
    p.add_argument(
        "--eval-json",
        type=Path,
        default=_DEFAULT_EVAL,
        help="Eval set JSON (default: data/eval/factuality_eval_20.json).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="JSON output (default: data/eval/factuality.json).",
    )
    p.add_argument(
        "--reports-json",
        type=Path,
        default=None,
        help='Skip research; judge only. Shape: { "queries": [ { "query", "language", "gold_claims", "report" } ] }.',
    )
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows (debug).")
    p.add_argument(
        "--per-query-cap-usd",
        type=float,
        default=None,
        help="Budget cap per research+judge cycle (default: Settings.per_query_cap_usd).",
    )
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-critic", action="store_true")
    p.add_argument(
        "--target-mean-supported",
        type=float,
        default=None,
        help="If set with --strict-exit, exit 1 when mean_supported_ratio < this threshold.",
    )
    p.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit non-zero if any query failed or target not met.",
    )
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
        prepared = load_factuality_reports_json(args.reports_json)
        work: list[tuple[str, Literal["vi", "en"], list[str], str | None]] = [
            (it.query, it.language, it.gold_claims, it.report) for it in prepared
        ]
    else:
        if not args.eval_json.is_file():
            logger.error("eval-json not found: %s", args.eval_json)
            return 2
        if not settings.has_search_credentials:
            logger.error("TAVILY_API_KEY missing (required unless --reports-json)")
            return 2
        items = load_factuality_eval(args.eval_json)
        work = [(it.query, it.language, list(it.gold_claims), None) for it in items]

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

    for idx, (query, lang, gold_claims, pre_report) in enumerate(work, start=1):
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
                        "gold_claims": gold_claims,
                        "status": "research_error",
                        "judgments": None,
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
            judgments, jres = judge_factuality(
                query=query,
                output_language=lang,
                report_markdown=report,
                gold_claims=gold_claims,
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
                    "gold_claims": gold_claims,
                    "status": "judge_error",
                    "research_cost_usd": round(research_cost, 6) if pre_report is None else None,
                    "wallclock_research_s": round(wall_r, 2) if pre_report is None else None,
                    "langfuse_trace_id": trace_id,
                    "langfuse_trace_url": trace_url,
                    "max_iterations_reached": max_iter_hit if pre_report is None else None,
                    "judgments": None,
                },
            )
            continue
        wall_j = time.perf_counter() - t1
        wall_judge += wall_j
        total_judge_cost += jres.cost_usd

        supported_ratio = per_query_supported_ratio(judgments)
        ver_counts = {"supported": 0, "contradicted": 0, "unsupported": 0}
        for row in judgments:
            v = row.get("verdict")
            if v in ver_counts:
                ver_counts[str(v)] += 1

        row_out: dict[str, Any] = {
            "index": idx,
            "language": lang,
            "query": query,
            "gold_claims": gold_claims,
            "status": "ok",
            "supported_ratio": supported_ratio,
            "verdict_counts": ver_counts,
            "judgments": judgments,
            "judge_cost_usd": round(jres.cost_usd, 6),
            "judge_model": jres.model,
            "wallclock_judge_s": round(wall_j, 2),
        }
        if pre_report is None:
            row_out["research_cost_usd"] = round(research_cost, 6)
            row_out["wallclock_research_s"] = round(wall_r, 2)
            row_out["langfuse_trace_id"] = trace_id
            row_out["langfuse_trace_url"] = trace_url
            row_out["max_iterations_reached"] = max_iter_hit

        query_rows.append(row_out)
        logger.info(
            "  ↳ supported_ratio=%.3f counts=%s judge=$%.4f",
            supported_ratio,
            ver_counts,
            jres.cost_usd,
        )

    mean_sr = macro_mean_supported_ratio(query_rows)
    judge_model = settings.anthropic_planner_model
    eval_path_str = (
        str(args.eval_json.resolve())
        if args.reports_json is None
        else f"inline:{args.reports_json}"
    )
    payload = build_eval_payload(
        eval_path=eval_path_str,
        query_rows=query_rows,
        judge_model=judge_model,
        mean_supported_ratio=mean_sr,
        total_research_cost_usd=total_research_cost,
        total_judge_cost_usd=total_judge_cost,
        total_wallclock_research_s=wall_research,
        total_wallclock_judge_s=wall_judge,
    )
    if args.target_mean_supported is not None:
        payload["target_mean_supported_ratio"] = float(args.target_mean_supported)
        payload["meets_target"] = mean_sr >= float(args.target_mean_supported)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "wrote %s — ok=%d mean_supported_ratio=%.4f",
        args.out,
        sum(1 for r in query_rows if r.get("status") == "ok"),
        mean_sr,
    )

    all_ok = all(r.get("status") == "ok" for r in query_rows)
    if args.strict_exit and not all_ok:
        return 1
    if (
        args.strict_exit
        and args.target_mean_supported is not None
        and mean_sr < float(args.target_mean_supported)
    ):
        return 1
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
