"""Week-1 end-to-end smoke test across the 5 seed queries.

Not a pytest test (it calls real external APIs and costs money). Run::

    uv run python scripts/week1_smoke.py

    # A/B: base (no rerank, no critic) then tuned (settings defaults); cost delta
    uv run python scripts/week1_smoke.py --ab

    # Single pass with CLI flags (same as ``research-assistant``)
    uv run python scripts/week1_smoke.py --no-rerank --no-critic --max-iterations 16

    # Force tool-router + compare_sources (overrides env for this run); JSON adds
    # ``router_plan_per_subq`` / ``n_conflicts`` per query
    uv run python scripts/week1_smoke.py --with-router --with-compare-sources

Outputs:
    * data/eval/week1_outputs.md   — concatenated Markdown reports.
    * data/eval/week1_metrics.json — per-query cost / timing / citations /
      ``max_iterations_reached`` / Langfuse ids / retrieval stats;
      ``router_plan_per_subq`` / ``n_conflicts`` (Tuần 4 smoke); optional
      ``ab_compare`` when ``--ab``; optional ``delta_vs_previous_file``.
    * Prints a compact summary table to stdout at the end.
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
from research_assistant.graph.research_graph import no_cross_encoder_rerank_fn, run_research

QUERIES: list[tuple[str, Literal["vi", "en"]]] = [
    ("So sánh LoRA và QLoRA cho fine-tuning LLM năm 2026", "vi"),
    ("Retrieval-Augmented Generation là gì, khi nào nên dùng thay vì fine-tuning?", "vi"),
    (
        "What are the latest advances in reasoning models like OpenAI o3 and DeepSeek R1 in 2026?",
        "en",
    ),
    ("Compare vector databases: Qdrant vs Weaviate vs Milvus for production RAG", "en"),
    ("What is Agentic RAG and how does it differ from traditional RAG?", "en"),
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "data" / "eval"
_REPORT_FILE = _OUT_DIR / "week1_outputs.md"
_METRICS_FILE = _OUT_DIR / "week1_metrics.json"

logger = logging.getLogger("week1_smoke")


def _step_attr(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _aggregate_retrieval_stats(final: Any) -> dict[str, Any]:
    """Corpus vs web hit counts from evidence + per-sub-q retriever StepLog details."""
    evidence = final.get("evidence") or {}
    by_source: dict[str, int] = {}
    for _sq, evs in evidence.items():
        for ev in evs:
            hit = _step_attr(ev, "hit")
            src = _step_attr(hit, "source", "web")
            by_source[src] = by_source.get(src, 0) + 1

    retriever_logs: list[dict[str, Any]] = []
    for step in final.get("trace") or []:
        if _step_attr(step, "node") != "retriever":
            continue
        det = _step_attr(step, "details") or {}
        if not isinstance(det, dict) or not det.get("sub_question_id"):
            continue
        entry = {
            k: det[k]
            for k in (
                "sub_question_id",
                "n_corpus",
                "n_web",
                "n_pool",
                "n_after_rerank",
                "retrieval_path",
            )
            if k in det
        }
        if entry:
            retriever_logs.append(entry)

    return {
        "evidence_hits_by_source": by_source,
        "retriever_steps": len(retriever_logs),
        "retriever_details": retriever_logs,
    }


def _router_plan_per_subq(final: Any) -> list[dict[str, Any]]:
    """Last retriever-step router snapshot per sub-question, in plan order."""
    plan = final.get("plan") or []
    id_order = [sq.id for sq in plan]
    last_by_sq: dict[str, dict[str, Any]] = {}
    for step in final.get("trace") or []:
        if _step_attr(step, "node") != "retriever":
            continue
        det = _step_attr(step, "details") or {}
        if not isinstance(det, dict):
            continue
        sq = det.get("sub_question_id")
        if not isinstance(sq, str):
            continue
        ord_raw = det.get("router_ordered_tools")
        if isinstance(ord_raw, str):
            ord_list = [x.strip() for x in ord_raw.split(",") if x.strip()]
        elif isinstance(ord_raw, list):
            ord_list = [str(x) for x in ord_raw]
        else:
            ord_list = []
        last_by_sq[sq] = {
            "sub_question_id": sq,
            "router_intent": det.get("router_intent"),
            "router_tools": det.get("router_tools"),
            "router_ordered_tools": ord_list,
            "router_overrode_planner": det.get("router_overrode_planner"),
            "planner_suggested_tools": det.get("planner_suggested_tools"),
        }
    return [last_by_sq[i] for i in id_order if i in last_by_sq]


def _n_conflicts(final: Any) -> int:
    """Total conflict items across sub-questions (compare_sources)."""
    reports = final.get("conflict_reports") or {}
    n = 0
    for rep in reports.values():
        items = _step_attr(rep, "items")
        if items is None and isinstance(rep, dict):
            items = rep.get("items")
        if isinstance(items, list):
            n += len(items)
    return n


def _run_smoke_pass(
    *,
    label: str,
    no_rerank: bool,
    no_critic: bool,
    max_iterations: int,
    per_query_cap_usd: float,
    tool_router_enabled_override: bool | None,
    compare_sources_mode_override: Literal["off", "heuristic", "auto"] | None,
) -> tuple[list[str], list[dict[str, Any]], float, float, int]:
    """Execute all seed queries; return report bodies, metrics rows, totals, hit-cap count."""
    all_reports: list[str] = []
    metrics: list[dict[str, Any]] = []
    total_cost = 0.0
    total_wallclock = 0.0
    n_cap = 0

    rerank_fn = no_cross_encoder_rerank_fn() if no_rerank else None
    critic_override = False if no_critic else None

    for idx, (query, lang) in enumerate(QUERIES, start=1):
        logger.info(
            "[%s %d/%d] Running: %r (lang=%s)",
            label,
            idx,
            len(QUERIES),
            query,
            lang,
        )
        start = time.perf_counter()

        try:
            final: Any = run_research(
                query=query,
                output_language=lang,
                max_iterations=max_iterations,
                per_query_cap_usd=per_query_cap_usd,
                rerank_fn=rerank_fn,
                critic_enabled_override=critic_override,
                tool_router_enabled_override=tool_router_enabled_override,
                compare_sources_mode_override=compare_sources_mode_override,
            )
        except Exception:
            logger.exception("Query %d failed; recording and continuing.", idx)
            metrics.append(
                {
                    "index": idx,
                    "query": query,
                    "language": lang,
                    "status": "error",
                    "cost_usd": 0.0,
                    "wallclock_s": round(time.perf_counter() - start, 2),
                    "max_iterations_reached": False,
                    "router_plan_per_subq": [],
                    "n_conflicts": 0,
                },
            )
            all_reports.append(f"# Query {idx}: {query}\n\n_(Run failed — see logs)_\n")
            continue

        wallclock = time.perf_counter() - start
        cost = float(final.get("total_cost_usd", 0.0))
        total_cost += cost
        total_wallclock += wallclock
        mir = bool(final.get("max_iterations_reached", False))
        if mir:
            n_cap += 1

        plan = final.get("plan", [])
        drafts = final.get("drafts", {})
        evidence_counts = {k: len(v) for k, v in final.get("evidence", {}).items()}
        total_citations = sum(len(d.citations) for d in drafts.values())

        retrieval = _aggregate_retrieval_stats(final)
        metrics.append(
            {
                "index": idx,
                "query": query,
                "language": lang,
                "status": "ok",
                "cost_usd": round(cost, 6),
                "wallclock_s": round(wallclock, 2),
                "n_sub_questions": len(plan),
                "n_drafts": len(drafts),
                "evidence_counts": evidence_counts,
                "total_citations": total_citations,
                "trace_steps": len(final.get("trace", [])),
                "max_iterations_reached": mir,
                "langfuse_trace_id": final.get("trace_id"),
                "langfuse_trace_url": final.get("trace_url"),
                "router_plan_per_subq": _router_plan_per_subq(final),
                "n_conflicts": _n_conflicts(final),
                **retrieval,
            },
        )

        report = final.get("final_report") or "(empty report)"
        all_reports.append(f"\n\n---\n\n# Query {idx} ({lang.upper()}): {query}\n\n{report}")

        logger.info(
            "  ↳ done in %.1fs, cost=$%.4f, %d sub-qs, %d cites, max_iter_reached=%s",
            wallclock,
            cost,
            len(plan),
            total_citations,
            mir,
        )

    return all_reports, metrics, total_cost, total_wallclock, n_cap


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week-1 five-query smoke test (real API cost).")
    p.add_argument(
        "--ab",
        action="store_true",
        help="Run base (no rerank, no critic) then tuned stack; record cost delta.",
    )
    p.add_argument(
        "--no-rerank",
        action="store_true",
        help="Single pass: skip cross-encoder rerank.",
    )
    p.add_argument(
        "--no-critic",
        action="store_true",
        help="Single pass: skip Critic LLM.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Iteration floor before planner (planner may raise per ADR-019).",
    )
    p.add_argument(
        "--with-router",
        action="store_true",
        help="Force rule-based tool router on for this run (overrides TOOL_ROUTER_ENABLED=false).",
    )
    p.add_argument(
        "--with-compare-sources",
        action="store_true",
        help="Force compare_sources mode to auto (overrides COMPARE_SOURCES_MODE=off).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    settings = get_settings()
    if not (settings.has_llm_credentials and settings.has_search_credentials):
        logger.error("Missing ANTHROPIC_API_KEY or TAVILY_API_KEY in .env")
        return 2

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_iters = args.max_iterations if args.max_iterations is not None else settings.max_iterations

    router_ov = True if args.with_router else None
    compare_ov: Literal["off", "heuristic", "auto"] | None = (
        "auto" if args.with_compare_sources else None
    )

    previous_metrics: dict[str, Any] | None = None
    if _METRICS_FILE.is_file():
        try:
            previous_metrics = json.loads(_METRICS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_metrics = None

    if args.ab:
        if args.no_rerank or args.no_critic:
            logger.error("Do not combine --ab with --no-rerank / --no-critic (profiles are fixed).")
            return 2

        body_base, metrics_base, cost_b, wall_b, cap_b = _run_smoke_pass(
            label="base",
            no_rerank=True,
            no_critic=True,
            max_iterations=max_iters,
            per_query_cap_usd=settings.per_query_cap_usd,
            tool_router_enabled_override=router_ov,
            compare_sources_mode_override=compare_ov,
        )
        body_tuned, metrics_tuned, cost_t, wall_t, cap_t = _run_smoke_pass(
            label="tuned",
            no_rerank=False,
            no_critic=False,
            max_iterations=max_iters,
            per_query_cap_usd=settings.per_query_cap_usd,
            tool_router_enabled_override=router_ov,
            compare_sources_mode_override=compare_ov,
        )

        report_md = (
            "# Research Assistant — Smoke A/B (base vs tuned)\n\n"
            f"Generated by `scripts/week1_smoke.py --ab`. "
            f"Floor max_iterations (pre-planner) = {max_iters}.\n\n"
            "## Profile: base (no rerank, no critic)\n"
            + "".join(body_base)
            + "\n\n## Profile: tuned (settings defaults)\n"
            + "".join(body_tuned)
        )
        _REPORT_FILE.write_text(report_md, encoding="utf-8")

        payload: dict[str, Any] = {
            "mode": "ab_compare",
            "run_flags": {
                "with_router": bool(args.with_router),
                "with_compare_sources": bool(args.with_compare_sources),
            },
            "max_iterations_floor": max_iters,
            "total_cost_usd": round(cost_t, 6),
            "total_wallclock_s": round(wall_t, 2),
            "queries": metrics_tuned,
            "ab_compare": {
                "base": {
                    "label": "base",
                    "no_rerank": True,
                    "no_critic": True,
                    "total_cost_usd": round(cost_b, 6),
                    "total_wallclock_s": round(wall_b, 2),
                    "queries": metrics_base,
                    "max_iterations_reached_count": cap_b,
                },
                "tuned": {
                    "label": "tuned",
                    "no_rerank": False,
                    "no_critic": False,
                    "total_cost_usd": round(cost_t, 6),
                    "total_wallclock_s": round(wall_t, 2),
                    "queries": metrics_tuned,
                    "max_iterations_reached_count": cap_t,
                },
                "cost_delta_tuned_minus_base_usd": round(cost_t - cost_b, 6),
                "wallclock_delta_tuned_minus_base_s": round(wall_t - wall_b, 2),
            },
        }
        if previous_metrics is not None:
            payload["delta_vs_previous_file"] = {
                "path": str(_METRICS_FILE),
                "cost_delta_usd": round(
                    cost_t - float(previous_metrics.get("total_cost_usd", 0.0)),
                    6,
                ),
                "wallclock_delta_s": round(
                    wall_t - float(previous_metrics.get("total_wallclock_s", 0.0)),
                    2,
                ),
                "previous_total_cost_usd": previous_metrics.get("total_cost_usd"),
                "previous_total_wallclock_s": previous_metrics.get("total_wallclock_s"),
            }
        _METRICS_FILE.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n=== SMOKE A/B SUMMARY ===")
        print(f"Base:   ${cost_b:.4f} · {wall_b:.1f}s · max_iter_cap_hits={cap_b}")
        print(f"Tuned:  ${cost_t:.4f} · {wall_t:.1f}s · max_iter_cap_hits={cap_t}")
        print(
            f"Delta cost (tuned - base): ${cost_t - cost_b:+.4f}  ·  Delta wall: {wall_t - wall_b:+.1f}s"
        )
        print(f"\nReport: {_REPORT_FILE}")
        print(f"Metrics: {_METRICS_FILE}")

        ok_b = all(m["status"] == "ok" for m in metrics_base)
        ok_t = all(m["status"] == "ok" for m in metrics_tuned)
        return 0 if ok_b and ok_t else 1

    # Single profile
    body, metrics, total_cost, total_wallclock, _cap_n = _run_smoke_pass(
        label="single",
        no_rerank=args.no_rerank,
        no_critic=args.no_critic,
        max_iterations=max_iters,
        per_query_cap_usd=settings.per_query_cap_usd,
        tool_router_enabled_override=router_ov,
        compare_sources_mode_override=compare_ov,
    )

    _REPORT_FILE.write_text(
        "# Research Assistant — Week 1 Smoke Test Outputs\n\n"
        f"Generated from `scripts/week1_smoke.py` across {len(QUERIES)} queries.\n"
        f"Flags: no_rerank={args.no_rerank} no_critic={args.no_critic} "
        f"with_router={args.with_router} with_compare_sources={args.with_compare_sources} "
        f"max_iterations_floor={max_iters}.\n"
        f"Total cost: ${total_cost:.4f} · Total wallclock: {total_wallclock:.1f}s\n"
        + "".join(body),
        encoding="utf-8",
    )
    payload_single: dict[str, Any] = {
        "mode": "single",
        "run_flags": {
            "no_rerank": args.no_rerank,
            "no_critic": args.no_critic,
            "with_router": bool(args.with_router),
            "with_compare_sources": bool(args.with_compare_sources),
            "max_iterations_floor": max_iters,
        },
        "total_cost_usd": round(total_cost, 6),
        "total_wallclock_s": round(total_wallclock, 2),
        "queries": metrics,
    }
    if previous_metrics is not None:
        payload_single["delta_vs_previous_file"] = {
            "path": str(_METRICS_FILE),
            "cost_delta_usd": round(
                total_cost - float(previous_metrics.get("total_cost_usd", 0.0)),
                6,
            ),
            "wallclock_delta_s": round(
                total_wallclock - float(previous_metrics.get("total_wallclock_s", 0.0)),
                2,
            ),
            "previous_total_cost_usd": previous_metrics.get("total_cost_usd"),
            "previous_total_wallclock_s": previous_metrics.get("total_wallclock_s"),
        }
    _METRICS_FILE.write_text(
        json.dumps(payload_single, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== WEEK 1 SMOKE TEST SUMMARY ===")
    print(f"Total cost: ${total_cost:.4f}  |  Budget cap: ${settings.max_budget_usd:.2f}")
    print(f"Total wallclock: {total_wallclock:.1f}s")
    for m in metrics:
        print(
            f"  [{m['index']}] {m['status']:5} · "
            f"${m.get('cost_usd', 0):.4f} · "
            f"{m.get('wallclock_s', 0):.1f}s · "
            f"sub-qs={m.get('n_sub_questions', 0)} · "
            f"cites={m.get('total_citations', 0)} · "
            f"cap={m.get('max_iterations_reached', False)} · "
            f"{m['language']} · {m['query'][:55]}…",
        )
    print(f"\nReport: {_REPORT_FILE}")
    print(f"Metrics: {_METRICS_FILE}")
    if "delta_vs_previous_file" in payload_single:
        d = payload_single["delta_vs_previous_file"]
        print(
            f"Delta vs previous metrics file: cost {d['cost_delta_usd']:+.4f} USD, "
            f"wallclock {d['wallclock_delta_s']:+.1f}s",
        )

    return 0 if all(m["status"] == "ok" for m in metrics) else 1


if __name__ == "__main__":
    sys.exit(main())
