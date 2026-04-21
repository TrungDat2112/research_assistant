"""Week-1 end-to-end smoke test across the 5 seed queries.

Not a pytest test (it calls real external APIs and costs money). Run::

    uv run python scripts/week1_smoke.py

Outputs:
    * data/eval/week1_outputs.md   — concatenated Markdown reports.
    * data/eval/week1_metrics.json — per-query cost / timing / citation stats.
    * Prints a compact summary table to stdout at the end.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

from research_assistant.config import get_settings
from research_assistant.graph.research_graph import build_graph
from research_assistant.graph.state import new_state

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


def main() -> int:
    # Windows consoles default to cp1252 which cannot encode Vietnamese
    # characters — force UTF-8 for stdout/stderr so the summary renders.
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

    graph = build_graph()
    all_reports: list[str] = []
    metrics: list[dict[str, Any]] = []
    total_cost = 0.0
    total_wallclock = 0.0

    for idx, (query, lang) in enumerate(QUERIES, start=1):
        logger.info("[%d/%d] Running: %r (lang=%s)", idx, len(QUERIES), query, lang)
        start = time.perf_counter()
        initial = new_state(
            query=query,
            output_language=lang,
            max_iterations=settings.max_iterations,
            per_query_cap_usd=settings.per_query_cap_usd,
        )

        try:
            final: Any = graph.invoke(initial)
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
                },
            )
            all_reports.append(f"# Query {idx}: {query}\n\n_(Run failed — see logs)_\n")
            continue

        wallclock = time.perf_counter() - start
        cost = float(final.get("total_cost_usd", 0.0))
        total_cost += cost
        total_wallclock += wallclock

        plan = final.get("plan", [])
        drafts = final.get("drafts", {})
        evidence_counts = {k: len(v) for k, v in final.get("evidence", {}).items()}
        total_citations = sum(len(d.citations) for d in drafts.values())

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
            },
        )

        report = final.get("final_report") or "(empty report)"
        all_reports.append(f"\n\n---\n\n# Query {idx} ({lang.upper()}): {query}\n\n{report}")

        logger.info(
            "  ↳ done in %.1fs, cost=$%.4f, %d sub-qs, %d citations",
            wallclock,
            cost,
            len(plan),
            total_citations,
        )

    _REPORT_FILE.write_text(
        "# Research Assistant — Week 1 Smoke Test Outputs\n\n"
        f"Generated from `scripts/week1_smoke.py` across {len(QUERIES)} queries.\n"
        f"Total cost: ${total_cost:.4f} · Total wallclock: {total_wallclock:.1f}s\n"
        + "".join(all_reports),
        encoding="utf-8",
    )
    _METRICS_FILE.write_text(
        json.dumps(
            {
                "total_cost_usd": round(total_cost, 6),
                "total_wallclock_s": round(total_wallclock, 2),
                "queries": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Pretty summary to stdout
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
            f"{m['language']} · {m['query'][:55]}…",
        )
    print(f"\nReport: {_REPORT_FILE}")
    print(f"Metrics: {_METRICS_FILE}")

    return 0 if all(m["status"] == "ok" for m in metrics) else 1


if __name__ == "__main__":
    sys.exit(main())
