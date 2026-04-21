"""Command-line entry point for running the research graph end-to-end.

Usage::

    uv run python -m research_assistant.cli "Your research question here"

Options:
    --language {vi,en}     Output language (default: from .env / vi).
    --out PATH             Write the report to a file in addition to stdout.
    --max-iterations N     Override MAX_ITERATIONS (default: from .env / 8).

Designed to be non-interactive so it can drive smoke-test scripts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, cast

from research_assistant.config import get_settings
from research_assistant.graph.research_graph import build_graph
from research_assistant.graph.state import new_state

logger = logging.getLogger("research_assistant.cli")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description="Run a single research query through the Week-1 agent graph.",
    )
    parser.add_argument("query", help="The research question (wrap in quotes).")
    parser.add_argument(
        "--language",
        choices=("vi", "en"),
        default=None,
        help="Output language; defaults to OUTPUT_LANGUAGE from .env.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the final Markdown report to.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override MAX_ITERATIONS guardrail for this run.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    _configure_logging(args.log_level or settings.log_level)

    if not settings.has_llm_credentials:
        print(
            "ERROR: ANTHROPIC_API_KEY is not configured (see .env).",
            file=sys.stderr,
        )
        return 2
    if not settings.has_search_credentials:
        print("ERROR: TAVILY_API_KEY is not configured (see .env).", file=sys.stderr)
        return 2

    output_language = cast("Literal['vi', 'en']", args.language or settings.output_language)
    max_iters = args.max_iterations or settings.max_iterations

    initial = new_state(
        query=args.query,
        output_language=output_language,
        max_iterations=max_iters,
        per_query_cap_usd=settings.per_query_cap_usd,
    )

    graph = build_graph()

    logger.info("Running graph for query=%r language=%s", args.query, output_language)
    final_state = graph.invoke(initial)

    report = final_state.get("final_report") or "(empty report)"
    cost = final_state.get("total_cost_usd", 0.0)
    logger.info(
        "Done. cost=$%.4f trace_steps=%d plan_size=%d",
        cost,
        len(final_state.get("trace", [])),
        len(final_state.get("plan", [])),
    )

    sys.stdout.write(report)
    if not report.endswith("\n"):
        sys.stdout.write("\n")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        logger.info("Wrote report to %s", args.out)

    return 0


def main() -> None:  # pragma: no cover — thin shim
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    main()
