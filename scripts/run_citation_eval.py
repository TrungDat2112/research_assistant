"""Batch citation coverage from smoke Markdown (``week1_outputs.md``).

Computes mean paragraph-level ``[^N]`` coverage (same core heuristic as the
Critic) over reporter output, with skips for section headings. Writes
``data/eval/citation_coverage.json`` by default.

Usage::

    uv run python scripts/run_citation_eval.py
    uv run python scripts/run_citation_eval.py --smoke-md path/to/outputs.md --out path/out.json
    uv run python scripts/run_citation_eval.py --target-mean 0.85 --strict-exit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from research_assistant.eval.smoke_citation import run_smoke_citation_eval

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_SMOKE = _REPO / "data" / "eval" / "week1_outputs.md"
_DEFAULT_OUT = _REPO / "data" / "eval" / "citation_coverage.json"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Citation coverage batch from smoke Markdown reports.",
    )
    p.add_argument(
        "--smoke-md",
        type=Path,
        default=_DEFAULT_SMOKE,
        help="Concatenated smoke output Markdown (default: data/eval/week1_outputs.md).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="JSON output path (default: data/eval/citation_coverage.json).",
    )
    p.add_argument(
        "--target-mean",
        type=float,
        default=0.8,
        help="Exit 1 with --strict-exit when mean coverage is below this (default: 0.8).",
    )
    p.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return exit code 1 if mean coverage < --target-mean.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.smoke_md.is_file():
        logger.error("smoke markdown not found: %s", args.smoke_md)
        return 2

    payload = run_smoke_citation_eval(
        args.smoke_md,
        target_mean_coverage=args.target_mean,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mean = float(payload["mean_coverage"])
    meets = bool(payload["meets_target"])
    n = int(payload["n_queries"])
    logger.info(
        "wrote %s — queries=%d mean_coverage=%.4f target=%.2f meets_target=%s",
        args.out,
        n,
        mean,
        float(args.target_mean),
        meets,
    )
    for row in payload["queries"]:
        logger.info(
            "  Q%s (%s): coverage=%.4f paragraphs=%d cited=%d — %s",
            row["query_index"],
            row["language"],
            float(row["paragraph_citation_coverage"]),
            row["substantive_paragraphs"],
            row["cited_paragraphs"],
            (row["title"][:72] + "…") if len(row["title"]) > 72 else row["title"],
        )

    if args.strict_exit and not meets:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
