"""Run stage-1 hybrid retrieval metrics on data/eval/retrieval_eval_30.json.

Requires an ingested Chroma store (``scripts/ingest_seed_corpus.py``).

Usage::

    uv run python scripts/run_retrieval_eval.py
    uv run python scripts/run_retrieval_eval.py --out data/eval/retrieval_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_assistant.config import get_settings
from research_assistant.eval.retrieval import load_retrieval_eval, run_hybrid_retrieval_eval
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.embedding import EmbeddingModel
from research_assistant.rag.vector_store import ChromaStore
from research_assistant.tools.vector_search import clear_vector_search_cache

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_EVAL = _REPO / "data" / "eval" / "retrieval_eval_30.json"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    p = argparse.ArgumentParser(description="Run hybrid retrieval eval (Recall@k, NDCG@10).")
    p.add_argument(
        "--eval-file",
        type=Path,
        default=_DEFAULT_EVAL,
        help="Path to JSON eval set.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full JSON results to this file.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="hybrid final_top_k (use at least 20 for recall@20).",
    )
    args = p.parse_args()
    if args.top_k < 20 or args.top_k > 100:
        logger.error("--top-k must be between 20 and 100")
        return 2

    _setup_logging()
    s = get_settings()
    if not args.eval_file.is_file():
        logger.error("Eval file not found: %s", args.eval_file)
        return 2

    items = load_retrieval_eval(args.eval_file)
    if len(items) != 30:
        logger.warning("Expected 30 items, got %d", len(items))

    clear_vector_search_cache()
    store = ChromaStore(s.chroma_persist_dir, s.corpus_collection)
    n_docs = store.count()
    if n_docs == 0:
        logger.error("Chroma collection is empty — run scripts/ingest_seed_corpus.py first.")
        return 2

    logger.info("Building BM25 index (n_chroma=%d)...", n_docs)
    bm25 = BM25CorpusIndex.from_chroma(store)
    embedder = EmbeddingModel(s.embedding_model, s.embedding_device)

    t0 = datetime.now(tz=UTC)
    result: dict[str, Any] = run_hybrid_retrieval_eval(
        store=store,
        bm25_index=bm25,
        embedder=embedder,
        items=items,
        final_top_k=args.top_k,
        k_recall=(10, 20),
    )
    result["settings_snapshot"] = {
        "embedding_model": s.embedding_model,
        "corpus_collection": s.corpus_collection,
        "chroma_n_vectors": n_docs,
    }
    result["eval_file"] = str(args.eval_file)
    result["wall_time_sec"] = (datetime.now(tz=UTC) - t0).total_seconds()

    print()
    print("--- Retrieval eval (stage-1 hybrid, no cross-encoder) ---")
    print(f"  queries:         {result.get('n')}")
    print(f"  mean recall@10:  {result.get('mean_recall@10', 0):.4f}")
    print(f"  mean recall@20:  {result.get('mean_recall@20', 0):.4f}")
    print(f"  mean NDCG@10:     {result.get('mean_ndcg@10', 0):.4f}")
    print(f"  wall time (s):   {result['wall_time_sec']:.2f}")
    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
