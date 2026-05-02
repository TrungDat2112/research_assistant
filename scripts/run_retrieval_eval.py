from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_assistant.config import get_settings
from research_assistant.eval.retrieval import (
    load_retrieval_eval,
    run_hybrid_retrieval_eval,
    run_rerank_retrieval_eval,
)
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.embedding import EmbeddingModel
from research_assistant.rag.vector_store import ChromaStore
from research_assistant.tools.vector_search import clear_vector_search_cache

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_EVAL = _REPO / "data" / "eval" / "retrieval_eval_100.json"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in ("httpx", "httpcore", "chromadb", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _print_block(title: str, r: dict[str, Any], wall: float) -> None:
    print(f"--- {title} ---")
    print(f"  queries:            {r.get('n')}")
    print(f"  mean recall@10:     {r.get('mean_recall@10', 0):.4f}")
    print(f"  mean recall@20:     {r.get('mean_recall@20', 0):.4f}")
    print(f"  mean NDCG@10:       {r.get('mean_ndcg@10', 0):.4f}")
    print(f"  mean MRR:           {r.get('mean_mrr', 0):.4f}")
    print(f"  mean precision@5:   {r.get('mean_precision@5', 0):.4f}")
    print(f"  wall time (s):      {wall:.2f}")
    print()


def _delta(a: dict[str, Any], b: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(b.get(k, 0.0)) - float(a.get(k, 0.0))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run hybrid retrieval eval (Recall@k, NDCG@10, MRR, P@5).",
    )
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
        help="Rank depth after fusion / re-rank (use at least 20 for recall@20).",
    )
    p.add_argument(
        "--with-rerank",
        action="store_true",
        help="Also run cross-encoder re-rank (stage-1 pool → CE → top-k) and print A/B vs baseline.",
    )
    p.add_argument(
        "--with-hyde",
        action="store_true",
        help="Also run stage-1 hybrid with HyDE dense rewrite on weak probe (needs ANTHROPIC_API_KEY).",
    )
    p.add_argument(
        "--candidate-pool",
        type=int,
        default=50,
        help="Stage-1 hybrid pool size before re-rank (only with --with-rerank; must be >= --top-k).",
    )
    args = p.parse_args()
    if args.top_k < 20 or args.top_k > 100:
        logger.error("--top-k must be between 20 and 100")
        return 2
    if args.with_rerank and args.candidate_pool < args.top_k:
        logger.error("--candidate-pool must be >= --top-k when using --with-rerank")
        return 2

    _setup_logging()
    s = get_settings()
    if not args.eval_file.is_file():
        logger.error("Eval file not found: %s", args.eval_file)
        return 2

    items = load_retrieval_eval(args.eval_file)
    logger.info("Loaded %d eval items from %s", len(items), args.eval_file)

    if args.with_hyde and not s.has_llm_credentials:
        logger.error("--with-hyde requires ANTHROPIC_API_KEY for HyDE passage generation.")
        return 2

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
    hybrid = run_hybrid_retrieval_eval(
        store=store,
        bm25_index=bm25,
        embedder=embedder,
        items=items,
        final_top_k=args.top_k,
        k_recall=(10, 20),
    )
    t1 = datetime.now(tz=UTC)
    wall_hybrid = (t1 - t0).total_seconds()

    result: dict[str, Any] = {
        "stage1_hybrid": hybrid,
        "settings_snapshot": {
            "embedding_model": s.embedding_model,
            "reranker_model": s.reranker_model,
            "corpus_collection": s.corpus_collection,
            "chroma_n_vectors": n_docs,
            "hyde_enabled_default": s.hyde_enabled,
            "hyde_min_top1_fused_score": s.hyde_min_top1_fused_score,
            "hyde_min_fused_margin": s.hyde_min_fused_margin,
        },
        "eval_file": str(args.eval_file),
    }

    print()
    _print_block("A — Stage-1 hybrid (no cross-encoder)", hybrid, wall_hybrid)

    if args.with_hyde:
        t_h0 = datetime.now(tz=UTC)
        hyde_run = run_hybrid_retrieval_eval(
            store=store,
            bm25_index=bm25,
            embedder=embedder,
            items=items,
            final_top_k=args.top_k,
            k_recall=(10, 20),
            use_hyde=True,
        )
        t_h1 = datetime.now(tz=UTC)
        wall_hyde = (t_h1 - t_h0).total_seconds()
        result["stage1_hybrid_hyde"] = hyde_run
        result["hyde_delta_vs_baseline"] = _delta(
            hybrid,
            hyde_run,
            (
                "mean_recall@10",
                "mean_recall@20",
                "mean_ndcg@10",
                "mean_mrr",
                "mean_precision@5",
            ),
        )
        result["wall_time_sec_hyde"] = wall_hyde
        print(
            f"--- HyDE triggers: {hyde_run.get('n_hyde_triggers', 0)} / {hyde_run.get('n', 0)} ---",
        )
        _print_block("A-prime - Stage-1 hybrid + HyDE (eval override)", hyde_run, wall_hyde)
        hd = result["hyde_delta_vs_baseline"]
        print("--- HyDE minus baseline (A-prime - A) ---")
        for k, v in hd.items():
            print(f"  {k}:  {v:+.4f}")
        print()

    if args.with_rerank:
        t2 = datetime.now(tz=UTC)
        rer = run_rerank_retrieval_eval(
            store=store,
            bm25_index=bm25,
            embedder=embedder,
            items=items,
            candidate_pool=args.candidate_pool,
            final_top_k=args.top_k,
            k_recall=(10, 20),
            use_hyde=args.with_hyde,
        )
        t3 = datetime.now(tz=UTC)
        wall_rer = (t3 - t2).total_seconds()
        result["stage1_hybrid_rerank"] = rer
        result["ab_delta"] = _delta(
            hybrid,
            rer,
            (
                "mean_recall@10",
                "mean_recall@20",
                "mean_ndcg@10",
                "mean_mrr",
                "mean_precision@5",
            ),
        )
        result["wall_time_sec"] = {"stage1_hybrid": wall_hybrid, "rerank_eval": wall_rer}
        if args.with_hyde:
            result["wall_time_sec"]["stage1_hybrid_hyde"] = result.get(
                "wall_time_sec_hyde",
                0.0,
            )
        _print_block(
            f"B — Stage-1 pool ({args.candidate_pool}) + cross-encoder → top-{args.top_k}",
            rer,
            wall_rer,
        )
        d = result["ab_delta"]
        print("--- A/B (B minus A) ---")
        for k, v in d.items():
            print(f"  {k}:  {v:+.4f}")
        print()
    else:
        wt: dict[str, Any] = {"stage1_hybrid": wall_hybrid}
        if args.with_hyde:
            wt["stage1_hybrid_hyde"] = result.get("wall_time_sec_hyde", 0.0)
        result["wall_time_sec"] = wt

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
