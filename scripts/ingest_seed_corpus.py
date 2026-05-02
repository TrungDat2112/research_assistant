from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make ``src/`` importable when run directly from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from research_assistant.config import get_settings  # noqa: E402
from research_assistant.rag.chunking import ChunkingConfig, chunk_documents  # noqa: E402
from research_assistant.rag.embedding import EmbeddingModel  # noqa: E402
from research_assistant.rag.ingest import SeedConfig, load_seed_corpus  # noqa: E402
from research_assistant.rag.vector_store import ChromaStore  # noqa: E402

logger = logging.getLogger("ingest")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the RAG seed corpus.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "seed_corpus.yaml",
        help="Path to the seed corpus YAML.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the Chroma collection before upserting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of docs to ingest (for smoke runs).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "data" / "eval" / "ingest_manifest.json",
        help="Where to write the run summary manifest.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Tame chatty libs.
    for noisy in ("urllib3", "httpx", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    settings = get_settings()
    config = SeedConfig.from_yaml(args.config)

    total_sources = (
        len(config.arxiv_ids)
        + sum(q.max_results for q in config.arxiv_queries)
        + len(config.html_urls)
    )
    logger.info(
        "Loaded seed config: ~%d sources (%d arxiv ids + %d queries + %d html)",
        total_sources,
        len(config.arxiv_ids),
        len(config.arxiv_queries),
        len(config.html_urls),
    )

    fetch_started = time.perf_counter()
    result = load_seed_corpus(config, arxiv_cache_dir=settings.raw_docs_dir / "arxiv")
    fetch_elapsed = time.perf_counter() - fetch_started

    docs = result.docs
    if args.limit is not None:
        docs = docs[: args.limit]
    logger.info(
        "Fetched %d docs (%d failures) in %.1fs",
        len(docs),
        result.failure_count,
        fetch_elapsed,
    )
    for ident, reason in result.failures:
        logger.warning("Failure: %s — %s", ident, reason)

    chunk_cfg = ChunkingConfig(
        model_id=settings.embedding_model,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunk_started = time.perf_counter()
    chunks = chunk_documents(docs, chunk_cfg)
    chunk_elapsed = time.perf_counter() - chunk_started
    logger.info("Chunked into %d chunks in %.1fs", len(chunks), chunk_elapsed)

    if not chunks:
        logger.error("No chunks to index — aborting.")
        return 2

    embedder = EmbeddingModel(
        model_id=settings.embedding_model,
        device=settings.embedding_device,
    )
    embed_started = time.perf_counter()
    embeddings = embedder.embed_documents([c.text for c in chunks])
    embed_elapsed = time.perf_counter() - embed_started
    logger.info(
        "Embedded %d chunks (dim=%d) in %.1fs (%.1f ch/s)",
        len(chunks),
        embeddings.shape[1],
        embed_elapsed,
        len(chunks) / max(embed_elapsed, 1e-6),
    )

    store = ChromaStore(
        persist_dir=settings.chroma_persist_dir,
        collection=settings.corpus_collection,
    )
    if args.rebuild:
        logger.info("Rebuild mode: dropping collection %s", settings.corpus_collection)
        store.reset()

    upsert_started = time.perf_counter()
    store.upsert_chunks(chunks, embeddings)
    upsert_elapsed = time.perf_counter() - upsert_started
    logger.info("Upserted %d chunks in %.1fs", len(chunks), upsert_elapsed)

    manifest = {
        "config_path": str(args.config),
        "embedding_model": settings.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "collection": settings.corpus_collection,
        "chroma_persist_dir": str(settings.chroma_persist_dir),
        "docs_fetched": len(docs),
        "docs_failed": result.failure_count,
        "chunks_indexed": len(chunks),
        "timings_sec": {
            "fetch": round(fetch_elapsed, 2),
            "chunk": round(chunk_elapsed, 2),
            "embed": round(embed_elapsed, 2),
            "upsert": round(upsert_elapsed, 2),
        },
        "sources": [
            {
                "source_id": d.source_id,
                "doc_type": d.doc_type,
                "title": d.title,
                "url": d.url,
                "chars": len(d.text),
            }
            for d in docs
        ],
        "failures": [{"id": ident, "reason": reason} for ident, reason in result.failures],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote manifest → %s", args.manifest)
    logger.info("Collection now holds %d chunks total.", store.count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
