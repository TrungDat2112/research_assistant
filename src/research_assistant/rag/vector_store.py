"""Chroma-backed vector store (dev; Qdrant takes over in prod per PLAN §3).

Design:
  * PersistentClient rooted at ``Settings.chroma_persist_dir`` so state
    survives restarts without running the Chroma server binary.
  * We manage embeddings ourselves (``embedding_function=None`` equivalent
    by always passing ``embeddings=...`` on ``add`` and ``query``). This
    keeps the store provider-agnostic and avoids Chroma downloading
    defaults when the embedding model is already loaded.
  * ``upsert_chunks`` (not ``add``) to make re-runs idempotent — re-indexing
    the same document won't duplicate rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from research_assistant.rag.schemas import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Single hit from a vector search."""

    chunk_id: str
    body: str
    metadata: dict[str, Any]
    distance: float


class ChromaStore:
    """Thin idempotent wrapper over a Chroma PersistentClient collection."""

    def __init__(
        self,
        persist_dir: Path,
        collection: str,
        *,
        distance: str = "cosine",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection
        self.distance = distance
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._make_client()
        self._collection = self._get_or_create_collection()

    def _make_client(self) -> Any:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        return chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

    def _get_or_create_collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )

    # -- mutation --------------------------------------------------------

    def reset(self) -> None:
        """Drop and recreate the collection — used by ingest --rebuild."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as exc:  # chromadb raises on missing; it's fine
            logger.debug("delete_collection no-op: %s", exc)
        self._collection = self._get_or_create_collection()

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: NDArray[np.float32],
    ) -> None:
        """Insert-or-update ``chunks`` with pre-computed ``embeddings``."""
        if len(chunks) == 0:
            return
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {embeddings.shape[0]}",
            )
        ids = [c.chunk_id for c in chunks]
        documents = [c.body for c in chunks]
        metadatas: list[dict[str, Any]] = [
            cast("dict[str, Any]", c.metadata.to_chroma()) for c in chunks
        ]
        embed_list = embeddings.tolist()
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embed_list,
            metadatas=metadatas,
        )
        logger.info(
            "Upserted %d chunks into collection=%s (total=%d)",
            len(chunks),
            self.collection_name,
            self.count(),
        )

    # -- query -----------------------------------------------------------

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 10,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        raw: dict[str, Any] = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids_nested: list[list[str]] = raw.get("ids") or [[]]
        docs_nested: list[list[str]] | None = raw.get("documents")
        metas_nested: list[list[dict[str, Any]]] | None = raw.get("metadatas")
        dists_nested: list[list[float]] | None = raw.get("distances")
        ids: list[str] = ids_nested[0] if ids_nested else []
        docs: list[str] = (docs_nested or [[""] * len(ids)])[0]
        metas: list[dict[str, Any]] = (metas_nested or [[{}] * len(ids)])[0]
        dists: list[float] = (dists_nested or [[0.0] * len(ids)])[0]
        results: list[SearchResult] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            results.append(
                SearchResult(
                    chunk_id=cid,
                    body=doc or "",
                    metadata=dict(meta) if meta else {},
                    distance=float(dist),
                ),
            )
        return results

    # -- introspection ---------------------------------------------------

    def count(self) -> int:
        return int(self._collection.count())
