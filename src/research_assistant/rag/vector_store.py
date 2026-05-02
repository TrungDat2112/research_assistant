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
    chunk_id: str
    body: str
    metadata: dict[str, Any]
    distance: float


class ChromaStore:
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

    def get_by_ids(
        self,
        chunk_ids: list[str],
        *,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not chunk_ids:
            return []
        raw: dict[str, Any] = self._collection.get(
            ids=chunk_ids,
            where=where,
            include=["documents", "metadatas"],
        )
        out_ids: list[str] = list(raw.get("ids") or [])
        docs: list[str] = list((raw.get("documents") or [""] * len(out_ids))[: len(out_ids)])
        metas: list[dict[str, Any]] = list(
            (raw.get("metadatas") or [{}] * len(out_ids))[: len(out_ids)],
        )
        return [
            SearchResult(
                chunk_id=cid,
                body=doc or "",
                metadata=dict(meta) if meta else {},
                distance=0.0,
            )
            for cid, doc, meta in zip(out_ids, docs, metas, strict=True)
        ]

    def fetch_all_documents(
        self,
        *,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:

        raw: dict[str, Any] = self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        out_ids: list[str] = list(raw.get("ids") or [])
        if not out_ids:
            return []
        docs: list[str] = list(
            (raw.get("documents") or [""] * len(out_ids))[: len(out_ids)],
        )
        metas: list[dict[str, Any]] = list(
            (raw.get("metadatas") or [{}] * len(out_ids))[: len(out_ids)],
        )
        return [
            SearchResult(
                chunk_id=cid,
                body=doc or "",
                metadata=dict(meta) if meta else {},
                distance=0.0,
            )
            for cid, doc, meta in zip(out_ids, docs, metas, strict=True)
        ]

    # -- introspection ---------------------------------------------------

    def count(self) -> int:
        return int(self._collection.count())
