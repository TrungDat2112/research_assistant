from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from research_assistant.rag.vector_store import ChromaStore, SearchResult

_TOKEN_RE = re.compile(r"[\w']+", flags=re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t]


@dataclass(frozen=True)
class BM25CorpusIndex:
    _chunk_ids: tuple[str, ...]
    _bm25: Any

    @classmethod
    def from_chroma(
        cls,
        store: ChromaStore,
        *,
        where: dict[str, Any] | None = None,
        limit: int | None = 10_000,
    ) -> BM25CorpusIndex:
        """Tokenise every document body currently in the collection."""
        rows = store.fetch_all_documents(where=where, limit=limit)
        if not rows:
            return cls((), _EmptyBM25())
        return cls.from_search_results(rows)

    @classmethod
    def from_search_results(cls, rows: list[SearchResult]) -> BM25CorpusIndex:
        if not rows:
            return cls((), _EmptyBM25())
        ids: list[str] = []
        tokenized: list[list[str]] = []
        for r in rows:
            ids.append(r.chunk_id)
            t = tokenize_for_bm25(r.body)
            tokenized.append(t if t else ["_"])
        return cls(tuple(ids), BM25Okapi(tokenized))

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return self._chunk_ids

    def size(self) -> int:
        return len(self._chunk_ids)

    def top_n(
        self,
        query: str,
        n: int,
    ) -> list[tuple[str, float]]:
        if n <= 0 or not self._chunk_ids or isinstance(self._bm25, _EmptyBM25):
            return []
        q = tokenize_for_bm25(query)
        scores = self._bm25.get_scores(q)
        arr = np.asarray(scores, dtype=np.float64)
        m = int(arr.shape[0])
        n_take = min(n, m)
        if m == 0 or n_take == 0:
            return []
        if m <= n_take:
            top_idx = np.argsort(-arr)
        else:
            part = np.argpartition(-arr, n_take - 1)[:n_take]
            top_idx = part[np.argsort(-arr[part])]
        out: list[tuple[str, float]] = []
        for i in top_idx:
            j = int(i)
            cid = self._chunk_ids[j]
            out.append((cid, float(arr[j])))
        return out

    def raw_score_for(
        self,
        query: str,
        chunk_id: str,
    ) -> float | None:
        if not self._chunk_ids or isinstance(self._bm25, _EmptyBM25):
            return None
        try:
            pos = self._chunk_ids.index(chunk_id)
        except ValueError:
            return None
        q = tokenize_for_bm25(query)
        scores = self._bm25.get_scores(q)
        if pos < 0 or pos >= len(scores):
            return None
        return float(scores[pos])


class _EmptyBM25:
    def get_scores(self, _query: list[str]) -> list[float]:
        return []


__all__ = [
    "BM25CorpusIndex",
    "tokenize_for_bm25",
]
