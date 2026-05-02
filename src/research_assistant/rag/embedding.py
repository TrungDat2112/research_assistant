from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _load_model(model_id: str, device: str) -> Any:

    from sentence_transformers import SentenceTransformer

    with _MODEL_LOCK:
        logger.info("Loading embedding model %s on %s", model_id, device)
        return SentenceTransformer(model_id, device=device)


def _needs_query_prefix(model_id: str) -> bool:
    lid = model_id.lower()
    return "bge" in lid and "reranker" not in lid


class EmbeddingModel:
    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "cpu",
        *,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._use_query_prefix = _needs_query_prefix(model_id)

    @property
    def model(self) -> Any:
        return _load_model(self.model_id, self.device)

    @property
    def dimension(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Cannot resolve embedding dimension for {self.model_id}")
        return int(dim)

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        payload = f"{_BGE_QUERY_PREFIX}{text}" if self._use_query_prefix else text
        vec = self.model.encode(
            [payload],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vec[0], dtype=np.float32)
