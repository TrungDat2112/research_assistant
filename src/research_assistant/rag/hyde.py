from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research_assistant.config import Settings, get_settings
from research_assistant.rag.bm25_index import BM25CorpusIndex
from research_assistant.rag.embedding import EmbeddingModel
from research_assistant.rag.hybrid import HybridSearchResult, hybrid_search_stage1
from research_assistant.rag.vector_store import ChromaStore

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You help dense retrieval. Write ONE short hypothetical paragraph (3-6 sentences) "
    "that could appear in an AI/ML paper or technical blog and that directly addresses "
    "the user's question. Use clear technical language in the **same language** as the "
    "question (Vietnamese or English). Do not include URLs, citation markers [^n], or "
    'a "References" section.'
)


def hyde_probe_triggers(
    probe: list[HybridSearchResult],
    *,
    min_top1_fused_score: float,
    min_fused_margin: float,
) -> bool:
    if not probe:
        return True
    top1 = float(probe[0].combined_score)
    if len(probe) == 1:
        return top1 < min_top1_fused_score
    margin = top1 - float(probe[1].combined_score)
    return top1 < min_top1_fused_score or margin < min_fused_margin


def generate_hyde_passage(
    query: str,
    *,
    settings: Settings | None = None,
) -> tuple[str, float]:
    from research_assistant.agents._llm import invoke_llm

    s = settings or get_settings()
    result = invoke_llm(
        s.anthropic_synthesizer_model,
        query.strip(),
        system=_HYDE_SYSTEM,
        temperature=0.25,
        max_tokens=s.hyde_max_tokens,
        current_cost_usd=0.0,
        per_query_cap_usd=s.per_query_cap_usd,
    )
    text = result.text.strip()
    if not text:
        logger.warning("HyDE returned empty text; falling back to raw query embedding only")
    return text, float(result.cost_usd)


def dense_embedding_for_retrieval(
    query: str,
    store: ChromaStore,
    bm25_index: BM25CorpusIndex,
    embedder: EmbeddingModel,
    *,
    where: dict[str, Any] | None = None,
    hyde_enabled: bool | None = None,
    settings: Settings | None = None,
    hypothesis_fn: Callable[[str], tuple[str, float]] | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:

    s = settings or get_settings()
    active = s.hyde_enabled if hyde_enabled is None else hyde_enabled
    meta: dict[str, Any] = {
        "hyde_applied": False,
        "hyde_reason": "disabled" if not active else None,
    }
    qvec = embedder.embed_query(query)
    if not active:
        return qvec, meta

    probe = hybrid_search_stage1(
        store,
        bm25_index,
        query,
        qvec,
        dense_top_k=50,
        bm25_top_k=50,
        final_top_k=2,
        where=where,
    )
    if not hyde_probe_triggers(
        probe,
        min_top1_fused_score=s.hyde_min_top1_fused_score,
        min_fused_margin=s.hyde_min_fused_margin,
    ):
        meta["hyde_reason"] = "probe_ok"
        return qvec, meta

    def _default_gen(q: str) -> tuple[str, float]:
        return generate_hyde_passage(q, settings=s)

    gen = hypothesis_fn or _default_gen
    hypo, hy_cost = gen(query)
    meta["hyde_llm_cost_usd"] = round(hy_cost, 6)
    if not hypo:
        meta["hyde_reason"] = "empty_hypothesis"
        return qvec, meta

    hvec = embedder.embed_query(hypo)
    meta["hyde_applied"] = True
    meta["hyde_reason"] = "probe_weak"
    meta["hypothesis_chars"] = len(hypo)
    logger.info(
        "HyDE applied (chars=%d, hyde_llm_cost_usd=%s)",
        len(hypo),
        meta.get("hyde_llm_cost_usd"),
    )
    return hvec, meta
