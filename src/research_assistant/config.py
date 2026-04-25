"""Application configuration loaded from environment variables / ``.env``.

Single source of truth for runtime settings. Downstream modules should call
:func:`get_settings` (cached) instead of reading ``os.environ`` directly.

Grounded in:
  * ADR-007 — output language default ``vi``.
  * ADR-008 — Anthropic Claude for Planner/Critic and Synthesizer.
  * ADR-009 — Langfuse Cloud for observability.
  * ADR-011 — hard budget caps ($10 total, $0.30 per query).
  * ADR-019 — ``max_iterations`` raised after planner from plan size + Critic attempts.
  * ADR-020 — Anthropic prompt caching on static system + repeating user prefix.
  * ADR-018 — default dense embedding ``BAAI/bge-m3`` (override with
    ``EMBEDDING_MODEL`` for English-only fast iteration).
  * ADR-026 — optional HyDE (hypothetical document embedding) for weak hybrid probe.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed view over environment variables.

    Fields with ``SecretStr`` are redacted in ``repr`` and must be accessed
    via ``.get_secret_value()`` to obtain the raw credential.
    """

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM provider (ADR-008) -----------------------------------------
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key. Required to run agents; empty default "
        "lets the package import in dev/CI without secrets.",
    )
    anthropic_planner_model: str = Field(
        default="claude-sonnet-4-5",
        description="Model id for Planner & Critic agents (strong reasoning).",
    )
    anthropic_synthesizer_model: str = Field(
        default="claude-haiku-4-5",
        description="Model id for Synthesizer. Fallback to "
        "'claude-3-5-haiku-latest' if Haiku 4.5 is unavailable at runtime.",
    )
    anthropic_prompt_cache_enabled: bool = Field(
        default=True,
        description="When True, mark static system text and repeating user-query prefix "
        "with Anthropic ephemeral prompt cache (reuse across sub-questions). ADR-020.",
    )

    # ---- Web search (Tavily) --------------------------------------------
    tavily_api_key: SecretStr = Field(default=SecretStr(""))

    # ---- Observability (Langfuse Cloud, ADR-009) ------------------------
    langfuse_public_key: SecretStr = Field(default=SecretStr(""))
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""))
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ---- Budget guardrails (ADR-011) ------------------------------------
    max_budget_usd: float = Field(default=10.0, ge=0.0)
    budget_alert_usd: float = Field(default=7.0, ge=0.0)
    per_query_cap_usd: float = Field(default=0.30, gt=0.0)

    # ---- Runtime --------------------------------------------------------
    max_iterations: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Default iteration cap before planner runs; after planner ADR-019 "
        "raises this to at least max(8, len(plan) * critic_max_attempts_per_sub_question), "
        "or higher if CLI/env set a larger value.",
    )
    output_language: Literal["vi", "en"] = Field(default="vi")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ---- RAG (ADR-013 + ADR-018) ----------------------------------------
    # Default ``bge-m3`` for multilingual / Vietnamese-ready indexing. For
    # fast iteration on English-only corpora, set ``EMBEDDING_MODEL`` to
    # ``BAAI/bge-small-en-v1.5`` and run ``ingest_seed_corpus.py --rebuild``.
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Sentence-transformers model id for dense embeddings.",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Torch device for local embedding inference.",
    )
    chroma_persist_dir: Path = Field(
        default_factory=lambda: _REPO_ROOT / "data" / "chroma",
        description="Directory for Chroma PersistentClient (dev store).",
    )
    corpus_collection: str = Field(
        default="ai_ml_corpus_v1",
        description="Name of the primary Chroma collection for the seed corpus.",
    )
    chunk_size_tokens: int = Field(
        default=500,
        ge=128,
        le=2048,
        description="Target chunk size in tokens (PLAN §5.1 / ADR-003).",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        le=512,
        description="Sliding-window overlap between adjacent chunks.",
    )
    raw_docs_dir: Path = Field(
        default_factory=lambda: _REPO_ROOT / "data" / "raw",
        description="Directory for cached raw source documents (PDF / HTML).",
    )

    # ---- RAG stage 2 — cross-encoder (PLAN §5.2 / ADR-002) ---------------
    reranker_enabled: bool = Field(
        default=True,
        description="When True, retriever re-ranks merged candidates with a "
        "CrossEncoder before the Synthesizer. Set False in tests to avoid "
        "downloading the reranker weights.",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="sentence-transformers CrossEncoder id for stage-2 precision.",
    )
    reranker_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Torch device for the reranker (match embedding_device if GPU).",
    )
    retrieval_candidate_pool: int = Field(
        default=20,
        ge=5,
        le=64,
        description="Max merged web+corpus hits before cross-encoder (stage-1 cap).",
    )
    synthesizer_evidence_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Hits passed to the Synthesizer after re-ranking.",
    )

    # ---- HyDE — optional dense query rewrite (PLAN §5.2) -----------------
    hyde_enabled: bool = Field(
        default=False,
        description="When True, weak hybrid top-2 probe may replace the dense embedding "
        "with an embedded hypothetical passage (HyDE). BM25 still uses the raw query.",
    )
    hyde_min_top1_fused_score: float = Field(
        default=0.38,
        ge=0.0,
        le=1.0,
        description="Trigger HyDE when fused top-1 score (after per-leg min-max) is below this.",
    )
    hyde_min_fused_margin: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="Trigger HyDE when top1-top2 fused margin is below this (ambiguous).",
    )
    hyde_max_tokens: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Max output tokens for the hypothetical passage (synthesizer model).",
    )

    # ---- Critic (PLAN §6.2 / ADR-005) -----------------------------------
    critic_enabled: bool = Field(
        default=True,
        description="When False, the Critic auto-passes (tests / dry runs).",
    )
    critic_max_attempts_per_sub_question: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Max synthesis rounds per sub-question (initial + retries).",
    )
    critic_min_paragraph_citation_coverage: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Reject drafts when deterministic paragraph citation coverage "
        "falls below this threshold (ADR-005).",
    )

    # ---- Validators -----------------------------------------------------
    @field_validator("budget_alert_usd")
    @classmethod
    def _alert_below_max(cls, v: float, info: object) -> float:
        # Pydantic v2: values available via info.data, but we keep this simple
        # — real cross-field enforcement happens in the budget tracker.
        return v

    # ---- Derived helpers -----------------------------------------------
    @property
    def repo_root(self) -> Path:
        return _REPO_ROOT

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key.get_secret_value(),
        )

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.anthropic_api_key.get_secret_value())

    @property
    def has_search_credentials(self) -> bool:
        return bool(self.tavily_api_key.get_secret_value())


def planned_max_iterations(num_sub_questions: int, critic_attempts_per_sub_question: int) -> int:
    """Iteration budget after planning (ADR-019).

    Each graph loop tick consumes one iteration when the Critic finishes a
    sub-question (including retries). Worst case is roughly one Critic step
    per attempt for every sub-question.
    """
    return max(8, int(num_sub_questions) * int(critic_attempts_per_sub_question))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so that environment is parsed once per process. Tests that mutate
    env vars should call :func:`get_settings.cache_clear`.
    """
    return Settings()
