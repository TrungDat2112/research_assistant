from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):


    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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
        "with Anthropic ephemeral prompt cache (reuse across sub-questions).",
    )

    # ---- Web search (Tavily) --------------------------------------------
    tavily_api_key: SecretStr = Field(default=SecretStr(""))

    langfuse_public_key: SecretStr = Field(default=SecretStr(""))
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""))
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    max_budget_usd: float = Field(default=10.0, ge=0.0)
    budget_alert_usd: float = Field(default=7.0, ge=0.0)
    per_query_cap_usd: float = Field(default=0.30, gt=0.0)

    # ---- Runtime --------------------------------------------------------
    max_iterations: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Default iteration cap before planner runs; "
        "raises this to at least max(8, len(plan) * critic_max_attempts_per_sub_question), "
        "or higher if CLI/env set a larger value.",
    )
    output_language: Literal["vi", "en"] = Field(default="vi")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


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
        description="Target chunk size in tokens.",
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
    tool_router_enabled: bool = Field(
        default=True,
        description="When True, retriever uses heuristic intent → tool order "
        "(vector / web / academic) before merge.",
    )
    tool_router_max_tools: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Max distinct tools to run per sub-question when routing is enabled.",
    )

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
        "falls below this threshol.",
    )
    critic_min_faithfulness: float = Field(
        default=4.0,
        ge=1.0,
        le=5.0,
        description="Minimum LLM faithfulness score (1-5) to pass without retry.",
    )
    critic_min_completeness: float = Field(
        default=4.0,
        ge=1.0,
        le=5.0,
        description="Minimum LLM completeness score (1-5) to pass without retry.",
    )
    critic_min_consistency: float = Field(
        default=4.0,
        ge=1.0,
        le=5.0,
        description="Minimum deterministic consistency score (from conflicts) to pass.",
    )
    compare_sources_mode: Literal["off", "heuristic", "auto"] = Field(
        default="auto",
        description="Cross-source conflict scan before Critic: off / regex+units only / "
        "auto (heuristic + Sonnet when comparative or heuristic hits).",
    )

    # ---- Validators -----------------------------------------------------
    @field_validator("budget_alert_usd")
    @classmethod
    def _alert_below_max(cls, v: float, info: object) -> float:

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

    return max(8, int(num_sub_questions) * int(critic_attempts_per_sub_question))


@lru_cache(maxsize=1)
def get_settings() -> Settings:

    return Settings()
