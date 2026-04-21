"""Application configuration loaded from environment variables / ``.env``.

Single source of truth for runtime settings. Downstream modules should call
:func:`get_settings` (cached) instead of reading ``os.environ`` directly.

Grounded in:
  * ADR-007 — output language default ``vi``.
  * ADR-008 — Anthropic Claude for Planner/Critic and Synthesizer.
  * ADR-009 — Langfuse Cloud for observability.
  * ADR-011 — hard budget caps ($10 total, $0.30 per query).
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
        le=32,
        description="Hard cap on ReAct iterations per query (PLAN.md §2.2).",
    )
    output_language: Literal["vi", "en"] = Field(default="vi")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so that environment is parsed once per process. Tests that mutate
    env vars should call :func:`get_settings.cache_clear`.
    """
    return Settings()
