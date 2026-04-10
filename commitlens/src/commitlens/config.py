"""Centralized configuration loaded from environment variables or .env file."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "COMMITLENS_"}

    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024
    max_diff_lines: int = 500
    bandit_enabled: bool = True


settings = Settings()
