"""Tests for :mod:`research_assistant.config`."""

from __future__ import annotations

import pytest

from research_assistant.config import Settings, get_settings, planned_max_iterations


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    get_settings.cache_clear()


def test_defaults_do_not_require_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.output_language == "vi"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.max_iterations == 8
    assert settings.max_budget_usd == pytest.approx(10.0)
    assert settings.per_query_cap_usd == pytest.approx(0.30)
    assert settings.has_llm_credentials is False
    assert settings.has_search_credentials is False
    assert settings.langfuse_enabled is False
    assert settings.hyde_enabled is False
    assert settings.tool_router_enabled is True
    assert settings.tool_router_max_tools == 3
    assert settings.compare_sources_mode == "auto"
    assert settings.critic_min_faithfulness == pytest.approx(4.0)
    assert settings.critic_min_completeness == pytest.approx(4.0)
    assert settings.critic_min_consistency == pytest.approx(4.0)


def test_credentials_flip_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.has_llm_credentials is True
    assert settings.has_search_credentials is True
    assert settings.langfuse_enabled is True


def test_per_query_cap_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, per_query_cap_usd=0.0)  # type: ignore[call-arg]


def test_planned_max_iterations_floor_eight() -> None:
    assert planned_max_iterations(1, 2) == 8
    assert planned_max_iterations(3, 2) == 8
    assert planned_max_iterations(5, 2) == 10
    assert planned_max_iterations(7, 2) == 14
