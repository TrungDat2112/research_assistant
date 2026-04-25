"""Tests for Anthropic-oriented LangChain message construction."""

from __future__ import annotations

import pytest

from research_assistant.agents._llm import build_lc_messages
from research_assistant.config import get_settings


def test_build_lc_messages_cache_control_on_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE_ENABLED", "true")
    get_settings.cache_clear()
    msgs = build_lc_messages(
        system="Static rules",
        user_text="Variable task",
        use_prompt_cache=True,
    )
    assert len(msgs) == 2
    blocks = msgs[0].content
    assert isinstance(blocks, list)
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_build_lc_messages_prefix_second_human_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE_ENABLED", "true")
    get_settings.cache_clear()
    msgs = build_lc_messages(
        system="Sys",
        user_text="Rest",
        cacheable_user_prefix="Same every call",
        use_prompt_cache=True,
    )
    assert len(msgs) == 2
    ublocks = msgs[1].content
    assert isinstance(ublocks, list) and len(ublocks) == 2
    assert ublocks[0]["cache_control"] == {"type": "ephemeral"}
    assert ublocks[1]["text"] == "Rest"


def test_build_lc_messages_disabled_no_cache_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE_ENABLED", "false")
    get_settings.cache_clear()
    msgs = build_lc_messages(
        system="Sys",
        user_text="Hi",
        cacheable_user_prefix="Prefix",
        use_prompt_cache=False,
    )
    assert msgs[0].content == "Sys"
    assert msgs[1].content == "Prefix\n\nHi"
