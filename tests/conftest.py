from __future__ import annotations

from collections.abc import Iterator

import pytest

from research_assistant.config import get_settings


@pytest.fixture(autouse=True)
def _disable_langfuse_for_tests(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
