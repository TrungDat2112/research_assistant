"""Shared pytest fixtures.

Keeps the unit-test environment hermetic:

* **Langfuse credentials are forcibly cleared** so the observability
  shim degrades to a transparent passthrough. Without this, a
  developer's real ``.env`` would cause the Langfuse SDK to attempt
  span export against the live backend mid-test (we once hit a 401
  storm that way).
* **Anthropic / Tavily keys left alone**: tests that need the chat
  model or the web tool already monkeypatch ``invoke_llm`` /
  ``invoke_structured_llm`` / the search function, so their real
  keys never get used.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from research_assistant.config import get_settings


@pytest.fixture(autouse=True)
def _disable_langfuse_for_tests(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure every unit test runs with Langfuse credentials scrubbed.

    Tests that explicitly want to exercise the enabled path can re-set
    the env vars inside the test body and call
    ``get_settings.cache_clear()``.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
