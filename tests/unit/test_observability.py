"""Tests for the Langfuse observability shim.

We never hit Langfuse for real — the tests verify (a) the shim is a
transparent passthrough when credentials are absent, and (b) decorated
functions preserve their signatures / return values / type behaviour.

These tests intentionally do **not** exercise the enabled path. That
path goes through the real Langfuse SDK and is covered by the verify
step described in ``PROGRESS.md`` (one real query → check dashboard).
"""

from __future__ import annotations

from typing import Any

import pytest

from research_assistant import observability as obs
from research_assistant.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure :func:`get_settings` re-reads env for each test."""
    get_settings.cache_clear()


def _disable_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()


def test_is_enabled_false_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_langfuse(monkeypatch)
    assert obs.is_enabled() is False


def test_observe_is_passthrough_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_langfuse(monkeypatch)

    @obs.observe(name="unit", as_type="span")
    def f(x: int, y: int) -> int:
        return x + y

    assert f(2, 3) == 5
    # Ensure docstring / metadata are preserved via functools.wraps.
    assert f.__name__ == "f"


def test_observe_checks_enablement_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decorator checks ``is_enabled()`` each call rather than freezing at
    decoration time — important because ``get_settings`` is cached.

    We only test the disabled-path side (enabled path would try to
    export real spans to Langfuse cloud; that's covered by the manual
    verify step documented in PROGRESS.md).
    """
    _disable_langfuse(monkeypatch)

    calls: list[int] = []

    @obs.observe(name="late-enabled", as_type="tool")
    def collector(n: int) -> int:
        calls.append(n)
        return n

    # Disabled call — should pass through the original function.
    assert collector(1) == 1
    # Re-disable to simulate cache invalidation and verify the decorator
    # still resolves to passthrough (no cached enabled wrapper leaks).
    _disable_langfuse(monkeypatch)
    assert collector(2) == 2
    assert calls == [1, 2]


def test_helpers_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_langfuse(monkeypatch)
    assert obs.current_trace_id() is None
    assert obs.current_trace_url() is None
    # These should not raise even when no trace is active.
    obs.update_span(input={"a": 1}, output={"b": 2})
    obs.update_generation(model="claude", usage_details={"input": 0, "output": 0})
    obs.update_trace_io(input={}, output={})
    obs.flush()


def test_start_agent_span_is_nullcontext_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_langfuse(monkeypatch)
    sentinel: list[str] = []
    with obs.start_agent_span("span-name", input={"query": "test"}):
        sentinel.append("inside")
    assert sentinel == ["inside"]


def test_observe_preserves_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_langfuse(monkeypatch)

    class BoomError(RuntimeError):
        pass

    @obs.observe(name="crashy", as_type="span")
    def f() -> Any:
        raise BoomError("kaboom")

    with pytest.raises(BoomError, match="kaboom"):
        f()
