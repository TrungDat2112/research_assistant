"""Langfuse instrumentation shim.

Wraps :mod:`langfuse` v4 so the rest of the codebase can decorate nodes /
tools / LLM calls without caring whether credentials are configured.

Design goals
------------
* ``@observe(...)`` — identical ergonomics to ``langfuse.observe`` but
  degrades to a transparent passthrough when
  :attr:`Settings.langfuse_enabled` is ``False``. Crucially, this means
  tests and CI runs without keys do not emit Langfuse auth warnings.
* ``current_trace_id`` / ``current_trace_url`` — always safe to call;
  return ``None`` when disabled or when not inside an active span.
* ``update_span`` / ``update_generation`` / ``update_trace_io`` /
  ``flush`` — no-op when disabled so nodes don't need local guards.
* Deliberately depends on ``config.get_settings()`` at *call* time (not
  import time) so the settings cache can be cleared in tests and the
  decorator still picks up new credentials on the next invocation.

All of this is kept in one module so Langfuse version bumps only touch
here — rest of the package imports from :mod:`research_assistant.observability`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast

from research_assistant.config import get_settings

P = ParamSpec("P")
R = TypeVar("R")

ObservationType = Literal[
    "generation",
    "embedding",
    "span",
    "agent",
    "tool",
    "chain",
    "retriever",
    "evaluator",
    "guardrail",
]


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return ``True`` iff Langfuse credentials are configured."""
    return get_settings().langfuse_enabled


# ---------------------------------------------------------------------------
# @observe decorator
# ---------------------------------------------------------------------------


def observe(
    *,
    name: str | None = None,
    as_type: ObservationType | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Tracing decorator with graceful no-op when Langfuse is disabled.

    Wraps the real :func:`langfuse.observe` lazily so the SDK is only
    imported (and therefore the client auth-checks only run) once we are
    sure credentials exist.

    Parameters mirror the upstream decorator. ``as_type`` uses the v4
    vocabulary ("agent" / "tool" / "retriever" / ...).
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cache: dict[str, Callable[P, R]] = {}

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not is_enabled():
                return func(*args, **kwargs)
            # Force-initialize the explicit Langfuse client from Settings
            # before the SDK's @observe tries to fetch one via
            # ``get_client()`` (which reads ``os.environ``).
            _client()
            decorated = cache.get("fn")
            if decorated is None:
                from langfuse import observe as _lf_observe

                # Langfuse's @observe has two overloads: (func) or
                # (None, *, name, as_type, ...). We pick the kw-only
                # variant so positional+kwargs dispatch cleanly.
                lf_decorator = _lf_observe(
                    name=name,
                    as_type=as_type,
                    capture_input=capture_input,
                    capture_output=capture_output,
                )
                decorated = lf_decorator(func)
                cache["fn"] = decorated
            return decorated(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Client access (lazy)
# ---------------------------------------------------------------------------


_CLIENT_CACHE: dict[str, Any] = {}


def _client() -> Any | None:
    """Return the singleton Langfuse client, or ``None`` when disabled.

    We instantiate the :class:`~langfuse.Langfuse` client explicitly with
    credentials drawn from :class:`Settings` rather than relying on
    ``langfuse.get_client()`` which only reads ``os.environ``.
    ``pydantic-settings`` loads ``.env`` into the ``Settings`` object
    *without* mirroring to ``os.environ``, so the SDK would otherwise
    see empty keys and disable itself even when ``.env`` is populated.

    Import is deferred so disabled runs don't trigger the SDK's auth
    warning at module load.
    """
    if not is_enabled():
        return None
    cached = _CLIENT_CACHE.get("client")
    if cached is not None:
        return cached

    from langfuse import Langfuse

    settings = get_settings()
    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    _CLIENT_CACHE["client"] = client
    return client


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def current_trace_id() -> str | None:
    """Return the active trace id, or ``None`` outside a trace / disabled."""
    client = _client()
    if client is None:
        return None
    try:
        trace_id = client.get_current_trace_id()
    except Exception:  # pragma: no cover — SDK internal error, stay safe
        return None
    return cast(str | None, trace_id)


def current_trace_url() -> str | None:
    """Return the browsable URL for the active trace, when available."""
    client = _client()
    if client is None:
        return None
    try:
        url = client.get_trace_url()
    except Exception:  # pragma: no cover
        return None
    return cast(str | None, url)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def update_span(
    *,
    input: Any | None = None,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
    level: Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"] | None = None,
) -> None:
    """Attach structured ``input`` / ``output`` / ``metadata`` to the
    current span. No-op when Langfuse is disabled."""
    client = _client()
    if client is None:
        return
    kwargs: dict[str, Any] = {}
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if metadata is not None:
        kwargs["metadata"] = metadata
    if level is not None:
        kwargs["level"] = level
    if kwargs:
        with contextlib.suppress(Exception):  # pragma: no cover
            client.update_current_span(**kwargs)


def update_generation(
    *,
    model: str | None = None,
    input: Any | None = None,
    output: Any | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Attach LLM-specific attributes (model / token usage / cost) to
    the current generation span. No-op when disabled."""
    client = _client()
    if client is None:
        return
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if input is not None:
        kwargs["input"] = input
    if output is not None:
        kwargs["output"] = output
    if usage_details is not None:
        kwargs["usage_details"] = usage_details
    if cost_details is not None:
        kwargs["cost_details"] = cost_details
    if metadata is not None:
        kwargs["metadata"] = metadata
    if kwargs:
        with contextlib.suppress(Exception):  # pragma: no cover
            client.update_current_generation(**kwargs)


def update_trace_io(
    *,
    input: Any | None = None,
    output: Any | None = None,
) -> None:
    """Set the top-level ``input`` / ``output`` on the current trace."""
    client = _client()
    if client is None:
        return
    with contextlib.suppress(Exception):  # pragma: no cover
        client.set_current_trace_io(input=input, output=output)


def flush() -> None:
    """Flush pending observations. Important at the end of CLI /
    script runs so traces reach the backend before the process exits."""
    client = _client()
    if client is None:
        return
    with contextlib.suppress(Exception):  # pragma: no cover
        client.flush()


@contextlib.contextmanager
def start_agent_span(
    name: str,
    *,
    input: Any | None = None,
) -> Iterator[None]:
    """Context manager that opens an ``agent`` span around a block.

    Useful where the ``@observe`` decorator can't be applied — e.g. a
    Streamlit handler that needs to stream LangGraph events inside a
    single Langfuse trace. No-op when Langfuse is disabled.
    """
    client = _client()
    if client is None:
        yield
        return
    ctx = client.start_as_current_observation(as_type="agent", name=name, input=input)
    with ctx:
        yield
