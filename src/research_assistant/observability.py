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

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cache: dict[str, Callable[P, R]] = {}

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not is_enabled():
                return func(*args, **kwargs)

            _client()
            decorated = cache.get("fn")
            if decorated is None:
                from langfuse import observe as _lf_observe

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
    client = _client()
    if client is None:
        return None
    try:
        trace_id = client.get_current_trace_id()
    except Exception:  # pragma: no cover — SDK internal error, stay safe
        return None
    return cast(str | None, trace_id)


def current_trace_url() -> str | None:
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
    client = _client()
    if client is None:
        return
    with contextlib.suppress(Exception):  # pragma: no cover
        client.set_current_trace_io(input=input, output=output)


def flush() -> None:
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

    client = _client()
    if client is None:
        yield
        return
    ctx = client.start_as_current_observation(as_type="agent", name=name, input=input)
    with ctx:
        yield
