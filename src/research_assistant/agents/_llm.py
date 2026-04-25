"""Shared LLM utilities for agent nodes.

Centralises: model instantiation (Anthropic via ``langchain-anthropic``),
token-usage extraction, USD cost estimation, and the per-query hard cap
from ADR-011. Kept internal (leading underscore) — only ``agents.*``
modules should import it.

Pricing table is a conservative snapshot; the source of truth remains
``https://www.anthropic.com/pricing``. When a new model is introduced,
add its rate here and default agents to it via ``config.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from research_assistant.config import get_settings
from research_assistant.observability import observe, update_generation

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

logger = logging.getLogger(__name__)

_EPHEMERAL_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


# (input_usd_per_mtok, output_usd_per_mtok)
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (3.00, 15.00),
    # Haiku 4.5 pricing is a conservative estimate; update when confirmed.
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}
_FALLBACK_PRICING: tuple[float, float] = (3.00, 15.00)  # worst-case for safety


class BudgetExceededError(RuntimeError):
    """Raised when completing a call would breach the per-query USD cap."""


@dataclass(frozen=True)
class LLMCallResult:
    """Outcome of a single LLM invocation."""

    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str


def _pricing_for(model: str) -> tuple[float, float]:
    if model in _PRICING_USD_PER_MTOK:
        return _PRICING_USD_PER_MTOK[model]
    logger.warning(
        "No pricing entry for %r — using conservative fallback %s; "
        "update agents._llm._PRICING_USD_PER_MTOK.",
        model,
        _FALLBACK_PRICING,
    )
    return _FALLBACK_PRICING


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return USD cost for a call. Accepts zero-token cases gracefully."""
    price_in, price_out = _pricing_for(model)
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


def _extract_usage(message: BaseMessage) -> tuple[int, int]:
    """Pull ``(input_tokens, output_tokens)`` from a LangChain ``AIMessage``.

    LangChain standardises ``usage_metadata`` across providers (>=0.3); we
    fall back to ``response_metadata`` if a provider bypasses it.
    """
    if isinstance(message, AIMessage):
        usage = message.usage_metadata
        if usage is not None:
            return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        meta = message.response_metadata or {}
        usage_raw = cast(dict[str, Any], meta.get("usage") or {})
        return (
            int(usage_raw.get("input_tokens", 0)),
            int(usage_raw.get("output_tokens", 0)),
        )
    return 0, 0


def build_chat_model(
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout_seconds: float = 60.0,
) -> ChatAnthropic:
    """Construct a ``ChatAnthropic`` client with credentials from settings."""
    settings = get_settings()
    if not settings.anthropic_api_key.get_secret_value():
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Populate `.env` before invoking an agent.",
        )
    return ChatAnthropic(
        model_name=model,
        temperature=temperature,
        max_tokens_to_sample=max_tokens,
        timeout=timeout_seconds,
        api_key=settings.anthropic_api_key,
        stop=None,
    )


def _preflight_budget_check(
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    current_cost_usd: float,
    per_query_cap_usd: float | None,
) -> None:
    """Raise :class:`BudgetExceededError` before contacting the provider
    when the projected worst-case spend would breach the per-query cap.

    Approximation is deliberately conservative (uses ``max_tokens`` for
    the output side and a coarse char→token ratio for input) so the cap
    is never silently overshot.
    """
    cap = per_query_cap_usd if per_query_cap_usd is not None else get_settings().per_query_cap_usd
    projected_worst = current_cost_usd + estimate_cost_usd(
        model,
        tokens_in=len(prompt) // 3,
        tokens_out=max_tokens,
    )
    if projected_worst > cap:
        raise BudgetExceededError(
            f"Refusing call to {model}: projected worst-case ${projected_worst:.4f} "
            f"would exceed per-query cap ${cap:.2f} "
            f"(already spent ${current_cost_usd:.4f}).",
        )


def _normalise_text(content: Any) -> str:
    """Flatten LangChain content (str or list-of-blocks) into plain text."""
    if isinstance(content, list):
        parts = [
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content)


def _use_prompt_cache(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return get_settings().anthropic_prompt_cache_enabled


def build_lc_messages(
    *,
    system: str | None,
    user_text: str,
    cacheable_user_prefix: str | None = None,
    use_prompt_cache: bool | None = None,
) -> list[BaseMessage]:
    """Build LangChain messages for Anthropic, optional ephemeral cache breakpoints.

    Caches the **system** block and an optional **user** prefix (same research
    query repeated on every Critic call). Variable evidence + sub-question tail
    stays in the second user text block so it is not read from cache.
    """
    use_cache = _use_prompt_cache(use_prompt_cache)
    out: list[BaseMessage] = []
    if system:
        if use_cache:
            out.append(
                SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": _EPHEMERAL_CACHE_CONTROL,
                        },
                    ],
                ),
            )
        else:
            out.append(SystemMessage(content=system))

    prefix = (cacheable_user_prefix or "").strip()
    if prefix and use_cache:
        out.append(
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": cacheable_user_prefix,
                        "cache_control": _EPHEMERAL_CACHE_CONTROL,
                    },
                    {"type": "text", "text": user_text},
                ],
            ),
        )
    elif prefix:
        out.append(HumanMessage(content=f"{cacheable_user_prefix}\n\n{user_text}"))
    else:
        out.append(HumanMessage(content=user_text))
    return out


def _budget_prompt_chars(
    *,
    system: str | None,
    user_text: str,
    cacheable_user_prefix: str | None,
) -> str:
    """Concatenate prompt parts for conservative preflight token estimation."""
    parts = [p for p in (system, cacheable_user_prefix, user_text) if p]
    return "\n\n".join(parts)


@observe(name="anthropic.invoke", as_type="generation", capture_input=False, capture_output=False)
def invoke_llm(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system: str | None = None,
    cacheable_user_prefix: str | None = None,
    use_prompt_cache: bool | None = None,
    current_cost_usd: float = 0.0,
    per_query_cap_usd: float | None = None,
) -> LLMCallResult:
    """Invoke the chat model and return a cost-annotated result.

    Enforces the per-query budget cap **before** the call (see
    :func:`_preflight_budget_check`), per ADR-011.

    Emits a Langfuse ``generation`` span with model + usage + cost so
    Langfuse native charts pick it up. Prompt text is attached manually
    instead of via ``capture_input=True`` to keep secrets/system prompts
    out of default captures if callers pre-redact them.
    """
    budget_text = _budget_prompt_chars(
        system=system,
        user_text=prompt,
        cacheable_user_prefix=cacheable_user_prefix,
    )
    _preflight_budget_check(
        model,
        budget_text,
        max_tokens=max_tokens,
        current_cost_usd=current_cost_usd,
        per_query_cap_usd=per_query_cap_usd,
    )

    chat = build_chat_model(model, temperature=temperature, max_tokens=max_tokens)
    messages = build_lc_messages(
        system=system,
        user_text=prompt,
        cacheable_user_prefix=cacheable_user_prefix,
        use_prompt_cache=use_prompt_cache,
    )

    response = chat.invoke(messages)
    tokens_in, tokens_out = _extract_usage(response)
    cost = estimate_cost_usd(model, tokens_in, tokens_out)

    text = _normalise_text(response.content)
    update_generation(
        model=model,
        input={
            "system": system,
            "prompt": prompt,
            "cacheable_user_prefix": cacheable_user_prefix,
            "prompt_cache": _use_prompt_cache(use_prompt_cache),
        },
        output=text,
        usage_details={"input": tokens_in, "output": tokens_out},
        cost_details={"input": cost, "total": cost},
    )

    return LLMCallResult(
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        model=model,
    )


@observe(
    name="anthropic.invoke_structured",
    as_type="generation",
    capture_input=False,
    capture_output=False,
)
def invoke_structured_llm(
    model: str,
    prompt: str,
    schema: type[_SchemaT],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system: str | None = None,
    cacheable_user_prefix: str | None = None,
    use_prompt_cache: bool | None = None,
    current_cost_usd: float = 0.0,
    per_query_cap_usd: float | None = None,
) -> tuple[_SchemaT, LLMCallResult]:
    """Invoke the chat model with a strict Pydantic schema and return
    ``(parsed_instance, cost_result)``.

    Uses ``ChatAnthropic.with_structured_output(schema, include_raw=True)``
    so Anthropic's native tool-use machinery guarantees a shape-valid
    JSON response — replacing the fragile "ask for JSON and hope" path
    used pre-fix. ``include_raw=True`` returns a dict with ``raw``,
    ``parsed`` and ``parsing_error`` keys so we can still recover token
    usage (which lives on the raw ``AIMessage``) after parsing.
    """
    budget_text = _budget_prompt_chars(
        system=system,
        user_text=prompt,
        cacheable_user_prefix=cacheable_user_prefix,
    )
    _preflight_budget_check(
        model,
        budget_text,
        max_tokens=max_tokens,
        current_cost_usd=current_cost_usd,
        per_query_cap_usd=per_query_cap_usd,
    )

    chat = build_chat_model(model, temperature=temperature, max_tokens=max_tokens)
    structured = chat.with_structured_output(schema, include_raw=True)

    messages = build_lc_messages(
        system=system,
        user_text=prompt,
        cacheable_user_prefix=cacheable_user_prefix,
        use_prompt_cache=use_prompt_cache,
    )

    response: dict[str, Any] = structured.invoke(messages)  # type: ignore[assignment]
    raw = response.get("raw")
    parsed = response.get("parsed")
    parsing_error = response.get("parsing_error")

    if parsed is None:
        raise RuntimeError(
            f"Structured output returned no parsed value (parsing_error={parsing_error!r}).",
        )

    if isinstance(raw, BaseMessage):
        tokens_in, tokens_out = _extract_usage(raw)
    else:
        tokens_in = tokens_out = 0
    cost = estimate_cost_usd(model, tokens_in, tokens_out)

    text = _normalise_text(raw.content) if isinstance(raw, BaseMessage) else ""
    parsed_payload: Any = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
    update_generation(
        model=model,
        input={
            "system": system,
            "prompt": prompt,
            "schema": schema.__name__,
            "cacheable_user_prefix": cacheable_user_prefix,
            "prompt_cache": _use_prompt_cache(use_prompt_cache),
        },
        output=parsed_payload,
        usage_details={"input": tokens_in, "output": tokens_out},
        cost_details={"input": cost, "total": cost},
    )

    return cast(_SchemaT, parsed), LLMCallResult(
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        model=model,
    )
