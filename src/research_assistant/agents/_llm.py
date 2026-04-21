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
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage

from research_assistant.config import get_settings

logger = logging.getLogger(__name__)


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


def invoke_llm(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system: str | None = None,
    current_cost_usd: float = 0.0,
    per_query_cap_usd: float | None = None,
) -> LLMCallResult:
    """Invoke the chat model and return a cost-annotated result.

    Enforces the per-query budget cap **before** the call by estimating
    worst-case cost (``max_tokens`` output at the model's rate) plus the
    already-accumulated ``current_cost_usd``. When the projected total
    would exceed ``per_query_cap_usd`` we raise :class:`BudgetExceededError`
    without contacting the provider — per ADR-011.
    """
    cap = per_query_cap_usd if per_query_cap_usd is not None else get_settings().per_query_cap_usd

    # Conservative pre-flight estimate: assume max_tokens output, ignoring
    # prompt length (which is bounded by Anthropic context limits anyway).
    projected_worst = current_cost_usd + estimate_cost_usd(
        model,
        tokens_in=len(prompt) // 3,  # rough char→token ratio, only for guard
        tokens_out=max_tokens,
    )
    if projected_worst > cap:
        raise BudgetExceededError(
            f"Refusing call to {model}: projected worst-case ${projected_worst:.4f} "
            f"would exceed per-query cap ${cap:.2f} "
            f"(already spent ${current_cost_usd:.4f}).",
        )

    chat = build_chat_model(
        model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    messages: list[BaseMessage | dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = chat.invoke(messages)

    tokens_in, tokens_out = _extract_usage(response)
    cost = estimate_cost_usd(model, tokens_in, tokens_out)

    # Normalise content to a flat string (LangChain can return list-of-blocks).
    content = response.content
    if isinstance(content, list):
        text_parts = [
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        ]
        text = "".join(text_parts)
    else:
        text = str(content)

    return LLMCallResult(
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        model=model,
    )
