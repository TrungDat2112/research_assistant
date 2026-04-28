"""Rule-based tool routing for retrieval (PLAN stage-1 router, ADR-027).

Maps sub-question text (+ optional rationale) to a heuristic **intent**, then to
an ordered tool list (≤ ``Settings.tool_router_max_tools``) drawn from:

* ``vector_search`` — seeded Chroma corpus
* ``web_search`` — Tavily (via injected ``web_search_with_fallback`` in graph)
* ``academic_search`` — arXiv metadata

The graph retriever merges and de-duplicates hits in tool order until the
candidate pool is full — no LLM in the router for v1.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["factual", "academic", "comparative", "internal_corpus"]

ToolId = Literal["vector_search", "web_search", "academic_search"]

_INTENT_ORDER: dict[Intent, tuple[ToolId, ...]] = {
    # Prefer fresh/open web; corpus as back-fill.
    "factual": ("web_search", "vector_search"),
    # Abstracts then corpus then web snippets.
    "academic": ("academic_search", "vector_search", "web_search"),
    # Corpus + literature + web for contradiction coverage.
    "comparative": ("vector_search", "academic_search", "web_search"),
    # Indexed corpus first; optional web gap-fill only if pool unfilled.
    "internal_corpus": ("vector_search", "web_search"),
}

_COMPARATIVE = re.compile(
    r"\b("
    r"vs\.?|versus|compare|comparison|compared\s+to|difference\s+between"
    r"|so\s+s[áa]nh|kh[áa]c\s+nhau|hay\s+l[àa]"
    r")\b",
    re.IGNORECASE,
)
_INTERNAL_CORPUS = re.compile(
    r"\b("
    r"within\s+(our\s+)?corpus|seed\s+corpus|vector\s+store|chroma(collection)?"
    r"|embedding\s+(index|store)|indexed\s+(documents?|corpus)|\bour\s+indexed\b"
    r"|bm25\s+index|trong\s+corpus|corpus\s+n[ộo]i\s+b[ộo]|(?:đ[aã])\s+(?:ược\s+)?index"
    r")\b",
    flags=re.IGNORECASE,
)
_ACADEMIC = re.compile(
    r"\b("
    r"arxiv|pre-?print|peer\s*review|benchmark|baseline|fine-?tun|pretrain"
    r"|transformer[s]?|BERT|GPT|SOTA|\bLlama\b|Mistral|\bBERT\b|citation|\bdoi\b"
    r"|architecture|ablat|nlp|paper\s+survey|survey\s+paper|empirical\s+study"
    r"|nghiên\s*cứu|bài\s+báo|h[aọ]c\s+thu[aậ]t|\blayer[s]?\s+norm\b"
    r")\b",
    flags=re.IGNORECASE,
)


RETRIEVAL_TOOL_IDS: frozenset[str] = frozenset({"vector_search", "web_search", "academic_search"})


def sanitize_planner_suggested_tools(raw: list[str] | None) -> list[str]:
    """Keep only known retrieval tool ids, first-seen order; default ``web_search``.

    Used for :class:`~research_assistant.graph.state.SubQuestion.suggested_tools`
    and for comparing against the rule-based router.
    """
    if not raw:
        return ["web_search"]
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if t not in RETRIEVAL_TOOL_IDS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out if out else ["web_search"]


def retrieval_tool_plan_differs_from_planner(
    planner_tools: list[str],
    router_ordered_tools: list[str],
) -> bool:
    """True when sanitized planner order differs from the router's order (execution uses router)."""
    return sanitize_planner_suggested_tools(planner_tools) != list(router_ordered_tools)


class ToolPlan(BaseModel):
    """Concrete tool order derived from heuristic intent."""

    intent: Intent
    ordered_tools: list[str] = Field(
        ...,
        min_length=1,
        description="Tool ids to invoke in sequence (dedup preserved order).",
    )

    @property
    def primary(self) -> str:
        return self.ordered_tools[0]

    @property
    def optional(self) -> list[str]:
        return list(self.ordered_tools[1:])


def classify_intent(question: str, rationale: str = "") -> Intent:
    """Assign a deterministic intent bucket from wording (EN + VI heuristics).

    Priority: comparative → internal corpus → academic → factual (default).
    """
    blob = f"{question} {rationale}".strip().lower()
    if _COMPARATIVE.search(blob):
        return "comparative"
    if _INTERNAL_CORPUS.search(blob):
        return "internal_corpus"
    if _ACADEMIC.search(blob):
        return "academic"
    return "factual"


def build_tool_plan(intent: Intent, *, max_tools: int) -> ToolPlan:
    """Select up to ``max_tools`` tools for ``intent`` (stable order within intent)."""
    bounded = max(1, max_tools)
    raw = list(_INTENT_ORDER[intent])[:bounded]
    return ToolPlan(intent=intent, ordered_tools=list(raw))


def plan_for_sub_question(
    question: str,
    rationale: str,
    *,
    max_tools: int,
) -> ToolPlan:
    """End-to-end: classify then build the :class:`ToolPlan`."""
    return build_tool_plan(classify_intent(question, rationale), max_tools=max_tools)
