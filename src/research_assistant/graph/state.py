"""State schema for the LangGraph research pipeline.

Follows PLAN.md §6.1. ``ResearchState`` is a ``TypedDict`` (required by
LangGraph reducers); richer objects inside the state (sub-questions,
evidence, drafts, traces) are Pydantic models so we get validation,
``model_dump``/``model_validate_json`` round-tripping, and clean diffs in
Langfuse traces.

Design notes:
  * Every ID uses a short ``sq_<n>`` / ``ev_<sq>_<m>`` scheme so citations
    ``[^N]`` in Synthesizer output stay readable.
  * ``evidence`` is keyed by ``sub_question_id`` to keep lookup O(1) while
    the graph iterates through sub-questions.
  * ``trace`` captures every node invocation (timings, cost) so the
    Reporter and tests can reconstruct the reasoning path deterministically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict, TypeVar

from pydantic import BaseModel, Field, HttpUrl

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A single sub-question planned by the Planner agent.

    ``suggested_tools`` is advisory — the Tool Router may override based on
    availability / budget. ``dependency_ids`` lists prior sub-questions that
    must be answered first (``[]`` means independent).
    """

    id: str = Field(..., pattern=r"^sq_\d+$", description="Sub-question id, e.g. 'sq_1'.")
    question: str = Field(..., min_length=3)
    rationale: str = Field(default="", description="Why this sub-question matters.")
    suggested_tools: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    """Single result from a search tool (Tavily, arXiv, vector store, ...).

    Kept intentionally provider-agnostic so RAG + web + academic search can
    all emit ``SearchHit`` and feed the same Synthesizer pipeline.
    """

    url: HttpUrl
    title: str
    snippet: str = Field(..., description="Text snippet returned by the provider.")
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    published_date: str | None = None
    source: Literal["web", "academic", "vector", "corpus"] = "web"
    raw_content: str | None = Field(
        default=None,
        description="Full content when the provider supplied it (Tavily "
        "`include_raw_content=True`). Kept optional to bound context size.",
    )


class Evidence(BaseModel):
    """A piece of evidence attached to a sub-question.

    One ``SearchHit`` becomes one ``Evidence`` after selection; the Synthesizer
    cites it via its ``ref_label`` which becomes ``[^N]`` in the final report.
    """

    ref_label: str = Field(..., pattern=r"^ev_sq_\d+_\d+$")
    sub_question_id: str = Field(..., pattern=r"^sq_\d+$")
    hit: SearchHit
    used: bool = Field(
        default=False,
        description="Flipped True after Synthesizer cites this evidence.",
    )


class Citation(BaseModel):
    """A single ``[^N]`` footnote emitted in a draft."""

    marker: int = Field(..., ge=1)
    ref_label: str = Field(..., pattern=r"^ev_sq_\d+_\d+$")


class Draft(BaseModel):
    """Synthesizer output for a single sub-question."""

    sub_question_id: str = Field(..., pattern=r"^sq_\d+$")
    content: str = Field(..., description="Markdown answer with inline [^N] refs.")
    citations: list[Citation] = Field(default_factory=list)
    model: str = Field(..., description="LLM model id used to produce the draft.")
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class Critique(BaseModel):
    """Critic verdict for one sub-question draft (Week 2 draft — LLM + deterministic checks)."""

    sub_question_id: str = Field(..., pattern=r"^sq_\d+$")
    passed: bool
    forced_pass: bool = Field(
        default=False,
        description="True when attempt budget exhausted but the graph must advance.",
    )
    overall_score: int = Field(..., ge=1, le=5)
    paragraph_citation_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deterministic share of substantive paragraphs containing [^N].",
    )
    addresses_sub_question: bool
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    model: str
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class StepLog(BaseModel):
    """One entry in the reasoning trace — emitted by every graph node."""

    node: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    duration_ms: float = Field(default=0.0, ge=0.0)
    status: Literal["ok", "error", "skipped"] = "ok"
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reducers — used by LangGraph to merge node outputs into state
# ---------------------------------------------------------------------------


def _extend_list(left: list[_T], right: list[_T]) -> list[_T]:
    """Append-only reducer for lists (trace, plan, drafts, evidence).

    LangGraph nodes return *partial* state updates; without a reducer the
    default is "replace", which would wipe previously appended items. We
    want append semantics everywhere except scalar fields.
    """
    return [*left, *right]


def _merge_dict(left: dict[_K, _V], right: dict[_K, _V]) -> dict[_K, _V]:
    """Shallow-merge reducer for dict-valued state fields."""
    return {**left, **right}


# ---------------------------------------------------------------------------
# Top-level graph state
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    ``total=False`` lets individual nodes return partial updates; LangGraph
    applies the annotated reducer (or "replace" by default) per key.
    """

    # Input --------------------------------------------------------------
    query: str
    output_language: Literal["vi", "en"]

    # Planning -----------------------------------------------------------
    plan: Annotated[list[SubQuestion], _extend_list]

    # Retrieval ----------------------------------------------------------
    evidence: Annotated[dict[str, list[Evidence]], _merge_dict]
    # key = sub_question_id  →  list of Evidence for that sub-question.

    # Synthesis ----------------------------------------------------------
    drafts: Annotated[dict[str, Draft], _merge_dict]
    # key = sub_question_id  →  Draft.

    # Critique -----------------------------------------------------------
    critiques: Annotated[dict[str, Critique], _merge_dict]
    critic_attempts: Annotated[dict[str, int], _merge_dict]
    synth_critic_feedback: str | None
    critic_route_next: Literal["retriever", "tick"] | None

    # Loop control -------------------------------------------------------
    iterations: int
    max_iterations: int
    current_sub_question_index: int
    # When True, the graph stopped early because ``iterations`` hit
    # ``max_iterations`` before every sub-question completed (set in reporter).
    max_iterations_reached: bool
    # If set, overrides :attr:`Settings.critic_enabled` for this run only (CLI).
    critic_enabled_override: bool | None

    # Reporting ----------------------------------------------------------
    final_report: str | None

    # Observability & cost -----------------------------------------------
    trace: Annotated[list[StepLog], _extend_list]
    total_cost_usd: float
    per_query_cap_usd: float
    # Langfuse trace identifiers — populated by the first decorated node
    # once a trace is active. ``None`` when Langfuse is disabled.
    trace_id: str | None
    trace_url: str | None


def new_state(
    query: str,
    *,
    output_language: Literal["vi", "en"] = "vi",
    max_iterations: int = 8,
    per_query_cap_usd: float = 0.30,
) -> ResearchState:
    """Factory for a fresh :class:`ResearchState` with sensible defaults."""
    return ResearchState(
        query=query,
        output_language=output_language,
        plan=[],
        evidence={},
        drafts={},
        critiques={},
        critic_attempts={},
        synth_critic_feedback=None,
        critic_route_next=None,
        iterations=0,
        max_iterations=max_iterations,
        current_sub_question_index=0,
        max_iterations_reached=False,
        critic_enabled_override=None,
        final_report=None,
        trace=[],
        total_cost_usd=0.0,
        per_query_cap_usd=per_query_cap_usd,
        trace_id=None,
        trace_url=None,
    )
