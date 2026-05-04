from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict, TypeVar

from pydantic import BaseModel, Field, HttpUrl

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")

class SubQuestion(BaseModel):
    id: str = Field(..., pattern=r"^sq_\d+$", description="Sub-question id, e.g. 'sq_1'.")
    question: str = Field(..., min_length=3)
    rationale: str = Field(default="", description="Why this sub-question matters.")
    suggested_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Advisory tool ids (vector_search, web_search, academic_search). "
            "Sanitized at plan build; rule-based router order wins at retrieval time."
        ),
    )
    dependency_ids: list[str] = Field(default_factory=list)


WebTrustTier = Literal["high", "medium", "low"]


class SearchHit(BaseModel):
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
    web_trust_tier: WebTrustTier | None = Field(
        default=None,
        description="Heuristic trust tier for open-web URLs — set by "
        "``web_search`` for router prioritisation; left None for academic / "
        "vector / corpus hits.",
    )


class Evidence(BaseModel):
    ref_label: str = Field(..., pattern=r"^ev_sq_\d+_\d+$")
    sub_question_id: str = Field(..., pattern=r"^sq_\d+$")
    hit: SearchHit
    used: bool = Field(
        default=False,
        description="Flipped True after Synthesizer cites this evidence.",
    )


class Citation(BaseModel):
    marker: int = Field(..., ge=1)
    ref_label: str = Field(..., pattern=r"^ev_sq_\d+_\d+$")


class Draft(BaseModel):
    sub_question_id: str = Field(..., pattern=r"^sq_\d+$")
    content: str = Field(..., description="Markdown answer with inline [^N] refs.")
    citations: list[Citation] = Field(default_factory=list)
    model: str = Field(..., description="LLM model id used to produce the draft.")
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


ConflictSeverity = Literal["low", "medium", "high"]


class ConflictItem(BaseModel):
    summary: str = Field(..., min_length=3, description="Short description of the conflict.")
    severity: ConflictSeverity = Field(
        ...,
        description="Heuristic severities are conservative; LLM may downgrade with rationale.",
    )
    involved_ref_labels: list[str] = Field(
        default_factory=list,
        description="Evidence ref_labels (ev_sq_*_*) touched by the disagreement.",
    )
    detection: Literal["heuristic", "llm"] = Field(
        default="heuristic",
        description="Whether this row originated from regex/number rules or Sonnet structured pass.",
    )
    detail: str = Field(
        default="",
        description="Optional excerpt: numbers, spans, or judge note (keep short for traces).",
    )


class ConflictReport(BaseModel):
    sub_question_id: str = Field(..., pattern=r"^sq_\d+$")
    mode_used: Literal["off", "heuristic", "heuristic+llm", "llm"] = Field(
        default="off",
        description="Which detection path produced the (possibly empty) item list.",
    )
    items: list[ConflictItem] = Field(default_factory=list)


class Critique(BaseModel):
    sub_question_id: str = Field(..., pattern=r"^sq_\d+$")
    passed: bool
    forced_pass: bool = Field(
        default=False,
        description="True when attempt budget exhausted but the graph must advance.",
    )
    overall_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Holistic LLM score (legacy / summary; gates use faithfulness completeness).",
    )
    faithfulness_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="LLM: claims grounded in cited evidence (1-5).",
    )
    completeness_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="LLM: sub-question scope fully covered (1-5).",
    )
    consistency_score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Deterministic from ConflictReport severities (1-5).",
    )
    paragraph_citation_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deterministic share of substantive paragraphs containing [^N].",
    )
    addresses_sub_question: bool
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    conflicts: list[ConflictItem] = Field(
        default_factory=list,
        description="Cross-source conflicts detected pre-Critic ; informs consistency review.",
    )
    model: str
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class StepLog(BaseModel):
    node: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    duration_ms: float = Field(default=0.0, ge=0.0)
    status: Literal["ok", "error", "skipped"] = "ok"
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


def _extend_list(left: list[_T], right: list[_T]) -> list[_T]:

    return [*left, *right]


def _merge_dict(left: dict[_K, _V], right: dict[_K, _V]) -> dict[_K, _V]:
    return {**left, **right}


class ResearchState(TypedDict, total=False):
    query: str
    output_language: Literal["vi", "en"]

    plan: Annotated[list[SubQuestion], _extend_list]

    evidence: Annotated[dict[str, list[Evidence]], _merge_dict]

    drafts: Annotated[dict[str, Draft], _merge_dict]

    critiques: Annotated[dict[str, Critique], _merge_dict]
    conflict_reports: Annotated[dict[str, ConflictReport], _merge_dict]
    critic_attempts: Annotated[dict[str, int], _merge_dict]
    synth_critic_feedback: str | None
    critic_route_next: Literal["retriever", "tick"] | None

    iterations: int
    max_iterations: int
    current_sub_question_index: int

    max_iterations_reached: bool

    critic_enabled_override: bool | None

    tool_router_enabled_override: bool | None

    compare_sources_mode_override: Literal["off", "heuristic", "auto"] | None

    final_report: str | None

    trace: Annotated[list[StepLog], _extend_list]
    total_cost_usd: float
    per_query_cap_usd: float
    trace_id: str | None
    trace_url: str | None


def new_state(
    query: str,
    *,
    output_language: Literal["vi", "en"] = "vi",
    max_iterations: int = 8,
    per_query_cap_usd: float = 0.30,
) -> ResearchState:
    return ResearchState(
        query=query,
        output_language=output_language,
        plan=[],
        evidence={},
        drafts={},
        critiques={},
        conflict_reports={},
        critic_attempts={},
        synth_critic_feedback=None,
        critic_route_next=None,
        iterations=0,
        max_iterations=max_iterations,
        current_sub_question_index=0,
        max_iterations_reached=False,
        critic_enabled_override=None,
        tool_router_enabled_override=None,
        compare_sources_mode_override=None,
        final_report=None,
        trace=[],
        total_cost_usd=0.0,
        per_query_cap_usd=per_query_cap_usd,
        trace_id=None,
        trace_url=None,
    )
