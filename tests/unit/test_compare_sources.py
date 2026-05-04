from __future__ import annotations

import pytest

from research_assistant.config import get_settings
from research_assistant.graph.state import ConflictItem, Draft, Evidence, SearchHit, SubQuestion
from research_assistant.tools.compare_sources import (
    build_conflict_report,
    extract_quantities,
    heuristic_compare_sources,
)


def _hit(url: str, snippet: str) -> SearchHit:
    return SearchHit(
        url=f"https://{url}",  # type: ignore[arg-type]
        title="t",
        snippet=snippet,
        source="web",
    )


def test_extract_quantities_finds_percent_and_count() -> None:
    text = "Model A reaches 92.5% accuracy while batch size is 64."
    got = extract_quantities(text)
    assert any(abs(v - 92.5) < 1e-6 and u == "pct" for v, u, _ in got)
    assert any(v == 64 and u == "_unitless" for v, u, _ in got)


def test_heuristic_flags_cross_url_percent_mismatch() -> None:
    evs = [
        Evidence(
            ref_label="ev_sq_1_1",
            sub_question_id="sq_1",
            hit=_hit("a.com", "Peak accuracy 95% on the benchmark."),
        ),
        Evidence(
            ref_label="ev_sq_1_2",
            sub_question_id="sq_1",
            hit=_hit("b.org", "Reported top score 72% same task."),
        ),
    ]
    items = heuristic_compare_sources(evs)
    assert len(items) >= 1
    assert items[0].detection == "heuristic"
    assert set(items[0].involved_ref_labels) == {"ev_sq_1_1", "ev_sq_1_2"}


def test_build_conflict_report_off() -> None:
    sq = SubQuestion(id="sq_1", question="What is X?")
    draft = Draft(sub_question_id="sq_1", content="Answer [^1].", model="m")
    evs = [
        Evidence(
            ref_label="ev_sq_1_1", sub_question_id="sq_1", hit=_hit("a.com", "One 10% claim.")
        ),
    ]
    report, cost = build_conflict_report(sub_q=sq, evidence=evs, draft=draft, mode="off")
    assert report.mode_used == "off"
    assert report.items == []
    assert cost == 0.0


def test_build_auto_skips_llm_when_non_comparative_and_clean_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMPARE_SOURCES_MODE", "auto")
    get_settings.cache_clear()
    sq = SubQuestion(id="sq_1", question="Define the term entropy.")  # factual
    draft = Draft(sub_question_id="sq_1", content="Entropy is ... [^1].", model="m")
    evs = [
        Evidence(
            ref_label="ev_sq_1_1",
            sub_question_id="sq_1",
            hit=_hit("a.com", "The value is 42 units."),
        ),
    ]

    called: dict[str, bool] = {"llm": False}

    def _boom(*_a: object, **_k: object) -> object:
        called["llm"] = True
        raise AssertionError("invoke_structured_llm should not run")

    monkeypatch.setattr(
        "research_assistant.tools.compare_sources.invoke_structured_llm",
        _boom,
    )
    report, cost = build_conflict_report(sub_q=sq, evidence=evs, draft=draft)
    assert called["llm"] is False
    assert report.mode_used == "heuristic"
    assert cost == 0.0


def test_build_auto_runs_llm_on_comparative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPARE_SOURCES_MODE", "auto")
    get_settings.cache_clear()
    sq = SubQuestion(id="sq_1", question="Compare approach A vs B for latency.")
    draft = Draft(sub_question_id="sq_1", content="A is faster [^1].", model="m")
    evs = [
        Evidence(
            ref_label="ev_sq_1_1", sub_question_id="sq_1", hit=_hit("a.com", "Latency 10 ms.")
        ),
        Evidence(
            ref_label="ev_sq_1_2", sub_question_id="sq_1", hit=_hit("b.com", "Latency 12 ms.")
        ),
    ]

    def _stub(**_kwargs: object) -> tuple[list[ConflictItem], float, str, int, int]:
        return (
            [
                ConflictItem(
                    summary="Sources disagree on exact latency.",
                    severity="low",
                    involved_ref_labels=["ev_sq_1_1", "ev_sq_1_2"],
                    detection="llm",
                    detail="10 vs 12 ms",
                ),
            ],
            0.0001,
            "claude-sonnet-4-5",
            10,
            20,
        )

    monkeypatch.setattr(
        "research_assistant.tools.compare_sources._llm_refine_conflicts",
        _stub,
    )
    report, cost = build_conflict_report(
        sub_q=sq,
        evidence=evs,
        draft=draft,
        cost_before=0.0,
        per_query_cap_usd=0.30,
    )
    assert report.mode_used in {"heuristic+llm", "llm"}
    assert len(report.items) == 1
    assert report.items[0].detection == "llm"
    assert cost == pytest.approx(0.0001)
