"""Streamlit UI for the Research Assistant Agent (Week 1 — tối giản).

Run with::

    uv run streamlit run ui/app.py

Exposes a single-page UI: user types a research question, clicks "Run",
watches the trace stream in real time, and reads the rendered Markdown
report. Not production-grade — meant for dev-loop visibility per
PLAN.md §10 (Week 1 exit criteria).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import streamlit as st

from research_assistant.config import get_settings
from research_assistant.graph.research_graph import build_graph
from research_assistant.graph.state import ResearchState, new_state
from research_assistant.observability import (
    current_trace_url,
    start_agent_span,
    update_trace_io,
)
from research_assistant.observability import (
    flush as lf_flush,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon=":books:",
    layout="wide",
)

st.title("Research Assistant Agent")
st.caption("Tuần 1 skeleton — planner + web_search + synthesizer + reporter.")

settings = get_settings()

with st.sidebar:
    st.header("Cấu hình")
    language = st.radio(
        "Ngôn ngữ báo cáo",
        options=("vi", "en"),
        index=0 if settings.output_language == "vi" else 1,
        horizontal=True,
    )
    max_iters = st.slider(
        "Max iterations",
        min_value=1,
        max_value=16,
        value=settings.max_iterations,
    )

    st.divider()
    st.subheader("Credentials")
    st.write(
        {
            "ANTHROPIC_API_KEY": "✓" if settings.has_llm_credentials else "✗ missing",
            "TAVILY_API_KEY": "✓" if settings.has_search_credentials else "✗ missing",
            "LANGFUSE": "on" if settings.langfuse_enabled else "off",
        },
    )
    st.subheader("Budget (ADR-011)")
    st.write(
        {
            "per_query_cap_usd": settings.per_query_cap_usd,
            "max_budget_usd": settings.max_budget_usd,
        },
    )


query = st.text_area(
    "Câu hỏi nghiên cứu",
    placeholder="Ví dụ: So sánh LoRA và QLoRA cho fine-tuning LLM năm 2025 ...",
    height=120,
)
run_btn = st.button("Chạy", type="primary", disabled=not query.strip())

if run_btn:
    if not settings.has_llm_credentials or not settings.has_search_credentials:
        st.error(
            "Thiếu API key. Điền `ANTHROPIC_API_KEY` và `TAVILY_API_KEY` vào `.env` "
            "rồi restart lại Streamlit.",
        )
        st.stop()

    initial = new_state(
        query=query.strip(),
        output_language=cast(Any, language),
        max_iterations=max_iters,
        per_query_cap_usd=settings.per_query_cap_usd,
    )

    graph = build_graph()

    status_panel = st.status("Đang chạy agent …", expanded=True)
    trace_container = status_panel.container()
    events_seen: set[str] = set()
    final_state: ResearchState | None = None

    # Manually open an agent-span so every node/tool/LLM observation
    # nests under one Langfuse trace (we can't use @observe here because
    # the stream() generator is consumed inside this handler).
    try:
        with start_agent_span(
            "research_agent",
            input={"query": query.strip(), "output_language": language},
        ):
            for event in graph.stream(initial, stream_mode="values"):
                state_snapshot = cast(ResearchState, event)
                trace = state_snapshot.get("trace", [])
                for step in trace:
                    marker = f"{step.node}|{step.started_at.isoformat()}"
                    if marker in events_seen:
                        continue
                    events_seen.add(marker)
                    trace_container.markdown(
                        f"- **{step.node}** `{step.status}` · "
                        f"{step.duration_ms:.0f} ms · "
                        f"{step.details}",
                    )
                final_state = state_snapshot
            if final_state is not None:
                update_trace_io(
                    input={"query": query.strip(), "output_language": language},
                    output={
                        "n_sub_questions": len(final_state.get("plan", [])),
                        "total_cost_usd": round(final_state.get("total_cost_usd", 0.0), 6),
                        "report_chars": len(final_state.get("final_report") or ""),
                    },
                )
                trace_url = current_trace_url()
                if trace_url:
                    final_state["trace_url"] = trace_url
    except Exception as exc:
        status_panel.update(label=f"Lỗi: {exc}", state="error")
        st.exception(exc)
        st.stop()
    finally:
        lf_flush()

    status_panel.update(label="Hoàn tất", state="complete")

    if final_state is None:
        st.warning("Graph không trả về state nào.")
        st.stop()

    report = final_state.get("final_report") or "(empty report)"
    col_report, col_meta = st.columns([3, 1])
    with col_report:
        st.subheader("Báo cáo")
        st.markdown(report)
    with col_meta:
        st.subheader("Chỉ số")
        st.metric("Chi phí (USD)", f"${final_state.get('total_cost_usd', 0.0):.4f}")
        st.metric("Sub-questions", len(final_state.get("plan", [])))
        st.metric("Steps", len(final_state.get("trace", [])))

    with st.expander("State thô (debug)"):
        st.json(
            {
                "plan": [sq.model_dump() for sq in final_state.get("plan", [])],
                "drafts": {k: v.model_dump() for k, v in final_state.get("drafts", {}).items()},
                "evidence_counts": {k: len(v) for k, v in final_state.get("evidence", {}).items()},
                "total_cost_usd": final_state.get("total_cost_usd", 0.0),
            },
        )
