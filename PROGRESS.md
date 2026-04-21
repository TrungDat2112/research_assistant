# PROGRESS — Research Assistant Agent

> **Đọc file này ĐẦU TIÊN khi bắt đầu session mới.**
> File này luôn được cập nhật cuối session để phản ánh đúng trạng thái. Chi tiết kế hoạch xem `PLAN.md`.

---

## Trạng thái hiện tại

**Phase**: `Tuần 1 hoàn tất — pipeline end-to-end chạy 5/5 query, cost $0.13/$10 budget.`
**Last updated**: 2026-04-21
**Last session summary**:
- Implement Part B đầy đủ: web_search (Tavily), ResearchState + 5 Pydantic models, 3 prompt templates Jinja2, 3 agent nodes (planner/synthesizer/reporter), LangGraph wiring, Streamlit UI, CLI.
- 24 unit tests (state, prompts, web_search, agents, graph) — tất cả mock LLM + Tavily, không tốn API.
- Toolchain: `ruff check` ✓ | `ruff format --check` ✓ | `mypy` strict ✓ (18 files) | `pytest -q` ✓ (31/31 passed trong 2.77s).
- **Smoke test 5 query thật**: tổng cost `$0.1282` (vs $10 cap) · tổng wallclock 178.8s · tất cả 5 status=ok · 71 citations tổng cộng · outputs lưu ở `data/eval/week1_outputs.md` + metrics `data/eval/week1_metrics.json`.
- **Exit criteria Tuần 1 đạt**: 5/5 query end-to-end ✓, citation `[^N]` đầy đủ ✓, cost $0.13 << $1 ✓, trace có (Langfuse optional, đang `enabled` trong `.env`).

---

## Trạng thái repo

```
d:\research-assistant\
├── AI_building_principles.png    # ảnh nguồn — giữ nguyên
├── PLAN.md · PROGRESS.md · DECISIONS.md · AGENTS.md · README.md
├── pyproject.toml · uv.lock · .env.example · .gitignore
├── .venv/                        # (gitignored) Python 3.11.15
├── src/research_assistant/
│   ├── __init__.py               # __version__ = "0.1.0"
│   ├── config.py                 # pydantic-settings Settings + get_settings()
│   ├── cli.py                    # argparse entry: python -m research_assistant.cli "query"
│   ├── agents/
│   │   ├── _llm.py               # ChatAnthropic wrapper + cost estimator + BudgetExceededError
│   │   ├── planner.py            # Sonnet 4.5 → JSON list[SubQuestion] + fallback
│   │   ├── synthesizer.py        # Haiku 4.5 → Draft với [^N] citations
│   │   └── reporter.py           # deterministic Jinja render + citation renumbering toàn cục
│   ├── tools/
│   │   └── web_search.py         # Tavily wrapper → list[SearchHit]
│   ├── graph/
│   │   ├── state.py              # ResearchState TypedDict + reducers + 5 Pydantic models
│   │   └── research_graph.py     # LangGraph: planner→retriever→synthesizer→tick→reporter
│   ├── prompts/
│   │   ├── loader.py             # Jinja2 Environment + render helper (StrictUndefined)
│   │   ├── planner_v1.jinja
│   │   ├── synthesizer_v1.jinja
│   │   └── reporter_v1.jinja
│   ├── rag/__init__.py           # placeholder — Tuần 2+
│   ├── safety/__init__.py        # placeholder — Tuần 5
│   └── eval/__init__.py          # placeholder — Tuần 2+
├── tests/unit/
│   ├── test_smoke.py             # import + version check
│   ├── test_config.py            # defaults, flags, validator
│   ├── test_state.py             # 6 cases: Pydantic models + new_state
│   ├── test_prompts.py           # 4 cases: template rendering + strict undefined
│   ├── test_web_search.py        # 5 cases: stub client, clamp, empty, malformed, backend error
│   ├── test_agents.py            # 10 cases: planner parse/fallback, synthesize citations, reporter renumber
│   └── test_graph.py             # 2 cases: end-to-end mocked + max_iterations
├── ui/app.py                     # Streamlit: input → stream trace → render MD + metrics panel
├── scripts/week1_smoke.py        # 5-query smoke test runner
├── data/eval/
│   ├── week1_outputs.md          # 5 báo cáo Markdown (53 KB)
│   └── week1_metrics.json        # cost / wallclock / citations per query
├── configs/                      # (empty) YAML configs sẽ vào đây Tuần 2+
└── notebooks/                    # (empty)
```

---

## Việc tiếp theo (Next actions)

### Đã xong Tuần 1 — Part A (2026-04-21)
1. [x] `.gitignore`, `pyproject.toml` + `uv.lock`, `.env.example`, `README.md`.
2. [x] Cấu trúc folder khớp `PLAN.md` §12.
3. [x] `src/research_assistant/config.py` + unit test.
4. [x] Ruff + mypy strict + pytest toolchain sạch.

### Đã xong Tuần 1 — Part B (2026-04-21)
1. [x] `tools/web_search.py` — Tavily wrapper, `list[SearchHit]`, timeout, error wrapping, injectable client.
2. [x] `graph/state.py` — `ResearchState` TypedDict + reducers + 5 Pydantic models (SubQuestion, SearchHit, Evidence, Draft, StepLog) + `new_state()` factory.
3. [x] 3 agent nodes: `planner.py` (Sonnet 4.5, JSON parse + fallback plan), `synthesizer.py` (Haiku 4.5, citation extraction, no-evidence fallback), `reporter.py` (deterministic Jinja, renumber `[^N]` globally).
4. [x] 3 prompt templates Jinja2 + `prompts/loader.py` (StrictUndefined, cached environment).
5. [x] `graph/research_graph.py` — LangGraph wiring: `planner → retriever → synthesizer → tick → {retriever | reporter} → END`, kèm `max_iterations` guard.
6. [x] `ui/app.py` (Streamlit) + `cli.py` (argparse) — 2 entry points.
7. [x] Langfuse tracing: detected qua `settings.langfuse_enabled` (chỉ mới log; instrument chi tiết để Tuần 2).
8. [x] `agents/_llm.py` — shared LLM helper + USD cost estimator + `BudgetExceededError` pre-flight guard (ADR-011).
9. [x] 31 unit test: state / prompts / web_search / agents / graph đều mock LLM + Tavily, không tốn API.
10. [x] `scripts/week1_smoke.py` — chạy 5 query, lưu outputs + metrics.

### Exit criteria Tuần 1 — ĐẠT
| Tiêu chí | Kết quả |
|---|---|
| 5 query mẫu chạy end-to-end | ✓ 5/5 status=ok |
| Mỗi report có citation `[^N]` | ✓ 71 citations tổng cộng |
| Tổng chi phí < $1 | ✓ $0.1282 / $10 budget (1.28%) |
| Có trace | ✓ 4–16 StepLog / query (Langfuse detect qua settings) |

### Việc tiếp theo — Tuần 2 (session sau)
1. [ ] **Fix**: Planner đôi khi xuất JSON invalid (query 1 VI đã fallback 1 sub-q). Cân nhắc `ChatAnthropic.with_structured_output(SubQuestion)` hoặc retry-with-repair prompt.
2. [ ] **Improve**: Retriever trả 0 hits cho 3/33 sub-queries (query 3 sq_5, query 4 sq_5+sq_6). Thêm fallback: giảm độ cụ thể của query, hoặc cho phép `search_depth="advanced"`.
3. [ ] **Instrument Langfuse**: decorator `@observe` cho từng node + `trace.update(input/output)`; link `trace_id` trong report footer.
4. [ ] **Bắt đầu RAG pipeline (PLAN.md §5)**: ingestion (arXiv + HTML), chunking, embedding bằng bge-m3, Chroma dev store.
5. [ ] **Hybrid retrieval stage 1**: BM25 + dense → top-50 candidates.
6. [ ] **Critic agent** (draft) — kiểm citation coverage, output schema kiểm query → quyết định loop thêm hay pass.
7. [ ] Chuyển `cli.py` vào `[project.scripts]` để gọi `uv run research-assistant "query"` trực tiếp.

### Notes / cảnh báo ghi nhớ
- Haiku 4.5 pricing trong `agents/_llm.py._PRICING_USD_PER_MTOK` đang là ước lượng conservative (input $1 / output $5 per MTok). Cần kiểm lại tại `https://www.anthropic.com/pricing` khi có thời gian và cập nhật nếu lệch > 20%.
- Console Windows mặc định cp1252; `scripts/week1_smoke.py` đã force UTF-8 cho stdout, nhưng nếu viết thêm CLI có tiếng Việt cần apply cùng pattern.
- `.env.example` từng bị user paste nhầm key thật (ngày hôm nay); đã revert về placeholder. **Luôn** double-check trước khi commit.

---

## Câu hỏi mở (cần user quyết định)

Tất cả câu hỏi khởi tạo đã được chốt (2026-04-21). Xem § "Quyết định dự án đã chốt" bên dưới.

## Quyết định dự án đã chốt

| # | Quyết định | Giá trị |
|---|---|---|
| 1 | Ngôn ngữ report mặc định | **Tiếng Việt** (prompt LLM vẫn tiếng Anh để ổn định reasoning, chỉ output tiếng Việt) |
| 2 | LLM chính cho Planner/Critic | **Claude Sonnet 4.5** (via `anthropic` SDK) |
| 3 | LLM cho Synthesizer | **Claude Haiku** (cùng provider, tiết kiệm auth/billing) |
| 4 | Observability | **Langfuse Cloud Free Tier** (`https://cloud.langfuse.com`) |
| 5 | Corpus mẫu Tuần 2 | **AI/ML** (arXiv cs.AI/cs.LG + blog posts + papers nổi bật 2024–2026) |
| 6 | Test budget tổng | **$10 USD** (dev + test API calls) → per-query cap $0.30, alert khi chạm $7 |

Chi tiết lý do ghi trong `DECISIONS.md` ADR-007 → ADR-011.

---

## Log session

### 2026-04-21 — Session 1 (Planning)
- Đọc `AI_building_principles.png`, tổng hợp 15+ nguyên tắc Stanford.
- Viết outline bài toán → kế hoạch chi tiết.
- Tạo bộ 4 file handoff (`PLAN.md`, `PROGRESS.md`, `DECISIONS.md`, `AGENTS.md`).
- Chốt 6 câu hỏi khởi tạo (ngôn ngữ, LLM, observability, corpus, budget) → ADR-007..011.
- **Next**: Vào Tuần 1 — Skeleton & tool cơ bản.

### 2026-04-21 — Session 2 (Tuần 1 Part A — Scaffolding)
- Verify môi trường: `uv 0.11.7` + `Python 3.12.10` + git OK trên Windows.
- `.gitignore` (Python/venv/data/.env/notebooks/qdrant/logs) + `.env.example` (10 biến, có ADR-ref).
- `pyproject.toml` tự viết (không `uv init`) với runtime deps + dev extras + ruff/mypy strict/pytest config.
- Folder tree khớp `PLAN.md` §12: `src/research_assistant/{agents,tools,rag,graph,prompts,safety,eval}`, `tests/{unit,integration}`, `configs/`, `notebooks/`, `data/{raw,processed,eval}/` với `.gitkeep`, `ui/`.
- `src/research_assistant/config.py` — pydantic-settings + SecretStr + `get_settings()` cache + các flag `has_llm_credentials` / `has_search_credentials` / `langfuse_enabled`.
- `README.md` + `tests/unit/test_smoke.py` + `tests/unit/test_config.py` (3 cases).
- `uv sync --all-extras` → 95 packages installed (venv dùng Python 3.11.15 — `>=3.11` theo `pyproject`).
- Verify toolchain: `ruff check` ✓ (tất cả passed) | `ruff format --check` ✓ (14 files) | `mypy` ✓ (9 src files) | `pytest -q` ✓ (4/4 passed).
- **Blocker**: không.
- **Next**: Part B — web_search tool, ResearchState, 3 agent nodes, LangGraph wiring, Streamlit UI, smoke test 5 query mẫu. User cần cung cấp API keys thật + danh sách 5 query.

### 2026-04-21 — Session 3 (Tuần 1 Part B — Agent logic + smoke test)
- **Security incident sớm session**: user paste nhầm real API keys vào `.env.example` (đã committed). Xử lý: tạo `.env` (gitignored) đặt key thật + revert `.env.example` về placeholder + yêu cầu user rotate 3 keys (Anthropic / Tavily / Langfuse). User confirm đã rotate trước khi chạy smoke.
- **Code Part B** (11 file src, ~1200 LoC):
  - `graph/state.py` — `ResearchState` TypedDict + reducers (`_extend_list`, `_merge_dict`) + 5 Pydantic models với validation pattern/min-length/HttpUrl.
  - `prompts/{planner,synthesizer,reporter}_v1.jinja` + `loader.py` (StrictUndefined, cached env).
  - `tools/web_search.py` — Tavily wrapper, injectable client cho test, `WebSearchError` wrapping backend errors, drop malformed rows với warning.
  - `agents/_llm.py` — `ChatAnthropic` builder, `invoke_llm()` với pre-flight budget guard (ADR-011), pricing table cho Sonnet 4.5 / Haiku 4.5 / Haiku 3.5 + conservative fallback.
  - `agents/planner.py` — JSON parse + fallback single-sub-q khi parse fail / LLM crash; renumber `sq_N` contiguously + rewrite `dependency_ids` qua id remap.
  - `agents/synthesizer.py` — extract `[^N]` markers → `Citation`, mark `Evidence.used=True`, no-evidence fallback message song ngữ.
  - `agents/reporter.py` — **deterministic (không gọi LLM)**, renumber markers globally theo offset cộng dồn → References list khớp `[^N]` cuối báo cáo.
  - `graph/research_graph.py` — LangGraph `planner → retriever → synthesizer → tick → {loop | reporter}`, retriever là closure để inject search_fn mock trong test.
  - `ui/app.py` (Streamlit): sidebar cấu hình + credential status, stream trace real-time, metrics panel, debug JSON.
  - `cli.py` — argparse, `--out`, `--language`, `--max-iterations`, exit code chuẩn.
  - `scripts/week1_smoke.py` — orchestrate 5 query, ghi `data/eval/week1_outputs.md` + `week1_metrics.json`.
- **Test**: 31 unit test (state/prompts/web_search/agents/graph), mock LLM + Tavily nên 0 cost. `pytest -q` chạy 2.77s.
- **Toolchain final**: `ruff check` ✓ · `ruff format --check` ✓ (29 files) · `mypy` strict ✓ (18 src files) · `pytest` 31/31 ✓.
- **Smoke test 5 query thật** (cost breakdown):
  1. LoRA vs QLoRA (VI) — $0.0197, 20.2s, 1 sub-q (planner fallback), 4 citations.
  2. RAG vs fine-tuning (VI) — $0.0272, 37.9s, 5 sub-q, 17 citations.
  3. o3 vs R1 reasoning models (EN) — $0.0263, 39.7s, 6 sub-q, 16 citations.
  4. Qdrant vs Weaviate vs Milvus (EN) — $0.0303, 44.0s, 7 sub-q, 15 citations.
  5. Agentic RAG (EN) — $0.0248, 37.1s, 5 sub-q, 19 citations.
  **Total**: $0.1282 · 178.8s · 71 citations · 5/5 ok.
- **Blocker**: không.
- **Next**: Tuần 2 — structured output cho planner, RAG pipeline (ingest + chunk + embed + BM25+dense), Critic agent, Langfuse instrumentation chi tiết. Xem § "Việc tiếp theo".

<!-- Khi kết thúc session, thêm entry mới theo format:
### YYYY-MM-DD — Session N (Tên phase)
- Việc đã làm
- Việc chưa xong
- Blocker (nếu có)
- Next action cụ thể
-->
