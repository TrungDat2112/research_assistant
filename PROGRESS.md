# PROGRESS — Research Assistant Agent

> **Đọc file này ĐẦU TIÊN khi bắt đầu session mới.**
> File này luôn được cập nhật cuối session để phản ánh đúng trạng thái. Chi tiết kế hoạch xem `PLAN.md`.

---

## Trạng thái hiện tại

**Phase**: `Tuần 1 — Part A hoàn tất (scaffolding). Part B pending (agent logic + smoke test).`
**Last updated**: 2026-04-21
**Last session summary**:
- Setup toàn bộ scaffolding: `.gitignore`, `pyproject.toml` (ruff + mypy strict + pytest), `.env.example`, `README.md`.
- Cài 95 packages qua `uv sync --all-extras` (langgraph 1.1.8, langchain-anthropic 1.4.1, anthropic 0.96.0, tavily-python 0.7.23, langfuse 4.3.1, streamlit 1.56.0, pydantic 2.13.3, ...).
- Folder tree khớp `PLAN.md` §12 với `__init__.py` + module docstrings.
- Implement `src/research_assistant/config.py` (pydantic-settings, SecretStr, `get_settings()` cached) + 3 unit test.
- Toolchain sạch: `ruff check` ✓ | `ruff format --check` ✓ | `mypy` ✓ (9 files) | `pytest` 4/4 ✓.

---

## Trạng thái repo

```
d:\research-assistant\
├── AI_building_principles.png    # ảnh nguồn — giữ nguyên
├── PLAN.md                       # kế hoạch đầy đủ
├── PROGRESS.md                   # file này
├── DECISIONS.md                  # decision log
├── AGENTS.md                     # quy ước làm việc cho AI assistant
├── README.md                     # setup + run
├── pyproject.toml                # deps + ruff + mypy strict + pytest
├── uv.lock                       # lockfile
├── .env.example                  # template env vars
├── .gitignore
├── .venv/                        # (gitignored) Python 3.11.15
├── src/research_assistant/
│   ├── __init__.py               # __version__ = "0.1.0"
│   ├── config.py                 # pydantic-settings Settings + get_settings()
│   ├── agents/__init__.py        # placeholder — Part B
│   ├── tools/__init__.py         # placeholder — Part B
│   ├── rag/__init__.py           # placeholder — Tuần 2+
│   ├── graph/__init__.py         # placeholder — Part B
│   ├── prompts/__init__.py       # placeholder — Part B
│   ├── safety/__init__.py        # placeholder — Tuần 5
│   └── eval/__init__.py          # placeholder — Tuần 2+
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_smoke.py         # import + version check
│   │   └── test_config.py        # 3 cases: defaults, flags, validator
│   └── integration/__init__.py
├── configs/                      # (empty) YAML configs sẽ vào đây Tuần 2+
├── notebooks/                    # (empty)
├── data/{raw,processed,eval}/    # với .gitkeep
└── ui/                           # (empty) streamlit app sẽ vào đây Part B
```

---

## Việc tiếp theo (Next actions)

### Đã xong (Part A — 2026-04-21)
1. [x] `.gitignore` (Python + venv + data + .env).
2. [x] `pyproject.toml` + `uv.lock` với deps cốt lõi (95 packages cài qua `uv sync`).
3. [x] Cấu trúc folder khớp `PLAN.md` §12.
4. [x] `.env.example` với 10 biến env + docstring ADR-ref.
5. [x] `README.md` (setup + run + handoff).
6. [x] `src/research_assistant/config.py` + unit test (4 passed).
7. [x] Ruff + mypy strict + pytest config đều chạy sạch.

### Part B — Tuần 1 (session kế tiếp)
1. [ ] Lấy `.env` thật: ANTHROPIC, TAVILY, LANGFUSE keys (user làm).
2. [ ] Implement `src/research_assistant/tools/web_search.py` — wrapper Tavily, kiểu return `list[SearchHit]` (Pydantic), có docstring LLM-friendly, rate limit + timeout.
3. [ ] Implement `src/research_assistant/graph/state.py` — `ResearchState` TypedDict theo `PLAN.md` §6.1; các model `SubQuestion`, `Evidence`, `Draft`, `StepLog` bằng Pydantic.
4. [ ] Implement agents tối giản (mỗi cái là 1 function `(state) -> state`):
   - `agents/planner.py` — gọi Claude Sonnet 4.5, output list[SubQuestion] (JSON).
   - `agents/synthesizer.py` — gọi Claude Haiku, input (sub_q + evidence) → draft + citations.
   - `agents/reporter.py` — gom drafts thành MD report tiếng Việt.
5. [ ] Prompt templates (Jinja2) trong `src/research_assistant/prompts/`:
   - `planner_v1.jinja`, `synthesizer_v1.jinja`, `reporter_v1.jinja`.
6. [ ] `src/research_assistant/graph/research_graph.py` — wire LangGraph:
   `planner → (loop sub-qs: web_search → synthesize) → reporter`.
7. [ ] `ui/app.py` — Streamlit tối giản: input query → stream output MD + debug panel.
8. [ ] Langfuse tracing optional (chỉ bật nếu `settings.langfuse_enabled`).
9. [ ] Smoke test 5 query mẫu → lưu trace/output vào `notebooks/week1_smoke_test.ipynb` + `data/eval/week1_outputs.md`.

**Exit criteria Tuần 1**: 5 query mẫu chạy end-to-end, mỗi report có citation `[^N]`, tổng chi phí < $1, có trace (Langfuse optional).

### Câu hỏi cho user trước Part B
- 5 query mẫu cụ thể là gì? (đề xuất: 2 tiếng Việt + 3 tiếng Anh, chủ đề AI/ML đúng ADR-010, ví dụ: *"So sánh LoRA và QLoRA cho fine-tuning LLM 2025"*, *"Retrieval-Augmented Generation là gì, khi nào nên dùng?"*, ...).
- Xác nhận tên model Haiku lúc chạy (default đang là `claude-haiku-4-5`; nếu 403 → fallback `claude-3-5-haiku-latest` theo ADR-008).

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

<!-- Khi kết thúc session, thêm entry mới theo format:
### YYYY-MM-DD — Session N (Tên phase)
- Việc đã làm
- Việc chưa xong
- Blocker (nếu có)
- Next action cụ thể
-->
