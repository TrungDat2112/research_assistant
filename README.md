---
title: Research Assistant Agent
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Research Assistant Agent

[![CI](https://github.com/TrungDat2112/research_assistant/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TrungDat2112/research_assistant/actions/workflows/ci.yml)

AI agent tự động nhận câu hỏi nghiên cứu → lập kế hoạch → tìm kiếm đa nguồn → tổng hợp có citation → xuất báo cáo Markdown/PDF.

Dựa trên nguyên tắc Stanford *"How to Build AI Agents"* (xem `AI_building_principles.png`). Kiến trúc đầy đủ trong [`PLAN.md`](./PLAN.md), lý do chọn công nghệ trong [`DECISIONS.md`](./DECISIONS.md).

> **Status**: `Tuần 1 — Skeleton` (Pre-agent logic). Xem [`PROGRESS.md`](./PROGRESS.md) cho trạng thái mới nhất.

---

## Yêu cầu hệ thống

- Python **3.11+** (test trên 3.12).
- [`uv`](https://docs.astral.sh/uv/) package manager.
- Git.
- OS: Windows / macOS / Linux.

## Setup

```powershell
# 1. Clone + vào repo
git clone <url> research-assistant
cd research-assistant

# 2. Tạo venv + cài deps (runtime + dev)
uv sync --all-extras

# 3. Copy env template và điền key
copy .env.example .env
# Mở .env, điền: ANTHROPIC_API_KEY, TAVILY_API_KEY, LANGFUSE_*
```

API keys cần lấy:

| Dịch vụ | Đăng ký | Dùng cho |
|---|---|---|
| Anthropic | <https://console.anthropic.com> | Planner (Sonnet 4.5) + Synthesizer (Haiku) |
| Tavily | <https://tavily.com> | Web search |
| Langfuse Cloud | <https://cloud.langfuse.com> | Tracing / observability (optional) |

## CI (GitHub Actions)

Lint, type-check, và pytest chạy trong [`.github/workflows/ci.yml`](./.github/workflows/ci.yml).

**Gắn CI với PR (bắt buộc CI xanh trước khi merge vào branch chính):** chỉ chủ repo cấu hình trên GitHub — làm theo [`docs/ci-branch-protection.md`](./docs/ci-branch-protection.md) (Bước 7 + **Bước 8**: xác nhận sau merge, script `scripts/verify_ci_local.py`).

**Deploy Streamlit Space (Hugging Face, Docker SDK):** xem [`docs/deploy-huggingface.md`](./docs/deploy-huggingface.md) — token `HF_TOKEN` trên GitHub; API keys chạy app đặt trên HF Space Secrets.

## Chạy

```powershell
# Streamlit UI (có khi hoàn thiện Tuần 1 Part B)
uv run streamlit run ui/app.py

# Smoke test
uv run pytest

# Lint + type check
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Cấu trúc repo

Chi tiết trong [`PLAN.md` §12](./PLAN.md). Tóm tắt:

```
src/research_assistant/
├── agents/   # planner, synthesizer, critic, reporter
├── tools/    # web_search, fetch_*, vector_search, ...
├── rag/      # ingest, chunking, retrieval, rerank (Tuần 2+)
├── graph/    # LangGraph state machine
├── prompts/  # Jinja2 templates, versioned
├── safety/   # guardrails, budget tracker
├── eval/     # metrics, eval datasets
└── config.py # pydantic-settings
```

## Safety & Budget

- **Hard budget**: $10 tổng, $0.30 per query (ADR-011). Agent refuse khi vượt.
- **Citation bắt buộc**: Synthesizer cấm claim không có `[^N]` ref; Critic reject khi coverage < 90%.
- **Không commit secret**: `.env` bị gitignore, chỉ `.env.example` có trong repo.

## Handoff giữa session (AI assistant)

Bắt đầu session mới, nhắc AI đọc theo thứ tự:

1. [`PROGRESS.md`](./PROGRESS.md) — đang ở đâu, làm tiếp gì.
2. [`PLAN.md`](./PLAN.md) — kiến trúc tổng thể.
3. [`DECISIONS.md`](./DECISIONS.md) — lý do chọn X thay vì Y.
4. [`AGENTS.md`](./AGENTS.md) — quy ước code & workflow.

## License

MIT.
