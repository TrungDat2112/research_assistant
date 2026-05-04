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

Agent nghiên cứu chủ đề: **lập kế hoạch (sub-questions)** → **thu thập đa nguồn** (corpus, web, arXiv…) → **RAG hai giai đoạn** (hybrid BM25+dense → cross-encoder rerank) → **tổng hợp có citation** `[^N]` → **kiểm Critic / mâu thuẫn nguồn** → **báo cáo Markdown**. Nguyên tắc nền xem [`AI_building_principles.png`](./AI_building_principles.png) (Stanford *How to Build AI Agents*).

---

## Mục lục

- [Tại sao dự án này](#tại-sao-dự-án-này)
- [Công nghệ chính](#công-nghệ-chính)
- [Yêu cầu & cài đặt](#yêu-cầu--cài-đặt)
- [Biến môi trường](#biến-môi-trường)
- [Corpus (RAG) — ingest](#corpus-rag--ingest)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Eval & scripts](#eval--scripts)
- [Kiểm thử & chất lượng code](#kiểm-thử--chất-lượng-code)
- [CI/CD & Hugging Face Space](#cicd--hugging-face-space)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Ngân sách & an toàn](#ngân-sách--an-toàn)
- [Tài liệu dự án](#tài-liệu-dự-án)
- [License](#license)

---

## Tại sao dự án này

- Pipeline **đa agent** qua **LangGraph** (planner → retriever → synthesizer → so sánh nguồn → critic → reporter).
- **Tool router** heuristic (intent → thứ tự `vector_search`, `web_search`, `academic_search`…).
- **Trace** có thể bật với **Langfuse Cloud** (tuỳ chọn).

Trạng thái roadmap: Tuần 1–4 đã khép theo [`PLAN.md`](./PLAN.md) roadmap mục 10 (chi tiết cập nhật trong [`PROGRESS.md`](./PROGRESS.md)). Tiếp theo chủ đề Tuần 5: guardrails, budget, polish observability.

---

## Công nghệ chính

| Lớp | Công cụ |
|-----|---------|
| Runtime | Python 3.11+, [`uv`](https://docs.astral.sh/uv/) |
| Đồ thị agent | LangGraph |
| LLM | Anthropic Claude (Sonnet planner/critic; Haiku synthesizer — xem [`DECISIONS.md`](./DECISIONS.md)) |
| Tìm kiếm web | Tavily |
| Embedding / rerank | `BAAI/bge-m3`, `bge-reranker-v2-m3` (tuỳ cấu hình) |
| Vector store (dev) | Chroma persistent |
| Retrieval | Hybrid BM25 + dense, HyDE tuỳ chọn |
| UI | Streamlit (`ui/app.py`) |
| Observability | Langfuse (tuỳ chọn) |

---

## Yêu cầu & cài đặt

**Cần:** Python **3.11+**, Git, **[`uv`](https://docs.astral.sh/uv/)**. Hỗ trợ Windows / macOS / Linux.

```bash
git clone https://github.com/TrungDat2112/research_assistant.git
cd research_assistant

uv sync --all-extras
cp .env.example .env   # Windows: copy .env.example .env
# Chỉnh .env và điền key (Anthropic, Tavily, …)
```

---

## Biến môi trường

Xem **[`.env.example`](./.env.example)** làm checklist đầy đủ.

| Biến (tối thiểu để chạy agent thật) | Mục đích |
|--------------------------------------|-----------|
| `ANTHROPIC_API_KEY` | Planner, Synthesizer, Critic, compare_sources LLM |
| `TAVILY_API_KEY` | Web search |

Tuỳ chọn: Langfuse (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`), embedding/rerank/critic/router (xem `.env.example`).

**Không commit** `.env`; chỉ commit `.env.example` (placeholder).

---

## Corpus (RAG) — ingest

Để corpus nội bộ và hybrid retrieval hoạt động, cần ít nhất một lần ingest (lần đầu hoặc sau khi đổi model embedding trong `.env`):

```bash
uv run python scripts/ingest_seed_corpus.py --rebuild
```

- Cấu hình nguồn seed: [`configs/seed_corpus.yaml`](./configs/seed_corpus.yaml).
- Chroma lưu tại `data/chroma/` (thường gitignored).

---

## Chạy ứng dụng

### CLI (một câu hỏi, Markdown ra stdout/file)

```bash
uv run research-assistant "Câu hỏi nghiên cứu của bạn" --language vi --out ./report.md
```

Tuỳ chọn: `--language en`, `--max-iterations`, `--no-rerank`, `--no-critic` — xem `research-assistant --help`.

### Streamlit

```bash
uv run streamlit run ui/app.py
```

Trên Hugging Face Space (Docker): xem [**`docs/deploy-huggingface.md`**](./docs/deploy-huggingface.md).

### Docker cục bộ (như HF)

```bash
docker build -t research-assistant-space .
docker run --rm -p 7860:7860 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e TAVILY_API_KEY="$TAVILY_API_KEY" \
  research-assistant-space
```

---

## Eval & scripts

| Script | Ý nghĩa |
|--------|---------|
| [`scripts/smoke_eval.py`](./scripts/smoke_eval.py) | 5 query cố định, metrics chi phí / trace (có `--with-router`, `--with-compare-sources`) |
| [`scripts/run_retrieval_eval.py`](./scripts/run_retrieval_eval.py) | Retrieval metrics (Recall, NDCG, tuỳ chọn rerank/HyDE) |
| [`scripts/run_citation_eval.py`](./scripts/run_citation_eval.py) | Citation coverage từ batch Markdown |
| [`scripts/run_factuality_eval.py`](./scripts/run_factuality_eval.py) | Factuality (LLM judge) — **tốn API** |

Ví dụ dataset trong `data/eval/`. Chi tiết bộ chỉ tiêu: [`PLAN.md`](./PLAN.md).

---

## Kiểm thử & chất lượng code

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/research_assistant
```

Mirror đúng thứ tự job CI (sau sync):

```bash
uv run python scripts/verify_ci_local.py
```

Quy ước và checklist session: **[`AGENTS.md`](./AGENTS.md)**.

---

## CI/CD & Hugging Face Space

- **CI:** [.github/workflows/ci.yml](./.github/workflows/ci.yml) — ruff, mypy, pytest trên push/PR vào `main`.
- **Bảo vệ nhánh PR / xác nhận sau merge:** [`docs/ci-branch-protection.md`](./docs/ci-branch-protection.md).
- **Deploy Space:** [`docs/deploy-huggingface.md`](./docs/deploy-huggingface.md).

---

## Cấu trúc thư mục

```
research-assistant/
├── src/research_assistant/   # agent, tools, rag, graph, prompts, eval, observability…
├── tests/unit/               # pytest
├── ui/app.py                  # Streamlit
├── configs/seed_corpus.yaml  # Corpus seed ingest
├── scripts/                   # ingest, smoke, eval runners
├── data/eval/                 # eval datasets / baseline (tuỳ phần thư mục)
├── Dockerfile                  # HF Space / container
├── PLAN.md · PROGRESS.md · DECISIONS.md · AGENTS.md
└── pyproject.toml · uv.lock
```

Đầy đủ trong [`PLAN.md`](./PLAN.md) phần cấu trúc repo đề xuất.

---

## Ngân sách & an toàn

- Theo ADR: ngân sách dev/tổng và **per-query cap** (xem `.env.example` và [`DECISIONS.md`](./DECISIONS.md) ADR-011).
- Citation được thiết kế để giảm bịa nội dung; Critic và optionally `compare_sources` hỗ trợ chất lượng.
- **Không crawl web tự động quy mô lớn** — chỉ search API và URL chỉ định theo luồng agent.

---

## Tài liệu dự án

| File | Nội dung |
|------|----------|
| [`PROGRESS.md`](./PROGRESS.md) | Trạng thái hiện tại và việc tiếp theo — **đọc đầu tiên khi vào phiên làm việc mới**. |
| [`PLAN.md`](./PLAN.md) | Kiến trúc, roadmap, metric mục tiêu |
| [`DECISIONS.md`](./DECISIONS.md) | ADR (quyết định kỹ thuật) |
| [`AGENTS.md`](./AGENTS.md) | Hướng dẫn cho AI trong Cursor/Code |

Prompt handoff AI (copy khi đổi phiên):

```text
Đọc theo thứ tự: PROGRESS.md → PLAN.md → DECISIONS.md → AGENTS.md, rồi giúp tôi với ...
```

---

## License

MIT.
