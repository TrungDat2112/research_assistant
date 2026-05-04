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

Agent nghiên cứu chủ đề: **lập kế hoạch (sub-questions)** → **thu thập đa nguồn** (corpus, web, arXiv…) → **RAG hai giai đoạn** (hybrid BM25+dense → cross-encoder rerank) → **tổng hợp có citation** `[^N]` → **kiểm Critic / mâu thuẫn nguồn** → **báo cáo Markdown**. 


## Yêu cầu & cài đặt

**Cần:** Python **3.11+**, Git, **[`uv`](https://docs.astral.sh/uv/)**. Hỗ trợ Windows / macOS / Linux.

```bash
git clone https://github.com/TrungDat2112/research_assistant.git
cd research_assistant

uv sync --all-extras
copy .env.example .env
```

---

## Biến môi trường


 `ANTHROPIC_API_KEY` | Planner, Synthesizer, Critic, compare_sources LLM |
 `TAVILY_API_KEY` | Web search |


## Corpus (RAG) — ingest

Để corpus nội bộ và hybrid retrieval hoạt động, cần ít nhất một lần ingest (lần đầu hoặc sau khi đổi model embedding trong `.env`):

```bash
uv run python scripts/ingest_seed_corpus.py --rebuild
```

- Cấu hình nguồn seed: [`configs/seed_corpus.yaml`](./configs/seed_corpus.yaml).
- Chroma lưu tại `data/chroma/`.

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
| [`scripts/run_factuality_eval.py`](./scripts/run_factuality_eval.py) | Factuality (LLM judge) |



## Kiểm thử & chất lượng code

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/research_assistant
```

## License

MIT.
