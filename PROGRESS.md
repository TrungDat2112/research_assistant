# PROGRESS — Research Assistant Agent

> **Đọc file này ĐẦU TIÊN khi bắt đầu session mới.**
> File này luôn được cập nhật cuối session để phản ánh đúng trạng thái. Chi tiết kế hoạch xem `PLAN.md`.

---

## Trạng thái hiện tại

**Phase**: `Tuần 4 khởi động — multi-source tools (academic_search / fetch_pdf), tool router rule-based, compare_sources, factuality eval 20 query (PLAN.md §10 Tuần 4).`
**Last updated**: 2026-04-27
**Last session summary**:
- **Retrieval eval 100** (`data/eval/retrieval_eval_100.json`): 70 EN + 30 VI, multi-gold qrels; `expand_retrieval_eval.py --write` validate theo manifest; `run_retrieval_eval.py` default → file này; `RetrievalEvalItem.language` trong `eval/retrieval.py`.
- **Rerank eval**: `run_retrieval_eval.py --with-rerank` — so sánh stage-1 vs pool+`bge-reranker-v2-m3`; thêm MRR, precision@5; `ab_delta` trong JSON output.
- **Full 5-query smoke re-run** (`scripts/week1_smoke.py`): tổng **$0.7565** · **2429.5s** wallclock · 5/5 ok. So với baseline Tuần 1 trong repo (**$0.1282** · **178.8s**): chênh chủ yếu do **Critic** (thêm structured Sonnet/sub-q) + **hybrid corpus + web + cross-encoder rerank** (CPU ~30–80s/retrieval batch lần đầu) + plan dài hơn (5–7 sub-q thay vì 1–7). Mỗi query có `langfuse_trace_id` / `langfuse_trace_url` trong `week1_metrics.json`. Script ghi thêm **retrieval**: `evidence_hits_by_source` (corpus vs web) + `retriever_details` (`n_corpus`/`n_web`/`retrieval_path`/`n_pool`/`n_after_rerank` per sub-q). **Cảnh báo**: 3/5 lần chạy log `Max iterations reached (8)` (query RAG so với fine-tuning, o3 vs R1, vector DB) — cần tune `max_iterations` hoặc Critic/retry nếu muốn hoàn tất mọi sub-q trước report.
- **CLI entry**: `[project.scripts]` `research-assistant` → `uv run research-assistant "query"` (vẫn hỗ trợ `python -m research_assistant.cli`).
- **bge-m3 default** (`config.py` / ADR-018): `embedding_model=BAAI/bge-m3`; chunking tokenizer `model_max_length` nới để PDF dài không cảnh báo 8k; `.env.example` ghi override `bge-small` khi cần lặp nhanh. **Sau khi pull**: chạy `uv run python scripts/ingest_seed_corpus.py --rebuild` để Chroma + `data/eval/ingest_manifest.json` khớp 1024-dim (CPU có thể ~30–60 phút / 819 chunk).
- **Critic (draft)**: node `critic` sau `synthesizer`; metric đoạn+citation + structured Sonnet; retry → `retriever` + feedback vào synthesizer; `CRITIC_ENABLED=false` trong unit graph tests; ADR-017.
- **Deps mới** (`pyproject.toml`) *(session trước)*: `chromadb>=0.5.15`, `sentence-transformers>=3.0.0`, `trafilatura>=1.12.0`, `arxiv>=2.1.3`, `pymupdf>=1.24.0`, `pyyaml>=6.0.2`. `uv sync` → 176 packages resolved.
- **Settings** (`config.py`) *(lịch sử + hiện tại)*: `embedding_model` (**default `BAAI/bge-m3`** — ADR-018; override `bge-small` để lặp nhanh), `embedding_device`, `chroma_persist_dir`, `corpus_collection`, chunk sizes, `raw_docs_dir`.
- **RAG package** (`src/research_assistant/rag/`, +~720 LoC):
  - `schemas.py` — `SourceDoc` / `Chunk` / `ChunkMetadata` Pydantic + `make_source_id` (SHA-1 tail cho HTML), `to_chroma()` flatten metadata sang scalar-only.
  - `chunking.py` — token-aware slicer dùng HF fast tokenizer của embedding model (lru-cached). `_iter_sections` dò Markdown `#` + ALL-CAPS ngắn; `_summarise_for_prepend` lấy `doc.summary` > 2 câu đầu > title. Cắt theo offset → decode body; `text` = `[Title] summary\n\nbody` cho embedding. Empty / overlap≥size → error rõ.
  - `embedding.py` — wrapper singleton `EmbeddingModel` qua `sentence-transformers`; prefix `"Represent this sentence for searching relevant passages: "` tự động cho BGE khi `embed_query`. Normalised cosine vectors.
  - `vector_store.py` — `ChromaStore` dùng `PersistentClient(path=data/chroma)`, collection `get_or_create_collection(metadata={"hnsw:space": "cosine"})`. API: `upsert_chunks(chunks, embeddings)` idempotent, `search(query_vec, top_k, where=...)` trả `SearchResult(chunk_id, body, metadata, distance)`, `reset()` / `count()`.
  - `ingest/arxiv_source.py` — `search_arxiv(query, date_from, categories)` + `fetch_arxiv_doc(id, cache_dir)` cache PDF dưới `data/raw/arxiv/`, extract text bằng pymupdf (cap 400k chars).
  - `ingest/html_source.py` — `fetch_html_doc(url)` qua trafilatura (`extract_metadata` → title/author/date/summary, `extract` txt body, favor precision, include tables). Không playwright ở v1.
  - `ingest/loader.py` — `SeedConfig.from_yaml(path)` + `load_seed_corpus(config)` dedup arXiv id giữa queries và explicit ids, trả `IngestResult(docs, failures)` — lỗi từng nguồn không crash toàn batch.
- **Config corpus**: `configs/seed_corpus.yaml` — 10 arXiv papers (RAG, Self-RAG, RAG Survey, LoRA, QLoRA, ReAct, Self-Refine, Llama 2, DeepSeek-R1, GraphRAG) + 5 blog (Anthropic Contextual Retrieval, Anthropic Building Agents, OpenAI Structured Outputs, HF ray-rag, LangChain LangGraph).
- **Ingest script** (`scripts/ingest_seed_corpus.py`, ~180 LoC): argparse `--config`/`--rebuild`/`--limit`/`--manifest`, fetch → chunk → embed → upsert, ghi `data/eval/ingest_manifest.json` với source list + timings.
- **Run thực tế** (`uv run python scripts/ingest_seed_corpus.py --rebuild`):
  - 15/15 docs fetched, 0 failures, 47.2s.
  - 766 chunks, 22.3s chunking.
  - 766 embeddings dim=384 (bge-small), 224.3s = 3.4 ch/s trên CPU, first-run kèm download ~130MB model.
  - Upsert Chroma 3.6s → collection `ai_ml_corpus_v1` holds 766 chunks.
  - Manifest ghi `data/eval/ingest_manifest.json`.
- **Retrieval smoke** (3 query, top-3 mỗi query):
  - "LoRA and trainable parameters" → top hits là LoRA + QLoRA papers (dist 0.256–0.259).
  - "Contextual retrieval RAG" → top hit Anthropic blog "Introducing Contextual Retrieval" (dist 0.132).
  - "ReAct reasoning loop" → toàn bộ top-3 từ paper ReAct (dist 0.177–0.193).
- **Tests**: +4 test module (`test_rag_schemas`, `test_rag_chunking`, `test_rag_vector_store`, `test_rag_ingest`) = +26 cases. Chunking mock HF tokenizer bằng MagicMock side-effect (window = 4 chars/token); vector-store dùng Chroma thật trên `tmp_path`; ingest mock `search_arxiv`/`fetch_arxiv_doc`/`fetch_html_doc`. Tổng **68/68 pass** trong 23.5s.
- **Toolchain**: `ruff check` ✓ · `ruff format` ✓ · `mypy strict` ✓ · `pytest 68/68` ✓.
- **`.gitignore`**: thêm `data/chroma/*` (giữ `.gitkeep`) để sqlite/HNSW dev store không bị commit. PDF arXiv đã ignored sẵn qua `data/raw/*`.
- **Decision mới**: ADR-013 (xem `DECISIONS.md`) — cố định Chroma PersistentClient dev, bge-small EN-only dev / bge-m3 prod, trafilatura không playwright cho v1, contextual prepend bằng abstract thay vì LLM summary.

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
│   ├── cli.py                    # CLI: uv run research-assistant (hoặc python -m research_assistant.cli)
│   ├── observability.py          # Langfuse shim: @observe / update_span / start_agent_span / flush
│   ├── agents/
│   │   ├── _llm.py               # ChatAnthropic wrapper + cost estimator + generation spans
│   │   ├── planner.py            # Sonnet 4.5 structured output → SubQuestion[]; @observe span
│   │   ├── synthesizer.py        # Haiku 4.5 → Draft với [^N] citations; @observe span
│   │   ├── critic.py             # Sonnet structured Critique; retry vs advance; @observe span
│   │   └── reporter.py           # deterministic Jinja render + trace_url footer; @observe span
│   ├── tools/
│   │   ├── web_search.py         # Tavily wrapper + fallback ladder; @observe tool span
│   │   ├── academic_search.py    # arXiv metadata → SearchHit; @observe
│   │   ├── fetch_pdf.py          # arXiv id / .pdf URL → Document; httpx + cache + pymupdf
│   │   ├── router.py             # rule-based intent → ToolPlan; ADR-027
│   │   └── vector_search.py     # hybrid BM25 + dense → SearchHit; @observe
│   ├── graph/
│   │   ├── state.py              # ResearchState TypedDict + trace_id/url + 5 Pydantic models
│   │   └── research_graph.py     # retriever: router or corpus+web → CE rerank → evidence; run_research
│   ├── prompts/
│   │   ├── loader.py             # Jinja2 Environment + render helper (StrictUndefined)
│   │   ├── planner_v1.jinja
│   │   ├── synthesizer_system_v1 / user_v1.jinja (+ legacy synthesizer_v1)
│   │   ├── critic_system / user_prefix / user_rest_v1.jinja (+ legacy critic_v1)
│   │   └── reporter_v1.jinja
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── schemas.py            # SourceDoc / Chunk / ChunkMetadata
│   │   ├── chunking.py           # token-aware slicer + contextual prepend
│   │   ├── embedding.py          # sentence-transformers wrapper (default bge-m3)
│   │   ├── vector_store.py       # ChromaStore (PersistentClient dev)
│   │   ├── bm25_index.py         # rank_bm25 BM25CorpusIndex (parallel lexical leg)
│   │   ├── hybrid.py             # stage-1 hybrid fusion 0.5/0.5, top-50+50
│   │   ├── hyde.py              # optional HyDE dense rewrite (weak fused top-2 probe)
│   │   └── ingest/
│   │       ├── __init__.py
│   │       ├── arxiv_source.py   # arxiv SDK + pymupdf
│   │       ├── html_source.py    # trafilatura
│   │       └── loader.py         # SeedConfig.from_yaml + load_seed_corpus
│   ├── safety/__init__.py        # placeholder — Tuần 5
│   └── eval/                     # metrics, retrieval eval, smoke citation, language_quality
├── tests/
│   ├── conftest.py
│   └── unit/
│       ├── test_smoke.py             # import + version check
│       ├── test_config.py            # defaults, flags, validator
│       ├── test_state.py
│       ├── test_prompts.py
│       ├── test_web_search.py
│       ├── test_agents.py
│       ├── test_graph.py             # LangGraph e2e mock; corpus/web + router unit
│       ├── test_router.py            # rule-based intent ToolPlan (ADR-027)
│       ├── test_observability.py
│       ├── test_rag_schemas.py       # 6 cases
│       ├── test_rag_chunking.py      # 8 cases (mocked tokenizer)
│       ├── test_rag_vector_store.py  # 5 cases (real Chroma on tmp_path)
│       ├── test_rag_ingest.py        # 5 cases (mocked fetches)
│       ├── test_hybrid_retrieval.py  # BM25 + hybrid fusion
│       ├── test_vector_search.py     # SearchHit corpus tool
│       ├── test_retrieval_metrics.py # DCG/NDCG/recall
│       └── test_retrieval_load.py    # JSON eval set
├── ui/app.py
├── scripts/
│   ├── week1_smoke.py
│   ├── ingest_seed_corpus.py     # fetch → chunk → embed → upsert + manifest
│   ├── run_retrieval_eval.py     # Recall@10/20, NDCG@10 (default: retrieval_eval_100.json)
│   ├── run_citation_eval.py      # smoke MD → citation_coverage.json (paragraph [^N] coverage)
│   ├── run_language_quality_eval.py  # 5 VI + 5 EN → language_quality.json (LLM judge rubric)
│   └── expand_retrieval_eval.py  # build/validate retrieval_eval_100.json from manifest
├── data/
│   ├── chroma/                   # (gitignored) Chroma PersistentClient store
│   ├── raw/arxiv/                # (gitignored) cached arXiv PDFs
│   └── eval/
│       ├── week1_outputs.md
│       ├── week1_metrics.json
│       ├── ingest_manifest.json  # sau ingest: xem file (chunks / bge-m3) — rebuild khi đổi model
│       ├── retrieval_eval_30.json  # 30 qrels legacy (EN, single-gold)
│       ├── retrieval_eval_100.json  # 100 qrels: 70 EN + 30 VI, multi-gold
│       ├── citation_coverage.json  # smoke batch: mean paragraph citation coverage
│       └── language_quality.json   # VI/EN judge rubric (run_language_quality_eval.py)
├── configs/
│   └── seed_corpus.yaml          # arXiv + HTML blogs (see file; expand when eval grows)
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

### Việc tiếp theo — Tuần 3 (12 mục chi tiết — Session 15+)

**Context**: Tuần 2 đã hoàn 12 mục nhưng phát hiện 3 issue qua smoke: `max_iterations` chạm, cost +6×, latency +13× do Critic+rerank CPU. Eval set vẫn 30-câu, corpus 15-doc quá hẹp. Tuần 3 sửa stack + mở rộng eval.

**Exit criteria**: 100-câu eval + corpus 30-doc; NDCG@10 ≥ 0.65 (rerank); citation coverage ≥ 80%; 0/5 query chạm max_iterations; cost smoke giảm ≥ 30%.

#### A. Eval foundation (mục 1–4)

1. [x] **Mở rộng corpus**: `configs/seed_corpus.yaml` thêm ~15 doc (target 30) — DPR, ColBERT, FiD, REPLUG, GraphRAG, RAFT, Constitutional AI; 2–3 blog VI (FPT/Zalo/VinAI); update comment YAML. Re-run `ingest_seed_corpus.py --rebuild`, commit `ingest_manifest.json`. 
   - **Result (2026-04-24)**: 20/22 docs fetched (15 arXiv + 5 HTML), 2 VI blogs failed (DNS/404 — acceptable). **1006 chunks** (1024-dim bge-m3, up from 766). Embedding ~30 phút CPU. Manifest ghi manifest mới.

2. [x] **Mở rộng retrieval eval**: `data/eval/retrieval_eval_100.json` (70 EN + 30 VI, multi-gold) + `RetrievalEvalItem.language`; `scripts/expand_retrieval_eval.py` (`--write` validate qrels, `--skeleton` từ manifest). ADR-021. `run_retrieval_eval.py` mặc định dùng file 100 câu.

3. [x] **Rerank pipeline eval**: `run_rerank_retrieval_eval` + `iter_source_ids_reranked` (`eval/retrieval.py`); `rerank_hybrid_results` (`rag/reranker.py`); metrics thêm MRR + `precision@5` (`eval/metrics.py`). `run_retrieval_eval.py --with-rerank [--candidate-pool 50]` so sánh A với B; output JSON gồm `ab_delta`. ADR-022.

4. [x] **Citation coverage batch**: `scripts/run_citation_eval.py` → `data/eval/citation_coverage.json` (≥ 80% target). Từ smoke outputs.
   - **Result (2026-04-24)**: `week1_outputs.md` → mean **96.8%** (Q2 91.7%, Q3 92.3%, còn lại 100%); `paragraph_citation_stats` + skip tiêu đề smoke; `apply_full_body_insufficient_guard=False` cho báo cáo ghép nhiều mục. ADR-023.

#### B. Tinh chỉnh stack (mục 5–7)

5. [x] **Fix max_iterations**: `max_iterations = max(8, len(plan) * critic_max_attempts_per_sub_question)` sau planner; `max(planned, prior)` nếu CLI/env đặt cao hơn. ADR-019.

6. [x] **Prompt caching**: `build_lc_messages` + `cache_control` ephemeral; Synthesizer: system tĩnh + parent query; Critic: system tĩnh + user prefix (query) | phần còn lại. `ANTHROPIC_PROMPT_CACHE_ENABLED`. ADR-020. *(Đo cost thực tế khi re-smoke.)*

7. [x] **Smoke A/B + re-run**: CLI `--no-rerank`, `--no-critic`, `--max-iterations N`; state `max_iterations_reached`; `week1_smoke.py --ab` → `ab_compare` + cost delta. ADR-024. *(Chạy `uv run python scripts/week1_smoke.py --ab` khi sẵn API để ghi số liệu thật.)*

#### C. Đa ngôn ngữ + chất lượng (mục 8–9)

8. [x] **VI/EN translation rubric**: `scripts/run_language_quality_eval.py` — 5 query VI + 5 EN, Sonnet judge 4 trục (1–5): accuracy, fluency, terminology, citation → `data/eval/language_quality.json`; `--reports-json` (chỉ judge), `--compare-previous`. Module `eval/language_quality.py` + prompts `language_quality_judge_*_v1.jinja`. ADR-025.

9. [x] **HyDE optional**: `rag/hyde.py` — probe top-2 fused yếu/mơ hồ → Haiku sinh đoạn giả định → `embed_query` → dense leg (BM25 giữ nguyên câu hỏi). `Settings.hyde_enabled=False` mặc định; `run_retrieval_eval.py --with-hyde` đo trên 100 câu + `hyde_delta_vs_baseline`. ADR-026.

#### D. Tài liệu (mục 10)

10. [x] **ADR + docs**: ADR-019 ✓, ADR-020 ✓, ADR-021 (retrieval 100) ✓ (đầy đủ trong `DECISIONS.md`); `PLAN.md` §10 snapshot + checklist Tuần 1–3; `PROGRESS.md` cập nhật session này.

---

### Việc tiếp theo — Tuần 4 (12 mục — A/B/C/D)

**Context**: Tuần 3 đóng. Demo `data/eval/demo_run.md` lộ điểm yếu **nguồn web nhiễu** (job/course pages), **References dính dòng**, **mục 4 thiếu evidence so sánh** → Tuần 4 mở thêm nguồn học thuật + router quyết định tool + Critic 4 trục + factuality eval.

**Exit criteria** (PLAN §10 Tuần 4 + concretize):
- Factuality ≥ **0.80** trên `factuality_eval_20.json` (LLM judge faithfulness, 15 EN + 5 VI).
- Tool router hit **≥ 80%** so gold tool plan (12 case sample).
- ≥ **2/20** report flag conflict đúng (kiểm tay).
- Cost smoke 5 query không tăng > **20%** so baseline Tuần 3.
- `pytest`, `ruff`, `mypy strict` ✓.

**Quyết định user (2026-04-26)**: `fetch_pdf` chỉ arXiv + URL `.pdf` (không Playwright); `compare_sources` heuristic mặc định, LLM-mode `auto` qua router; factuality judge dùng Sonnet 4.5; bộ 20 query AI/ML tự đề xuất theo seed corpus.

#### A. Tool layer — multi-source retrieval (mục 1–3)

1. [x] **`academic_search`** (`src/research_assistant/tools/academic_search.py`, ~150 LoC): wrap `rag/ingest/arxiv_source.search_arxiv` thành tool trả `list[SearchHit]` (`source="academic"`); injectable `search_fn` cho test; `@observe(as_type="tool")` + `update_span`. Validate qua `TypeAdapter[HttpUrl]`, dedupe `arxiv_id`, drop malformed rows, cap snippet 500 chars, year filter qua `date(year_from, 1, 1)`. Test `tests/unit/test_academic_search.py` (10 cases) ✓; full suite **131/131** pass; `ruff` + `mypy strict` ✓.

2. [x] **`fetch_pdf`** (`src/research_assistant/tools/fetch_pdf.py`): arXiv id / `arxiv.org` URL / URL trực tiếp `.pdf`; cache `data/raw/pdf_cache/<sha1>.pdf` (SHA-1 của URL tải); httpx stream 25 MiB cap, timeout 30s; pymupdf extract tối đa 400k chars; trả `Document` (= `SourceDoc` trong `rag/schemas.py`); inject `client` / `extract_pdf_text_fn` cho test; `@observe`. **`httpx`** thêm vào `pyproject.toml`. Test `tests/unit/test_fetch_pdf.py` (18 cases) — `httpx.MockTransport` + patch `pymupdf.open`. Full suite **149/149** pass.

3. [x] **(scope giữ nguyên)** `web_search` / `web_search_with_fallback` — không đổi signature; mỗi hit web có `SearchHit.web_trust_tier` (`high` | `medium` | `low`) theo hostname (`.edu`/`.gov`, danh mục lab/docs vs job/course/Q&A); hàm public `web_trust_tier_for_url()` dùng lại cho router; Langfuse `web_trust_tier_counts` trong `update_span`. Academic/vector/corpus để `web_trust_tier=None`. Test `test_web_search.py` bổ sung.

#### B. Routing & Planner (mục 4)

4. [x] **Tool router rule-based** (`tools/router.py`, `graph/research_graph.py::_route_then_collect`): `classify_intent` + `plan_for_sub_question` → `ToolPlan(primary` + `optional`); merge theo thứ tự tool tới `retrieval_candidate_pool`; `Settings.tool_router_enabled` (mặc định **true**), `tool_router_max_tools=3`; tắt router → `_corpus_then_web_hits`. **`build_graph` / `run_research`**: `academic_search_fn` inject. Trace: `router_intent`, `router_tools`, `n_academic`. **`DECISIONS.md` ADR-027**; **`.env.example`** `TOOL_ROUTER_*`. Test `test_router.py` + `test_graph` (router off cho e2e legacy). Full suite **160** pass.

5. [x] **Planner `suggested_tools`**: `planner_v1.jinja` + `_PlanItemDraft` (1–3 tool: `vector_search` / `web_search` / `academic_search`); `sanitize_planner_suggested_tools`; retriever ghi `planner_suggested_tools`, `router_ordered_tools`, `router_overrode_planner` — chạy retrieval luôn theo rule router khi lệch. `SubQuestion` docstring cập nhật.

#### C. Synthesis quality (mục 6–8)

6. [ ] **`compare_sources`** (`src/research_assistant/tools/compare_sources.py`): mode `heuristic` (regex số/đơn vị) + `llm` (Sonnet structured); chạy trước Critic; output `ConflictReport` → `Critique.conflicts`. Settings `compare_sources_mode: Literal["off","heuristic","auto"]="auto"`. ADR-028.

7. [ ] **Critic 4 trục** (mở rộng `agents/critic.py`): faithfulness (LLM) + completeness + citation coverage (heuristic) + consistency (từ `ConflictReport`); `critic_min_faithfulness=4.0`. Retry như cũ; forced_pass kèm conflict block.

8. [ ] **Reporter** (`prompts/reporter_v1.jinja` + `agents/reporter.py`): References mỗi entry **một dòng**; section optional `## Conflicts noted` khi có conflict severity ≥ medium.

#### D. Eval & docs (mục 9–12)

9. [ ] **`factuality_eval_20.json`** (`data/eval/factuality_eval_20.json`): 15 EN + 5 VI, mỗi query có 3–5 atomic gold claims; `eval/factuality.py` + `scripts/run_factuality_eval.py` (Sonnet judge: supported / contradicted / unsupported). ADR-029.

10. [ ] **Smoke A/B mới**: `week1_smoke.py --with-router --with-compare-sources`; JSON ghi `router_plan_per_subq`, `n_conflicts`.

11. [ ] **ADR**: ADR-028 (compare_sources), ADR-029 (factuality eval); `.env.example` thêm `COMPARE_SOURCES_MODE`.

12. [ ] **Docs**: cập nhật `PLAN.md §10 Tuần 4`, `PROGRESS.md` (log + checklist tick).

---

### Việc tiếp theo — Tuần 2 (session sau)
1. [x] ~~**Fix**: Planner JSON invalid → structured output~~ (done 2026-04-21, session 4).
2. [x] ~~**Improve**: Retriever 0-hits fallback ladder~~ (done 2026-04-21, session 4).
3. [x] ~~**Instrument Langfuse**: `@observe` shim, run_research root span, trace_url footer~~ (done 2026-04-22, session 5; ADR-012).
4. [x] ~~**RAG pipeline (PLAN.md §5 ingestion half)**: ingest arXiv + HTML → chunk 500/50 + contextual prepend → bge-small-en-v1.5 → Chroma persistent~~ (done 2026-04-22, session 6; ADR-013).
5. [x] **Hybrid retrieval stage 1**: BM25 index song song qua `rank_bm25` + combine với dense top-50 (weighted 0.5/0.5 baseline). Thêm `tools/vector_search.py` wrap `ChromaStore.search` theo contract `SearchHit` hiện hữu → expose cho graph retriever.
6. [x] **Integrate vector_search vào graph**: `research_graph.py` gọi `vector_search` trước, ghép với `web_search_with_fallback` theo URL-dedup; `build_graph` / `run_research` nhận `vector_search_fn` (test truyền `[]` để khỏi load embedder).
7. [x] **Cross-encoder rerank** (`BAAI/bge-reranker-v2-m3`, `rag/reranker.py`) — pool `retrieval_candidate_pool` (20) → rerank → `synthesizer_evidence_top_k` (5); `reranker_enabled` + inject `rerank_fn` cho test. ADR-015.
8. [x] **Retrieval eval set** — `data/eval/retrieval_eval_30.json` (qrels `source_id`); `eval/metrics.py` + `eval/retrieval.py`; `scripts/run_retrieval_eval.py` (stage-1 hybrid, không rerank). Baseline mẫu (dev, seed 15 doc): mean recall@10/20 ≈ 0.97, mean NDCG@10 ≈ 0.94. ADR-016.
9. [x] **Critic agent** (draft) — kiểm citation coverage (paragraph `[^N]` metric), structured Sonnet (`critic_v1.jinja`) kiểm sub-question → retry `retriever` hoặc advance; `critic_enabled` tắt cho test. ADR-017.
10. [x] **Swap sang bge-m3** + hướng dẫn re-embed (`ingest_seed_corpus.py --rebuild`); ADR-018; chunking fix tokenizer dài. *Manifest trong repo cập nhật khi bạn chạy ingest local xong.*
11. [x] Chuyển `cli.py` vào `[project.scripts]` — `uv run research-assistant "query"` (PEP 621 `research-assistant = research_assistant.cli:main`; `python -m research_assistant.cli` vẫn tương đương).
12. [x] Re-run full 5-query smoke — `data/eval/week1_metrics.json` + `week1_outputs.md` (trace id/url; corpus vs web + retriever step stats; delta so với file metrics cũ khi chạy).

### Notes / cảnh báo ghi nhớ
- Haiku 4.5 pricing trong `agents/_llm.py._PRICING_USD_PER_MTOK` đang là ước lượng conservative (input $1 / output $5 per MTok). Cần kiểm lại tại `https://www.anthropic.com/pricing` khi có thời gian và cập nhật nếu lệch > 20%.
- Console Windows mặc định cp1252; `scripts/week1_smoke.py` đã force UTF-8 cho stdout, nhưng nếu viết thêm CLI có tiếng Việt cần apply cùng pattern.
- `.env.example` từng bị user paste nhầm key thật (ngày hôm nay); đã revert về placeholder. **Luôn** double-check trước khi commit.
- pymupdf extract text một số paper có figure → ra unicode garbage (e.g. "ҼНҞБЈП" trong ReAct Figure 1). Chấp nhận cho dev; nếu precision@k bị kéo xuống thì thêm filter regex lọc chunk có > 20% non-printable trước embed.
- Embedding CPU chạy 3.4 ch/s — với 766 chunks hết ~4 phút. Khi tăng corpus lên 50–100 docs (≈4-5 nghìn chunks) sẽ 20–25 phút. Nếu có GPU, đổi `EMBEDDING_DEVICE=cuda` trong `.env`.
- Khi đổi `EMBEDDING_MODEL` trong `.env` **PHẢI** rebuild collection (`--rebuild`), vì dimension + distribution khác nhau giữa các model.

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

### 2026-04-21 — Session 4 (Quality fixes — structured output + retriever fallback)
- **Fix 1 — Planner**: `agents/_llm.py` tách helper `_preflight_budget_check` + `_normalise_text`, thêm `invoke_structured_llm(model, prompt, schema, ...)` trả `(parsed: BaseModel, LLMCallResult)` bằng `with_structured_output(include_raw=True)`. Planner rewrite: định nghĩa `_PlanDraft` + `_PlanItemDraft` (no min_length ở draft — validation chặt dời sang `SubQuestion`), hàm `_drafts_to_plan` renumber ids + drop unknown dependency_ids. Prompt đơn giản hoá (bỏ phần "return JSON only"), field description gắn thẳng trên schema.
- **Fix 2 — Retriever**: `tools/web_search.py` thêm `_YEAR_PATTERN`/`_IN_YEAR_PATTERN`, helper `_simplify_query`, và `web_search_with_fallback(query, ...)` theo ladder basic → advanced → simplified-advanced. `graph/research_graph.py` đổi default `search_fn` sang hàm fallback.
- **Tests**: update `test_agents.py` (stub `invoke_structured_llm`, loại test JSON-markdown-wrapped, thêm case drop unknown deps + invalid draft), thêm 4 case fallback ladder trong `test_web_search.py`, refactor `test_graph.py` sang dual-stub (planner structured + synthesizer llm). Total 36 pass (từ 31).
- **Verify thực tế**: `verify_q1.md` (LoRA/QLoRA, VI) plan 5 sub-q · cost $0.0333 · 25 refs. `verify_q4.md` (vector DB, EN) plan 6 sub-q · cost $0.0389 · 30 refs (tất cả sub-q có evidence). Files đã xoá sau verify (không cần giữ như regression baseline — `data/eval/week1_outputs.md` vẫn là baseline chính).
- **Blocker**: không. Còn observation: câu trả lời "Insufficient evidence" trong query 4 xuất phát từ Synthesizer không tìm được fact cụ thể trong snippet — cần Critic/RAG sâu hơn để khắc phục, sẽ vào Tuần 2.
- **Next**: Langfuse `@observe` instrumentation + bắt đầu RAG ingestion.

### 2026-04-22 — Session 6 (Tuần 2 — RAG ingestion pipeline)
- **Scope chốt đầu session** (user confirm 3 lựa chọn): corpus nhỏ ~15 docs (10 arXiv + 5 blogs) để verify pipeline trước; embedding dev `BAAI/bge-small-en-v1.5` (~130 MB, EN-only) thay `bge-m3` để tiết kiệm thời gian download — sẽ swap trước khi thêm VI; wipe Chroma hiện có (file lock nên bỏ wipe, thay bằng `store.reset()` trong script).
- **Deps** (`pyproject.toml`): `chromadb>=0.5.15`, `sentence-transformers>=3.0.0`, `trafilatura>=1.12.0`, `arxiv>=2.1.3`, `pymupdf>=1.24.0`, `pyyaml>=6.0.2`. mypy overrides cho `sentence_transformers.*`, `trafilatura.*`, `arxiv.*`, `pymupdf.*`, `yaml.*` (chromadb có stubs nên dùng `Any` thủ công).
- **Config** (`config.py`): thêm `embedding_model`, `embedding_device`, `chroma_persist_dir`, `corpus_collection`, `chunk_size_tokens` (500), `chunk_overlap_tokens` (50), `raw_docs_dir`.
- **Code** (~720 LoC src + ~320 LoC tests):
  - `rag/schemas.py` — Pydantic `SourceDoc` / `Chunk` / `ChunkMetadata`; `SourceDoc.make_source_id(url)` = `h_<sha1[:16]>` cho HTML; `ChunkMetadata.to_chroma()` flatten scalars (Chroma không nhận nested).
  - `rag/chunking.py` — `ChunkingConfig(model_id, 500, 50)`, tokenize bằng HF fast tokenizer của embedding model (lru_cache 4 models); slice token ids với step=size-overlap, decode body qua offset_mapping; `_iter_sections` dò `^#+ ` và ALL-CAPS ngắn; `_summarise_for_prepend` = `doc.summary` | 2 câu đầu | title, cap char. Embedding text = `"[Title] summary\n\nbody"`; body giữ nguyên cho citation. Raise nếu overlap≥size; skip empty docs (warn).
  - `rag/embedding.py` — `EmbeddingModel(model_id, device, batch_size=32)`; `_load_model` lru_cache + Lock; `embed_documents` normalise cosine; `embed_query` tự prefix `"Represent this sentence for searching relevant passages: "` khi model chứa "bge" (bỏ qua reranker). Return `np.ndarray[float32]`.
  - `rag/vector_store.py` — `ChromaStore(persist_dir, collection, distance="cosine")`; `get_or_create_collection(metadata={"hnsw:space": "cosine"})`; `upsert_chunks(chunks, embeddings)` idempotent với chunk_id; `search(query_vec, top_k, where)` → `list[SearchResult(chunk_id, body, metadata, distance)]`; `reset()` wrap `delete_collection` để `--rebuild`.
  - `rag/ingest/arxiv_source.py` — `search_arxiv(query, max_results, date_from, categories)` build Tavily-style Lucene filter; `fetch_arxiv_doc(id, cache_dir)` cache PDF theo id, pymupdf extract tối đa 400k chars. `_iter_results` tách thành helper để test monkeypatch.
  - `rag/ingest/html_source.py` — trafilatura `fetch_url` + `extract(txt, favor_precision, include_tables)`; `extract_metadata` → title/author/date/summary. Không playwright (cp PLAN §3 "playwright khi JS-heavy" → tuần sau nếu cần).
  - `rag/ingest/loader.py` — `SeedConfig.from_yaml(path)` parse `arxiv.{ids,queries}` + `html[{url,title}|str]`, skip malformed entries với warning. `load_seed_corpus(config, arxiv_cache_dir)` dedup arXiv ids, collect failures per-source (không crash batch).
- **`configs/seed_corpus.yaml`**: 10 arXiv paper (Lewis RAG, Self-RAG, RAG Survey, LoRA, QLoRA, ReAct, Self-Refine, Llama 2, DeepSeek-R1, GraphRAG) + 5 blog (Anthropic Contextual Retrieval / Building Effective Agents, OpenAI Structured Outputs, HF ray-rag, LangChain LangGraph).
- **`scripts/ingest_seed_corpus.py`** (180 LoC): argparse `--config`, `--rebuild`, `--limit`, `--manifest`; tune `urllib3/httpx/chromadb.telemetry` về WARNING; manifest JSON với sources + timings + failures. Exit 2 nếu 0 chunks.
- **Run thực tế** (`uv run python scripts/ingest_seed_corpus.py --rebuild`):
  - Fetch: 15/15 docs, 47.2s (arXiv metadata delay_seconds=3.0 mỗi request = bottleneck chính).
  - Chunk: 766 chunks, 22.3s (tokenizer load lần đầu tính trong đó).
  - Embed: 384-dim vectors, 224.3s, 3.4 ch/s trên CPU (bge-small lần đầu tự download từ HF).
  - Upsert Chroma: 3.6s, collection `ai_ml_corpus_v1` = 766 chunks.
  - Manifest: `data/eval/ingest_manifest.json`.
- **Retrieval smoke (3 query, top-3)**:
  - "LoRA and trainable parameters" → LoRA + QLoRA papers, cosine-dist 0.256–0.259.
  - "Contextual retrieval RAG" → Anthropic blog top-1 (dist 0.132), RAG Survey #2/#3.
  - "ReAct reasoning loop" → cả top-3 từ paper ReAct (dist 0.177–0.193).
- **Tests**: `test_rag_schemas` (6), `test_rag_chunking` (8, mock HF tokenizer), `test_rag_vector_store` (5, Chroma thật trên tmp_path), `test_rag_ingest` (5, mock fetches) → 26 cases mới. Tổng **68/68 pass** trong 23.5s.
- **Toolchain**: `ruff check` ✓ · `ruff format` ✓ · `mypy strict` ✓ (phải thay relative imports → absolute, lift một số chromadb/pymupdf/sentence-transformers sang `Any` vì stubs hoặc lack thereof) · `pytest` 68/68 ✓.
- **`.gitignore`**: thêm `data/chroma/*` (giữ `.gitkeep`) — chroma sqlite + HNSW bin không commit. PDF arXiv trong `data/raw/arxiv/` đã ignored qua `data/raw/*`. `ingest_manifest.json` giữ commit được (summary dùng để audit).
- **Blocker**: không. Observation: pymupdf extract thỉnh thoảng đẻ unicode garbage cho chunk chứa figure/table PDF (ví dụ `ҼНҞБЈП` trong ReAct). Không blocking retrieval baseline nhưng có thể pollute synthesis snippets — ghi note trong PROGRESS, filter theo non-printable ratio nếu eval kéo xuống.
- **ADR mới**: `DECISIONS.md` ADR-013 — Chroma PersistentClient dev / Qdrant prod; bge-small EN dev / bge-m3 prod; trafilatura không playwright v1; contextual prepend dùng abstract/first-2-sentences thay LLM summary (zero cost).
- **Next**: BM25 index song song qua `rank_bm25` + hybrid retrieval stage 1 → wrap thành `tools/vector_search.py` emit `SearchHit` → integrate vào `research_graph.py` (ưu tiên corpus trước Tavily). Retrieval eval 30 câu. Xem § "Việc tiếp theo".

### 2026-04-22 — Session 5 (Tuần 2 — Langfuse full instrumentation)
- **Shim design** (`src/research_assistant/observability.py`, +283 LoC): lazy-load `Langfuse(...)` client từ `Settings` (không dựa vào `os.environ` — `pydantic-settings` chỉ populate Settings object), cached singleton. `observe(name, as_type, capture_input, capture_output)` wrapper: nếu `is_enabled()` False thì passthrough transparent; nếu True thì cache decorated function per-call, import `langfuse.observe` lazily. `update_span`, `update_generation`, `update_trace_io`, `flush`, `current_trace_id`, `current_trace_url`, `start_agent_span` đều no-op khi disabled. Test confirm `@wraps` giữ nguyên `__name__` / docstring.
- **Instrumentation pass**:
  - `_llm.py`: `invoke_llm` + `invoke_structured_llm` → `as_type="generation"`, set `model`, `input` (prompt tail), `output` (text tail), `usage_details` (`input`/`output`/`total` tokens), `cost_details` (`input`/`output`/`total` USD).
  - `tools/web_search.py`: `web_search` + `web_search_with_fallback` → `as_type="tool"`, set query / results_count / search_depth / stage ladder result.
  - `graph/research_graph.py`: đổi chỗ `retriever_node` factory để decorator `@observe(as_type="retriever")` không bị reshadow; thêm `run_research(query, output_language, max_iterations, per_query_cap_usd)` làm root span `@observe(name="research_agent", as_type="agent")`. `run_research` capture `trace_id` + `trace_url` ngay tại entry, inject vào `initial` state (để consistent qua toàn graph vì LangGraph có thể phá OTel context giữa nodes), invoke graph, `update_trace_io` với summary, `flush()` trước khi return.
  - Nodes `planner`/`synthesizer`/`reporter` được `@observe` span riêng; planner làm fallback capture `trace_id/url` nếu run ngoài `run_research`.
  - `graph/state.py`: thêm `trace_id: str | None` + `trace_url: str | None` vào `ResearchState` + `new_state()`.
  - `prompts/reporter_v1.jinja`: footer `{% set has_trace = (trace_url is defined) and trace_url %}{% if has_trace %}*Full trace on Langfuse*: [URL]{% endif %}`. `agents/reporter.py.build_report` thêm tham số `trace_url`.
  - `cli.py`, `scripts/week1_smoke.py`: migrate sang `run_research(...)` thay vì gọi `build_graph() + invoke`. `cli.py` force stdout UTF-8 trên Windows (tránh `UnicodeEncodeError` trên em-dash / diacritics).
  - `ui/app.py`: `graph.stream(...)` ôm trong `start_agent_span("research_agent")` context; sau stream set `final_state["trace_url"] = current_trace_url()`; `flush()` cuối handler.
- **Test hermeticity**: thêm `tests/conftest.py` autouse fixture (`monkeypatch` set `LANGFUSE_PUBLIC_KEY=""`/`LANGFUSE_SECRET_KEY=""` + `get_settings.cache_clear()`) → loại hoàn toàn 401 spam. Thêm `tests/unit/test_observability.py` (5 cases).
- **Fix `_client()` init**: bug phát hiện khi verify — `langfuse.get_client()` đọc env vars nên thấy empty dù `Settings.langfuse_enabled` True. Sửa: instantiate explicit `Langfuse(public_key=s.langfuse_public_key.get_secret_value(), secret_key=..., host=...)`, cache singleton, gọi `_client()` ngay đầu mỗi observed wrapper để SDK `@observe` sau đó tìm thấy client toàn cục.
- **Verify run**: `uv run python -m research_assistant.cli "What is LangGraph state persistence?" --language en --out data/eval/verify_langfuse3.md` → cost $0.0224 · trace_steps 12 · plan 5 sub-q · `trace_id=995f6a8874ce7bcf7735711c23e0966a`. Report footer có link Langfuse. `auth_check()` trả `True` sau khi instantiate qua shim. Dashboard kiểm tra: root `research_agent` → lồng `planner`, `retriever`×N, `synthesizer`, `reporter` + `generation` (`invoke_structured_llm` / `invoke_llm`) + `tool` (`web_search_with_fallback`).
- **Toolchain final**: `ruff check` ✓ · `ruff format` ✓ · `mypy` strict ✓ · `pytest` 42/42 ✓ trong 1.74s.
- **Blocker**: không.
- **Next**: khởi động RAG pipeline (PLAN.md §5) — ingestion arXiv/HTML → chunking → bge-m3 embedding → Chroma dev store, rồi hybrid BM25 + dense stage 1.

### 2026-04-22 — Session 7 (Tuần 2 — Hybrid stage 1 + vector_search tool)
- **Deps**: `rank-bm25>=0.2.2` (`pyproject.toml`); mypy override `rank_bm25.*`.
- **`ChromaStore`**: `fetch_all_documents` / `get_by_ids` để BM25 đồng bộ toàn corpus và hydrate ứng viên chỉ có ở nhánh BM25.
- **`rag/bm25_index.py`**: `BM25CorpusIndex` (Okapi, `tokenize_for_bm25`), build từ `from_chroma` hoặc test rows; placeholder token cho body rỗng.
- **`rag/hybrid.py`**: `hybrid_search_stage1` — dense `top_k=50` + BM25 `top_n=50`, min-max từng chân, fuse `(w_d·d + w_b·b)/(w_d+w_b)` (mặc định 0.5/0.5), `final_top_k` hits; `HybridSearchResult`.
- **`tools/vector_search.py`**: `vector_search(...)` → `list[SearchHit]` (`source="corpus"`), `TypeAdapter(HttpUrl)` cho URL; cache BM25 theo `(collection_name, count)`; `clear_vector_search_cache()` cho test; `@observe` tool + `update_span`.
- **Tests**: `test_hybrid_retrieval.py` (6), `test_vector_search.py` (2). Tổng **75/75** `pytest`; `ruff` + `mypy` strict ✓.
- **Next**: Session 8 — wire graph (đã xong trong Session 8 log dưới).

### 2026-04-22 — Session 8 (Graph retriever — corpus + web)
- **`research_graph.py`**: `_corpus_then_web_hits` — `vector_search` (top_k = budget sub-q) trước, dedup URL, rồi Tavily chỉ số slot còn thiếu; trace/Langfuse `n_corpus` / `n_web` / `retrieval_path` (`corpus_only` | `web_only` | `corpus_then_web`). `build_graph` & `run_research` thêm `vector_search_fn` (mặc định `vector_search`). Lỗi web nghiêm trọng → vẫn giữ phần corpus nếu có.
- **Tests**: `test_graph.py` dùng `vector_search_fn` trả `[]`; thêm 2 test cho `_corpus_then_web_hits` (merge + no web khi đủ 5 corpus).
- **Verify**: `pytest` 77/77, ruff, mypy.

### 2026-04-22 — Session 10 (Retrieval eval 30)
- **`data/eval/retrieval_eval_30.json`**: 30 query EN, 1 gold `source_id`/câu (khớp manifest ingest).
- **`eval/metrics.py`**: DCG/NDCG@k, recall (doc-in-top-k-chunks), `per_query_metrics`.
- **`eval/retrieval.py`**: `load_retrieval_eval`, `run_hybrid_retrieval_eval` (stage-1 hybrid only).
- **`scripts/run_retrieval_eval.py`**: in-memory BM25, embed query/chunk; in `--out` JSON. Dev run ~57s, mean recall@10/20 **0.967**, mean NDCG@10 **0.945** (có thể thay đổi theo model/corpus).
- **Tests**: `test_retrieval_metrics`, `test_retrieval_load` · **87/87** pytest. ADR-016.

### 2026-04-22 — Session 9 (Stage-2 cross-encoder rerank)
- **`config.py`**: `reranker_enabled`, `reranker_model` (`BAAI/bge-reranker-v2-m3`), `reranker_device`, `retrieval_candidate_pool=20`, `synthesizer_evidence_top_k=5`.
- **`rag/reranker.py`**: `rerank_search_hits` (CrossEncoder, passage = raw_content|snippet), min--max score; **`research_graph`**: sau merge corpus+web gọi `rerank_fn` (mặc định từ setting); `build_graph`/`run_research` thêm `rerank_fn`, `retrieval_candidate_pool`. Trace: `n_pool`, `n_after_rerank`.
- **Tests**: `test_reranker.py` (4), `test_graph` dùng `rerank_fn` slice + `pool=5`. **81/81** pytest. ADR-015, `.env.example` gợi ý biến rerank.

### 2026-04-23 — Session 12 (bge-m3 default)
- **`config.py`**: default `embedding_model=BAAI/bge-m3`; docstring ADR-018.
- **`rag/embedding.py`**: default ctor + docstring khớp m3.
- **`rag/chunking.py`**: sau `AutoTokenizer.from_pretrained`, set `model_max_length=1_000_000` để tokenize full PDF cho offset chunking (tránh cảnh báo / truncate 8192 của tokenizer m3).
- **`DECISIONS.md`**: **ADR-018**; ADR-013 ghi chú default hiện tại là m3.
- **`.env.example`**: block `EMBEDDING_MODEL` / `bge-small` override.
- **`tests/unit/test_config.py`**: assert default `BAAI/bge-m3`.
- **Ops**: committer chạy `uv run python scripts/ingest_seed_corpus.py --rebuild` để làm mới `data/chroma/` + `ingest_manifest.json` (ingest nền có thể vẫn đang embed CPU).

### 2026-04-23 — Session 11 (Critic draft)
- **`graph/state.py`**: thêm `Critique`, `critiques` / `critic_attempts` / `synth_critic_feedback` / `critic_route_next` trên `ResearchState`.
- **`config.py`**: `critic_enabled`, `critic_max_attempts_per_sub_question` (default 2), `critic_min_paragraph_citation_coverage` (0.9).
- **`agents/critic.py`**: `paragraph_citation_coverage`, `critic_node`, `critic_route_edge`; kết hợp metric + `_CritiqueDraft` structured; lỗi LLM → forced pass.
- **`prompts/critic_v1.jinja`**, **`synthesizer_v1.jinja`**: feedback block (StrictUndefined-safe).
- **`agents/synthesizer.py`**: không tự tăng `current_sub_question_index`; nhận `synth_critic_feedback`.
- **`graph/research_graph.py`**: `synthesizer → critic` → `{retriever|tick}`; `tick` chỉ trace (iterations do critic).
- **Tests**: `test_critic.py`, cập nhật `test_graph` (`CRITIC_ENABLED=false`), `test_agents`, `test_prompts`.
- **Toolchain**: `ruff` / `mypy strict` / `pytest` **92/92** pass.
- **Next**: bge-m3 swap, `[project.scripts]` CLI, full smoke 5 query.

### 2026-04-23 — Session 13 (Console script entry)
- **`pyproject.toml`**: `[project.scripts]` `research-assistant` → `research_assistant.cli:main` — `uv run research-assistant "…"` khớp `prog=` argparse.
- **`cli.py`**: docstring Usage cập nhật (ưu tiên lệnh script, ghi thêm dạng `-m`).

### 2026-04-23 — Session 14 (Full smoke re-run + retrieval metrics in JSON)
- **`scripts/week1_smoke.py`**: `_aggregate_retrieval_stats` (evidence `corpus`/`web` + `retriever_details` từ `StepLog`); đọc `week1_metrics.json` cũ trước khi ghi để thêm `delta_vs_previous_file` (cost + wallclock); stdout in một dòng delta.
- **Chạy thực tế 5 query**: tổng $0.7565 · 2429.5s; mỗi query có `langfuse_trace_id` / `langfuse_trace_url`. 3 query chạm `max_iterations=8` (critic+retry ăn iteration).
- **Next**: tùy ưu tiên — tăng `max_iterations` hoặc giảm critic retry; hoặc re-run `run_retrieval_eval.py` sau ingest.

### 2026-04-24 — Session 15 (Tuần 3 planning + corpus mục 1)
- **PROGRESS.md**: ghi 12 mục Tuần 3 chi tiết (A–D: eval foundation, stack tuning, language, docs).
- **mục 1 — Corpus expand**: `seed_corpus.yaml` thêm 5 arXiv (DPR, ColBERT, HotpotQA, Constitutional AI, multi-hop) + 2 VI blogs (failed). Rebuild → 1006 chunks (bge-m3), 1917s total embed. Commit thành công.
- **Next**: mục 2–4 (retrieval eval 100 câu, rerank pipeline, citation batch).

### 2026-04-24 — Session 16 (Retrieval eval 100 + expand script)
- **`data/eval/retrieval_eval_100.json`**: q01–q70 EN, q71–q100 VI; multi-`relevant_source_ids`; validate với `ingest_manifest.json`.
- **`scripts/expand_retrieval_eval.py`**: `--write` / `--skeleton`; ADR-021.
- **`eval/retrieval.py`**: `RetrievalEvalItem.language` (mặc định `en` cho `retrieval_eval_30.json`).
- **`run_retrieval_eval.py`**: default eval file → `retrieval_eval_100.json`.
- **Tests**: `test_retrieval_load` cập nhật cho 100 + legacy 30.

### 2026-04-24 — Session 17 (Rerank pipeline eval + MRR/P@5)
- **`eval/metrics.py`**: `reciprocal_rank_first_relevant`, `precision_at_k`; `per_query_metrics` thêm `mrr`, `precision@5`.
- **`rag/hybrid.py`**: `hybrid_result_to_search_hit` (dùng chung với `vector_search`).
- **`rag/reranker.py`**: `rerank_hybrid_results` → trả `HybridSearchResult` cho eval.
- **`eval/retrieval.py`**: `iter_source_ids_reranked`, `run_rerank_retrieval_eval`.
- **`scripts/run_retrieval_eval.py`**: `--with-rerank`, `--candidate-pool`, in A/B + `ab_delta`.
- **Next**: mục 4 (citation coverage batch).

### 2026-04-28 — Session 29 (Planner suggested_tools advisory)
- **Prompt** `planner_v1.jinja` + **`_PlanItemDraft.suggested_tools`**: gợi ý công cụ; **`sanitize_planner_suggested_tools`**; retrieval log conflict vs router.
- **Next**: Tuần 4 C — `compare_sources` / Critic nếu theo roadmap.

### 2026-04-28 — Session 28 (Tuần 4 B1 tool router / ADR-027)
- **`tools/router.py`**: `classify_intent`, `build_tool_plan`, `plan_for_sub_question`, `ToolPlan` (primary + optional list).
- **`graph/research_graph.py`**: `_route_then_collect`; `tool_router_enabled` / `tool_router_max_tools` từ Settings; `build_graph` + `run_research` nhận `academic_search_fn`; trace `n_academic`, `router_intent`, `router_tools`. Graph tests: `TOOL_ROUTER_ENABLED=false` cho e2e legacy.
- **`config.py`**, **`.env.example`**, **`DECISIONS.md` ADR-027**.
- **Tests**: `tests/unit/test_router.py`, extension `test_graph.py`; **pytest 160** pass.

### 2026-04-27 — Session 27 (Tuần 4 A3 web_search trust tiers)
- **`SearchHit.web_trust_tier`**: optional `high`/`medium`/`low`; `web_search` gán theo domain heuristic + `web_trust_tier_for_url()`; span output thêm `web_trust_tier_counts`.
- **Next**: B1 `tools/router.py` tiêu thụ `web_trust_tier`.

### 2026-04-27 — Session 26 (Tuần 4 A2 fetch_pdf)
- **`tools/fetch_pdf.py`**: `_resolve_fetch_target` (arXiv mới + URL cũ `cs.XX/1234567`, link abs/pdf, hoặc `https` path kết thúc `.pdf`); `_stream_download_pdf` kiểm `%PDF`; cache key SHA-1 URL; `FetchPdfError` cho input/HTTP/oversize/non-PDF/empty text.
- **`rag/schemas.py`**: `Document: TypeAlias = SourceDoc` (PLAN §4.1).
- **Deps**: `httpx>=0.27.0`.
- **Tests**: `tests/unit/test_fetch_pdf.py` — mock transport, cache hit, oversize, HTTP 404, patch `pymupdf.open` cho extract + truncate + empty body.
- **Toolchain**: `ruff` ✓ · `mypy strict` ✓ · `pytest 149/149` ✓.
- **Next**: Tuần 4 B1 — tool router (`tools/router.py`) hoặc tích hợp `fetch_pdf` vào graph nếu roadmap yêu cầu trước.

### 2026-04-26 — Session 25 (Tuần 4 plan + A1 academic_search)
- **PROGRESS.md**: ghi 12 mục Tuần 4 (A/B/C/D), exit criteria, quyết định user 2026-04-26 (fetch_pdf scope, compare_sources mode, Sonnet judge, 20 query AI/ML).
- **A1 `academic_search`** (`src/research_assistant/tools/academic_search.py`): wrap `search_arxiv` → `SearchHit(source="academic")`; injectable `search_fn`, `@observe(as_type="tool")`, dedupe arXiv id, cap snippet 500, year filter, validate URL via `TypeAdapter[HttpUrl]`. Drop malformed rows / non-list payload với log warning thay vì crash batch.
- **Tests**: `tests/unit/test_academic_search.py` (10 cases) — happy path, clamp max, year_from → date, dedupe, malformed rows, snippet cap, non-list payload. Mock `_RecordingSearch` capture kwargs để verify wrapper truyền đúng arXiv backend.
- **Toolchain**: `ruff check` ✓ · `ruff format` ✓ (2 file) · `mypy strict` ✓ (38 src) · `pytest 131/131` ✓ trong 41.3s.
- **Next**: A2 — `fetch_pdf` (arXiv id + URL `.pdf` direct, cache `data/raw/pdf_cache/`, pymupdf extract, timeout 30s, cap 25 MB / 400k chars).

### 2026-04-25 — Session 24 (ADR + roadmap docs)
- Đồng bộ **PLAN.md §10**: snapshot 2026-04-25, checklist Tuần 1–3 [x], mô tả Tuần 3 mở rộng (eval 100, rerank A/B, HyDE, language quality, ADR-019..026); exit criteria trỏ về `PROGRESS.md` / script eval.
- Xác nhận **DECISIONS.md** đã có ADR-019, ADR-020, ADR-021 (và ADR-022..026 liên quan Tuần 3); không cần entry ADR mới cho mục 10.
- **Next**: Tuần 4 (PLAN §10) — `academic_search` / `fetch_pdf`, tool router, `compare_sources`, factuality eval.

### 2026-04-25 — Session 23 (HyDE)
- **`rag/hyde.py`**: `dense_embedding_for_retrieval`, probe thresholds từ Settings, sinh passage bằng Haiku.
- **`vector_search`**, **`eval/retrieval.py`** (`use_hyde`, `n_hyde_triggers`); **`run_retrieval_eval.py --with-hyde`**; **`.env.example`**.
- **`test_hyde.py`**, **ADR-026**.

### 2026-04-25 — Session 22 (Language quality rubric)
- **`run_language_quality_eval.py`**: 10 seed queries (5 `vi` + 5 `en`), `run_research` + structured judge; JSON baseline + optional `baseline_comparison`.
- **`eval/language_quality.py`**, judge Jinja v1, **`test_language_quality.py`**.
- **ADR-025** trong `DECISIONS.md`.

### 2026-04-25 — Session 21 (Smoke flags + A/B)
- **ADR-024**: CLI `--no-rerank` / `--no-critic`; `no_cross_encoder_rerank_fn`; `critic_enabled_override` + planner planned cap khi tắt Critic; reporter ghi `max_iterations_reached`; `week1_smoke.py` argparse, `--ab`, payload `mode` / `run_flags` / `ab_compare`.
- **Tests**: `test_state`, `test_graph`, `test_agents` (planner cap khi critic off).
- **Next**: Tuần 3 mục 8–9 (language eval, HyDE).

### 2026-04-24 — Session 20 (ADR-020 prompt caching)
- **`build_lc_messages`**, `invoke_llm` / `invoke_structured_llm`: `cacheable_user_prefix`, `use_prompt_cache`; Haiku synthesizer + Sonnet critic.
- **Templates**: `synthesizer_system_v1` / `synthesizer_user_v1`, `critic_system_v1`, `critic_user_prefix_v1`, `critic_user_rest_v1`; legacy `*_v1.jinja` dùng `{% include %}`.
- **`synthesize_one`**: tham số `user_query`; **`Settings.anthropic_prompt_cache_enabled`**.
- **Tests**: `test_llm_messages.py`, cập nhật `test_prompts`.

### 2026-04-24 — Session 19 (ADR-019 max_iterations)
- **`planned_max_iterations`** (`config.py`); **`planner_node`** set `max_iterations` + trace details; `max(prior, planned)` khi user/CLI tăng trần.
- **`Settings.max_iterations`** `le=64`; `.env.example` ghi chú.
- **Tests**: `test_graph` patch planner cho case cap=1; `test_agents` + `test_config`; ADR-019 trong `DECISIONS.md`.

### 2026-04-24 — Session 18 (Citation coverage batch)
- **`paragraph_citation_stats`** (`agents/critic.py`): tham số `apply_full_body_insufficient_guard` (mặc định giữ hành vi Critic); `skip_paragraph` giữ nguyên.
- **`eval/smoke_citation.py`**: parse `# Query N (VI|EN):`, tách body (bỏ plan + References), skip heading/placeholder/đoạn disclaimer; gọi stats với guard tắt.
- **`scripts/run_citation_eval.py`**: `--smoke-md`, `--out`, `--target-mean`, `--strict-exit`.
- **Chạy**: `week1_outputs.md` → mean coverage **0.968**, `meets_target` ✓ (target 0.8).
- **Tests**: `test_smoke_citation.py`, mở rộng `test_critic.py`.
- **Next**: Tuần 3 mục 5–7 (max_iterations, prompt cache, smoke flags).

<!-- Khi kết thúc session, thêm entry mới theo format:
### YYYY-MM-DD — Session N (Tên phase)
- Việc đã làm
- Việc chưa xong
- Blocker (nếu có)
- Next action cụ thể
-->
