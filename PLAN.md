# Research Assistant Agent — Build Plan

> **Mục đích file này**: Bản kế hoạch chi tiết để xây dựng Research Assistant Agent. Dùng làm nguồn sự thật duy nhất (single source of truth) để handoff giữa các phiên làm việc với AI assistant.
>
> **Cách dùng khi bắt đầu conversation mới**:
> 1. Đọc `PROGRESS.md` trước để biết đang ở đâu, làm tiếp việc gì.
> 2. Đọc `PLAN.md` (file này) khi cần chi tiết kiến trúc/quyết định.
> 3. Đọc `DECISIONS.md` khi cần hiểu "tại sao" chọn giải pháp X thay vì Y.

---

## 0. TL;DR

Xây **AI Agent** tự động research một chủ đề: nhận câu hỏi → lập kế hoạch → tìm kiếm đa nguồn (web, arXiv, PDF) → tổng hợp có citation → xuất báo cáo Markdown/PDF.

- **Nguyên tắc nền tảng**: Stanford "How to Build AI Agents" (xem `AI_building_principles.png`).
- **Kiến trúc**: Multi-agent theo ReAct loop (Observe → Plan → Act), RAG 2 giai đoạn (recall → precision), tool calling chuẩn hoá (MCP-ready).
- **Triết lý**: *Start small, start smart* — v1 dùng model mạnh + tool đơn giản, tối ưu sau.
- **Timeline**: 6 tuần tới v1 beta.

---

## 1. Mục tiêu & Phạm vi

### 1.1. Mục tiêu sản phẩm (v1)

Agent có khả năng:
1. Nhận một **research question** (VD: *"So sánh LoRA và QLoRA cho fine-tuning LLM năm 2025"*).
2. **Lập kế hoạch** phân rã thành 3–7 sub-questions.
3. **Thu thập** dữ liệu từ nhiều nguồn (web search, arXiv, PDF, corpus nội bộ).
4. **Tổng hợp, đối chiếu** thông tin, flag mâu thuẫn giữa các nguồn.
5. **Xuất báo cáo** Markdown/PDF có citation đầy đủ `[^N]`.

### 1.2. Non-goals (KHÔNG làm ở v1)

- Không tự sinh kết quả thí nghiệm mới.
- Không crawl web quy mô lớn (dùng search API sẵn có).
- Không multi-user / real-time collaboration.
- Không fine-tune LLM (dùng base model + prompting).

### 1.3. Success Metrics (đo ở cuối tuần 6)

| Metric | Target v1 | Đo bằng |
|---|---|---|
| Factual accuracy | ≥ 85% | Human eval, 20 queries |
| Citation coverage | ≥ 90% | % câu claim có ref / tổng |
| Retrieval Recall@10 | ≥ 0.8 | Eval set 100 câu |
| NDCG@10 sau re-rank | ≥ 0.7 | Eval set 100 câu |
| End-to-end latency | < 3 phút | p95 |
| Chi phí/report (prod target) | < $0.50 | Trung bình trên 20 queries |
| Per-query cap (dev) | ≤ $0.30 | Hard cap, ADR-011 |
| Tổng test budget | $10 USD | ADR-011, alert tại $7, hard stop tại $10 |
| Ngôn ngữ output | Tiếng Việt | ADR-007 (prompt vẫn tiếng Anh) |

---

## 2. Kiến trúc tổng thể

### 2.1. High-level diagram

```
┌──────────────┐   ┌─────────────────────────────────┐   ┌──────────────┐
│   User Query │──▶│   Orchestrator (ReAct Loop)     │──▶│  Final Report│
└──────────────┘   │   Observe → Plan → Act          │   │   (MD/PDF)   │
                   └───┬─────────────┬────────────┬──┘   └──────────────┘
                       │             │            │
                       ▼             ▼            ▼
                 ┌──────────┐  ┌──────────┐ ┌──────────┐
                 │  Tool    │  │  RAG     │ │ Synth &  │
                 │  Router  │  │ Pipeline │ │ Critique │
                 │  (MCP)   │  │ (2-stage)│ │  Agents  │
                 └────┬─────┘  └────┬─────┘ └────┬─────┘
                      │             │            │
                      ▼             ▼            ▼
                 [Search APIs]  [Vector DB]  [LLM judge]
                 [Scraper]      [BM25 idx]
                 [arXiv/PDF]    [Re-ranker]
```

### 2.2. ReAct loop

Mỗi iteration:
- **Observe**: Parse query + state (sub-questions đã trả lời, gaps).
- **Plan**: Quyết định sub-question tiếp theo + tool phù hợp.
- **Act**: Gọi tool → append kết quả vào state.
- **Stop condition**: đủ evidence cho mọi sub-question HOẶC đạt `max_iterations=8`.

### 2.3. Multi-agent roles

| Agent | Trách nhiệm | Input | Output |
|---|---|---|---|
| **Planner** | Phân rã query thành sub-questions, lập plan | User query | `list[SubQuestion]` + strategy |
| **Retriever** | Search đa nguồn, RAG 2 giai đoạn | Sub-question | Top-k chunks có score |
| **Synthesizer** | Viết trả lời, gắn citations | Sub-question + chunks | Đoạn văn + refs |
| **Critic** | Đánh giá độ đầy đủ, flag hallucination, mâu thuẫn | Draft + evidence | Feedback + điểm 1–5 |
| **Reporter** | Ghép thành báo cáo cuối, format MD/PDF | Drafts đã critic | Final report |

---

## 3. Tech Stack (quyết định chính thức)

> Xem `DECISIONS.md` cho lý do chọn từng cái.

| Lớp | Công nghệ | Version/Ghi chú |
|---|---|---|
| Language | Python | 3.11+ |
| Package manager | `uv` | (nhanh hơn pip/poetry) |
| Orchestration | **LangGraph** | ưu tiên; CrewAI là backup |
| LLM (Planner/Critic) | **Claude Sonnet 4.5** (`claude-sonnet-4-5`) | chốt ADR-008 |
| LLM (Synthesizer) | **Claude Haiku** (`claude-haiku-4-5` / fallback `claude-3-5-haiku-latest`) | cùng provider, tận dụng prompt caching |
| Embeddings | `BAAI/bge-m3` | đa ngôn ngữ, tốt cho tiếng Việt |
| Vector DB | **Qdrant** (prod), Chroma (dev) | hybrid search native |
| Keyword search | `rank_bm25` (v1) → Elasticsearch (v2) | |
| Re-ranker | `BAAI/bge-reranker-v2-m3` | cross-encoder |
| Web search | **Tavily API** (primary), SerpAPI (backup) | Tavily tối ưu cho agent |
| Scraping | `trafilatura` + `playwright` (cho JS sites) | |
| PDF/arXiv | `arxiv` + `pymupdf` | |
| Tool protocol | **MCP** (Model Context Protocol) | v2 mới expose MCP server |
| Observability | **Langfuse Cloud** (`cloud.langfuse.com`, free tier) | ADR-009 |
| Storage | PostgreSQL (meta) + local FS (raw docs) | S3 khi deploy |
| Caching | Redis + prompt caching (Anthropic/OpenAI) | |
| UI | Streamlit (v1) → Next.js (v2) | |
| Config | YAML + `pydantic-settings` | |
| Testing | `pytest` + `pytest-asyncio` | |
| Linting | `ruff` + `mypy` | strict mode |

---

## 4. Thiết kế Tools (MCP-compliant)

3 nhóm chuẩn Stanford: **Information retrieval**, **Computation**, **Action execution**.

### 4.1. Information retrieval

```python
web_search(query: str, max_results: int = 10, time_range: str = "year") -> list[SearchHit]
academic_search(query: str, max_results: int = 10, year_from: int | None = None) -> list[Paper]
fetch_url(url: str) -> Document           # trafilatura + playwright fallback
fetch_pdf(url_or_path: str) -> Document   # pymupdf
vector_search(query: str, top_k: int = 20, filters: dict | None = None) -> list[Chunk]
```

### 4.2. Computation

```python
summarize(text: str, focus: str | None = None, max_tokens: int = 500) -> str
extract_entities(text: str, types: list[str]) -> dict
compare_sources(claims: list[Claim]) -> ConflictReport
```

### 4.3. Action execution

```python
save_to_corpus(doc: Document, tags: list[str]) -> str
generate_report(sections: list[Section], format: str = "markdown") -> str
```

### 4.4. Tool routing (khi >10 tools)

- **Stage 1 — Rule-based router**: phân loại intent (factual / academic / comparative / internal) → shortlist 3-5 tools.
- **Stage 2 — LLM selector**: chọn 1-2 tool cụ thể.
- Tool descriptions ở **tool registry** riêng, chỉ load mô tả tool được shortlist vào prompt (tránh context overload).

---

## 5. RAG Pipeline 2 giai đoạn

### 5.1. Ingestion

```
Raw Doc ──▶ Clean (trafilatura) ──▶ Chunk (~500 tokens, overlap 50)
            │
            ├─▶ Prepend contextual summary (1-2 câu tóm tắt doc)
            │
            ▼
     Embedding (bge-m3) ──▶ Qdrant
            │
            └─▶ Tokenize ──▶ BM25 index
```

**Chunk rules**: ~500 tokens, overlap 10%, metadata bắt buộc: `source_url`, `published_date`, `author`, `doc_type`, `section`.

### 5.2. Retrieval flow

**Stage 1 — Candidate retrieval (maximize recall, top 50):**
- Hybrid = `0.5 * BM25_score + 0.5 * cosine_embedding` (tune sau).
- Optional **HyDE** cho query khó.
- Metadata filter (VD: `year >= 2024`).

**Stage 2 — Re-ranking (maximize precision, top 5-10):**
- Cross-encoder `bge-reranker-v2-m3` chấm điểm (query, chunk).
- Truyền top-k vào Synthesizer context.

### 5.3. Retrieval eval

Eval set **100 (query, relevant_doc_ids)**:
- Stage 1: Recall@10, Recall@20.
- Stage 2: NDCG@10, MRR, Precision@5.
- CI gate: fail build nếu NDCG giảm > 5% so baseline.

---

## 6. Workflow chi tiết

### 6.1. State schema (LangGraph)

```python
class ResearchState(TypedDict):
    query: str
    plan: list[SubQuestion]
    evidence: dict[str, list[Chunk]]   # sub_q_id → chunks
    drafts: dict[str, Draft]           # sub_q_id → draft + citations
    critique: dict[str, Critique]
    iterations: int
    final_report: str | None
    trace: list[StepLog]
```

### 6.2. Graph flow

```
START
  ▼
[Planner] ── phân rã 3-7 sub-questions, xếp theo dependency
  ▼
[Loop over sub-questions]
  │
  ├─▶ [Tool Router] → chọn tool phù hợp
  ├─▶ [Retriever]   → RAG 2-stage, top-5 chunks
  ├─▶ [Synthesizer] → draft + gắn [^refN]
  ├─▶ [Critic]      → score 1-5, flag issues
  │     ├─ score ≥ 4: ✓ accept
  │     └─ score < 4 & iter < 2: loop lại với feedback
  │
[Reporter] → gom drafts → final report + references + TOC
  ▼
END
```

### 6.3. Prompt skeletons

**Planner**:
```
Bạn là research planner. Phân rã câu hỏi thành 3-7 sub-questions:
- Độc lập kiểm chứng được
- Có thể answer bằng search/retrieval
- Xếp theo dependency (general → specific)
Output JSON: [{id, question, rationale, suggested_tools}]
```

**Synthesizer** (bắt buộc citation):
```
Trả lời sub-question CHỈ dựa trên <evidence>.
Mỗi câu claim kèm [^N] trỏ chunk.
Evidence không đủ → nói "Chưa đủ dữ liệu để kết luận".
KHÔNG bịa thông tin ngoài evidence.
```

**Critic**:
```
Chấm draft theo 4 tiêu chí (1-5):
1. Faithfulness: claim khớp evidence?
2. Completeness: đã trả lời đủ?
3. Citation coverage: ≥90% câu claim có ref?
4. Consistency: có mâu thuẫn nội bộ?
Output JSON: {scores, issues[], suggested_fixes[]}
```

Chi tiết prompt (versioned) ở `src/prompts/`.

---

## 7. Safety & Guardrails

Defense-in-depth 3 lớp:

### 7.1. Training-time
- Dùng base model đã qua harmlessness SFT/RL (Claude, GPT-4o).

### 7.2. Inference-time

| Lớp | Kiểm soát |
|---|---|
| Input | Prompt injection filter, PII redaction |
| Tool | Rate limit, domain allow/blocklist, timeout, budget cap per query ($) |
| Output | Hallucination detect (so evidence), factuality classifier, PII scrub |
| Report | Watermark "AI-generated", disclaimer giới hạn |

### 7.3. Anti data-exfiltration
- Agent KHÔNG được gửi data ra external endpoint ngoài whitelist.
- Log mọi tool call với args (redact secrets).

---

## 8. Observability & Debug

- **Langfuse** log mỗi step: input, output, tool call, tokens, latency, cost.
- Dashboard theo dõi divergence Plan ↔ Observe.
- Replay tool: chạy lại trace cũ với prompt mới (A/B test).
- Tag failure modes: `hallucination`, `bad_retrieval`, `tool_error`, `infinite_loop`.

---

## 9. Evaluation Framework

### 9.1. Offline (tự động, CI)

| Bộ test | Size | Metric |
|---|---|---|
| Retrieval eval set | 100 (query, relevant_docs) | Recall@k, NDCG, MRR |
| Factuality set | 50 reports có ground-truth | Faithfulness (LLM-as-judge + spot check) |
| Regression suite | 20 queries "kinh điển" | Pass/fail theo rubric |

### 9.2. Online (human-in-the-loop)

- Human rating 1-5 theo 4 trục: accuracy, completeness, citation, readability.
- Reviewer cần **foundation knowledge** về chủ đề (Stanford: *"judging correctness is hard"*).
- Thumbs up/down per section trong UI.

### 9.3. Benchmarks tham khảo
- HotpotQA, 2WikiMultiHopQA (multi-hop QA).
- ASQA (long-form answer + citation).

---

## 10. Roadmap 6 tuần

> Bám nguyên tắc **Start small, start smart**: tuần 1 single-tool + model mạnh; mở rộng dần.

> **Snapshot 2026-04-29** (chi tiết execution: `PROGRESS.md`): **Tuần 1–4** checklist roadmap §10 đã tick. Tuần 4: `academic_search` / `fetch_pdf`, `web_search` **trust tiers**, rule-based **tool router** + Planner **`suggested_tools`** (ADR-027), **`compare_sources`** (ADR-028), Critic **bốn trục** + Reporter **Conflicts noted** (ADR-030/031), **factuality eval 20** + `run_factuality_eval.py` (ADR-029), `week1_smoke.py` **`--with-router` / `--with-compare-sources`** + JSON **`router_plan_per_subq`** / **`n_conflicts`**, `run_research` **router/compare overrides**. Tuần 3 trước đó: corpus **~20 doc / 1006 chunks** (bge-m3), eval **100** + rerank A/B, citation / language / HyDE, smoke A/B (ADR-019–026). **Tiếp: Tuần 5** (safety, budget, observability) theo mục dưới.

### Tuần 1 — Skeleton & tool cơ bản
- [x] Setup repo (`uv init`, pre-commit, ruff, mypy, pytest).
- [x] LangGraph skeleton (nodes: planner → retriever → synthesizer → reporter).
- [x] 1 tool duy nhất: `web_search` (Tavily).
- [x] Streamlit UI tối giản: input query → output MD.
- [x] Agent chạy end-to-end (multi sub-question); output MD có citation.
- **Exit criteria**: chạy được 5 query mẫu, có trace log — **đạt** (xem `PROGRESS.md`).

### Tuần 2 — RAG pipeline
- [x] Ingest seed corpus (arXiv + blog; mở rộng dần — hiện ~20 doc trong YAML, không bắt buộc 50–100 ở v1 dev).
- [x] Chunking (500 tokens, overlap 50, contextual summary).
- [x] Embedding pipeline (bge-m3 mặc định; Chroma PersistentClient thay Qdrant ở dev).
- [x] BM25 index.
- [x] Hybrid retrieval; cross-encoder re-rank tích hợp (chi tiết Tuần 2–3, ADR-015).
- [x] Eval set 30 câu (ADR-016); mở rộng 100 câu ở Tuần 3 (ADR-021).
- **Exit criteria**: Recall@20 ≥ 0.7 — **đạt** trên dev seed (số liệu trong `PROGRESS.md` / `run_retrieval_eval.py`).

### Tuần 3 — Re-ranker, eval 100, tinh chỉnh stack
- [x] Cross-encoder re-rank (`bge-reranker-v2-m3`, eval A/B vs stage-1 — ADR-015/022).
- [x] Planner agent phân rã sub-questions (từ Tuần 1; cải thiện theo session).
- [x] Citation tracking chuẩn `[^N]` + Critic draft (ADR-017) + batch coverage (ADR-023).
- [x] Mở rộng eval set lên **100 câu** (70 EN + 30 VI, multi-gold — ADR-021); VI/EN language quality + HyDE optional (ADR-025/026).
- [x] Tinh chỉnh: `max_iterations` sau planner (ADR-019), Anthropic prompt cache (ADR-020), smoke CLI flags / A/B (ADR-024).
- **Exit criteria (mục tiêu Tuần 3)**: NDCG@10, citation coverage, cost/latency — đo bằng `run_retrieval_eval.py`, `run_citation_eval.py`, `week1_smoke.py`; bảng số thực tế trong `PROGRESS.md`.

### Tuần 4 — Critic loop + Multi-source
- [x] Tools: `academic_search`, `fetch_pdf`; `web_search` trust tiers — Tuần 4 Part A / input router.
- [x] Planner **`suggested_tools`** (advisory) + rule-based tool router — ADR-027.
- [x] `compare_sources` trước Critic — ADR-028.
- [x] Critic **bốn trục** + retry (ADR-017/030); Reporter References một dòng + `## Conflicts noted` — ADR-031.
- [x] Factuality eval: `factuality_eval_20.json` + `run_factuality_eval.py` — ADR-029.
- [x] Smoke metrics: `week1_smoke.py` `--with-router` / `--with-compare-sources`; JSON `router_plan_per_subq`, `n_conflicts`; `run_research` overrides.
- **Exit criteria (đo khi chạy API)**: **mean_supported_ratio** ≥ **0.80** trên 20 query (`factuality.json`); router gold-plan / conflict flags / cost smoke — chi tiết `PROGRESS.md` Tuần 4. **Toolchain code**: `pytest`, `ruff`, `mypy strict` ✓.

### Tuần 5 — Safety, Observability, Cost
- [ ] Guardrails input/output (PII, injection, domain whitelist).
- [ ] Budget cap per query.
- [ ] Langfuse dashboard đầy đủ.
- [ ] Prompt caching, batching.
- [ ] Thử downsize Synthesizer sang model rẻ hơn.
- **Exit criteria**: cost < $0.80/report; p95 latency < 4 phút.

### Tuần 6 — Polish & Beta
- [ ] Streamlit UI hoàn chỉnh + export PDF.
- [ ] 10 beta users, thu feedback.
- [ ] Regression suite 20 query cố định chạy trong CI.
- [ ] README, docs.
- **Exit criteria**: đạt TẤT CẢ target ở §1.3.

### v2 (sau tuần 6)
- [ ] MCP server chuẩn hoá (swap LLM dễ dàng).
- [ ] Tool selector LLM-based (khi có >15 tools).
- [ ] Personal corpus (upload tài liệu cá nhân).
- [ ] Semantic cache (reuse giữa các query tương tự).
- [ ] Next.js frontend.

---

## 11. Rủi ro & Mitigation

| Rủi ro | Xác suất | Impact | Mitigation |
|---|---|---|---|
| Hallucination | Cao | Cao | Citation bắt buộc + Critic + faithfulness eval |
| Search API rate-limit/đắt | Trung | Trung | Cache + backup provider |
| Scraper bị block | Trung | Thấp | Playwright + UA rotate + robots.txt |
| Agent infinite loop | Thấp | Cao | Hard cap iterations + budget cap |
| Nguồn mâu thuẫn không flag | Trung | Cao | `compare_sources` + Critic consistency check |
| Vượt ngân sách | Trung | Trung | Prompt caching + small model + per-query cap |
| Legal/Copyright | Thấp | Cao | Snippet only + respect robots.txt + disclaimer |

---

## 12. Cấu trúc repo đề xuất

```
research-assistant/
├── PLAN.md                  # file này
├── PROGRESS.md              # trạng thái session hiện tại
├── DECISIONS.md             # decision log (vì sao chọn X)
├── AGENTS.md                # hướng dẫn cho AI assistant (Cursor/Claude)
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── src/
│   └── research_assistant/
│       ├── agents/          # planner, synthesizer, critic, reporter
│       ├── tools/           # web_search, academic_search, fetch_*
│       ├── rag/             # ingestion, chunking, retrieval, reranking
│       ├── graph/           # LangGraph definition + state
│       ├── prompts/         # Jinja2 templates, versioned
│       ├── safety/          # guardrails, validators
│       ├── eval/            # eval harness, metrics
│       └── config.py
├── tests/
├── configs/                 # model, retrieval configs (YAML)
├── notebooks/               # experiments
├── data/
│   ├── raw/                 # raw docs
│   ├── processed/           # chunked
│   └── eval/                # eval datasets
├── mcp_server/              # MCP tool server (v2)
└── ui/                      # Streamlit app
```

---

## 13. Bám nguyên tắc Stanford (checklist)

| Nguyên tắc | Áp dụng ở |
|---|---|
| ReAct loop | §2.2 + LangGraph state machine |
| RAG 2-stage (recall → precision) | §5.2 |
| Chunk ~500 tokens + contextual | §5.1 |
| Hybrid BM25 + embedding | §5.2 stage 1 |
| HyDE cho query khó | §5.2 (optional) |
| Cross-encoder re-rank | §5.2 stage 2 |
| Metrics NDCG/MRR/Precision@k | §9.1 |
| Tool với API rõ ràng, 3 loại | §4 |
| Tool router khi >10 tools | §4.4 |
| MCP chuẩn hoá | §3 + v2 roadmap |
| Multi-agent, giao tiếp chuẩn | §2.3 |
| Safety defense-in-depth | §7 |
| Debug reasoning traces | §8 |
| Start small, start smart | Roadmap tuần 1–6 |
| Human evaluation | §9.2 |

---

## 14. Hướng dẫn handoff conversation

Khi conversation cũ quá dài và cần chuyển sang conversation mới, prompt mẫu:

```
Tôi đang xây Research Assistant Agent. Hãy đọc các file sau theo thứ tự:
1. PROGRESS.md   — trạng thái hiện tại và việc tiếp theo
2. PLAN.md       — kế hoạch tổng thể và kiến trúc
3. DECISIONS.md  — lý do các quyết định kỹ thuật
4. AGENTS.md     — quy ước làm việc

Sau đó xác nhận bạn đã hiểu, rồi giúp tôi làm <việc cụ thể>.
```

Assistant phải:
1. Đọc 4 file trên trước khi đề xuất gì.
2. Cập nhật `PROGRESS.md` sau mỗi milestone.
3. Thêm entry vào `DECISIONS.md` khi đưa ra quyết định kỹ thuật quan trọng.
4. KHÔNG đi chệch kế hoạch mà không hỏi user.
