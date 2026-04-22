# DECISIONS — Architecture Decision Log

> Ghi lại **các quyết định kỹ thuật quan trọng** và **lý do**. Format ADR rút gọn.
> Thêm entry mới khi quyết định: chọn công nghệ, đổi kiến trúc, đánh đổi performance/cost, bỏ feature.

---

## ADR-001: Chọn LangGraph làm orchestrator (thay vì CrewAI, AutoGen)

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Context**: Cần framework cho multi-agent + stateful workflow.
- **Options cân nhắc**:
  - LangGraph: graph-based, state explicit, debuggable.
  - CrewAI: role-based, dễ prototype nhưng abstract state.
  - AutoGen: chat-based, khó control flow.
- **Quyết định**: LangGraph.
- **Lý do**:
  - State machine tường minh → dễ replay, debug reasoning traces (đúng nguyên tắc Stanford).
  - Tích hợp sẵn với Langfuse/LangSmith.
  - Kiểm soát loop cứng (max_iterations) dễ hơn.
- **Hệ quả**: Code verbose hơn CrewAI nhưng production-ready hơn.

---

## ADR-002: RAG 2 giai đoạn (hybrid → cross-encoder rerank)

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Context**: Cần retrieval chất lượng cao cho report có citation.
- **Quyết định**: Stage 1 hybrid BM25 + bge-m3 (top 50) → Stage 2 cross-encoder bge-reranker-v2-m3 (top 5-10).
- **Lý do**: Theo nguyên tắc Stanford — hybrid luôn thắng single method; cross-encoder tăng precision đáng kể; 2-stage cân bằng recall/precision.
- **Hệ quả**: Thêm ~200ms latency cho re-rank nhưng NDCG tăng 10-20%.

---

## ADR-003: Chunk size ~500 tokens, overlap 50, contextual prepend

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Lý do**: Theo Stanford ~500 tokens cân bằng context preservation và embedding quality. Contextual prepend (Anthropic-style) giữ nghĩa khi chunk lẻ.
- **Hệ quả**: Mỗi chunk tốn thêm ~50-100 tokens context summary → chi phí indexing tăng nhưng retrieval tốt hơn nhiều.

---

## ADR-004: Model split — mạnh cho Planner/Critic, rẻ cho Synthesizer

- **Ngày**: 2026-04-21
- **Trạng thái**: Tentative (validate ở tuần 5)
- **Quyết định**: Planner + Critic dùng Claude Sonnet 4.5 / GPT-4o; Synthesizer dùng GPT-4o-mini / Claude Haiku.
- **Lý do**: Planning và judging đòi hỏi reasoning sâu; viết lại content từ evidence có sẵn dùng model nhỏ đủ tốt. Theo nguyên tắc "start smart, optimize later" — v1 dùng model mạnh hết, đo baseline, rồi downsize.
- **Hệ quả**: Cần ablation test ở tuần 5 để confirm.

---

## ADR-005: Citation bắt buộc — hallucination guard #1

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Quyết định**: Synthesizer prompt cấm output claim không có `[^N]` ref; Critic reject nếu citation coverage < 90%.
- **Lý do**: Hallucination là rủi ro cao nhất; citation ép model bám evidence; dễ verify bằng human.
- **Hệ quả**: Một số câu thông tin nền có thể bị drop nếu không tìm được evidence; chấp nhận đánh đổi này.

---

## ADR-006: MCP là v2, không phải v1

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Quyết định**: v1 dùng LangChain tool interface native; v2 mới wrap thành MCP server.
- **Lý do**: v1 chỉ cần chạy được; MCP thêm complexity mà chưa cần portability giữa LLM. Theo "start small".
- **Hệ quả**: Code tools phải viết theo chuẩn dễ wrap MCP sau (docstring rõ, signature sạch).

---

## ADR-007: Ngôn ngữ output mặc định — Tiếng Việt

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Quyết định**: Báo cáo cuối output tiếng Việt. Prompt LLM (system + instructions) vẫn viết tiếng Anh để ổn định reasoning; chỉ phần "Answer in Vietnamese" ở cuối prompt ép format output.
- **Lý do**:
  - User làm việc bằng tiếng Việt, báo cáo phục vụ độc giả Việt.
  - Prompt tiếng Anh ổn định hơn cho Claude/GPT (nhiều eval/benchmark).
  - Source evidence có thể mixed (EN + VI) → LLM tự dịch khi synthesize.
- **Hệ quả**:
  - Cần eval set có câu hỏi tiếng Việt để đo retrieval (embed `bge-m3` đã hỗ trợ đa ngôn ngữ).
  - Citation giữ nguyên tiêu đề nguồn gốc (EN nếu nguồn EN), không dịch.
  - Cần test translation quality ở tuần 3-4 (LLM-as-judge rubric riêng).

---

## ADR-008: LLM provider — Anthropic (Claude) toàn bộ

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Quyết định**:
  - Planner + Critic: **Claude Sonnet 4.5** (`claude-sonnet-4-5`).
  - Synthesizer: **Claude Haiku** (latest: `claude-haiku-4-5` nếu có, fallback `claude-3-5-haiku-latest`).
  - Embedding vẫn dùng `bge-m3` local (không dùng Voyage/Anthropic vì chi phí).
- **Lý do**:
  - User chốt Sonnet 4.5 cho Planner/Critic → dùng cùng provider cho Synthesizer giúp: (1) một API key, một billing dashboard, (2) tận dụng **prompt caching của Anthropic** (giảm cost ~90% cho context lặp lại), (3) tránh inconsistency giữa các provider khi chain agent.
  - Haiku đủ mạnh để viết lại content từ evidence có sẵn.
- **Hệ quả**:
  - Bỏ dependency `openai` khỏi v1 (có thể thêm sau nếu cần GPT-4o làm arbiter).
  - Supersedes ADR-004 (partial): model split giữ nguyên triết lý, nhưng cụ thể hoá sang Claude family.
  - Cần verify model name Haiku 4.5 khi dev (nếu chưa release, dùng `claude-3-5-haiku-latest`).

---

## ADR-009: Langfuse Cloud (thay vì self-host)

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Quyết định**: Dùng **Langfuse Cloud free tier** (`https://cloud.langfuse.com`), không self-host Docker.
- **Lý do**:
  - Free tier đủ cho v1 (50k observations/tháng).
  - Không mất thời gian ops cho Docker/Postgres local.
  - Dễ share dashboard cho teammate/reviewer sau này.
- **Hệ quả**:
  - Cần `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST=https://cloud.langfuse.com` trong `.env`.
  - Trace data đi ra ngoài → KHÔNG log PII / user query nhạy cảm nguyên văn ở giai đoạn prod. Hiện v1 là dev nên OK.
  - Nếu vượt free tier → đổi sang self-host hoặc upgrade.

---

## ADR-010: Corpus mẫu Tuần 2 — AI/ML papers + blogs

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Quyết định**: Seed corpus 50-100 docs gồm:
  - arXiv papers (`cs.AI`, `cs.LG`, `cs.CL`) 2024–2026, ưu tiên high-citation.
  - Blog posts từ nguồn uy tín: Anthropic, OpenAI, Google Research, DeepMind, Hugging Face, LangChain.
  - Một số survey papers (RAG, agents, reasoning).
- **Lý do**:
  - Chủ đề quen thuộc với user (dễ human eval).
  - Nhiều ground truth công khai → xây eval set nhanh.
  - Trùng domain với cuốn dự án (meta: agent xây agent).
- **Hệ quả**:
  - Script `scripts/ingest_seed_corpus.py` download + index.
  - Eval set 30 câu tuần 2, mở rộng 100 câu tuần 3, theo chủ đề AI/ML.
  - Khi production hóa với domain khác (finance/health), sẽ cần re-ingest và có thể tune chunk size/retrieval weights lại.

---

## ADR-011: Test budget $10 — hard cap & alerting

- **Ngày**: 2026-04-21
- **Trạng thái**: Accepted
- **Quyết định**:
  - Ngân sách tổng dev/test API: **$10 USD**.
  - **Per-query cap**: $0.30 (guardrail chặn nếu 1 query vượt).
  - **Alert threshold**: $7 → log warning + Slack/console notify.
  - **Hard stop**: $10 → agent refuse chạy tới khi user nâng cap.
- **Lý do**:
  - Budget nhỏ → phải tiết kiệm: prompt caching (Anthropic), batch request, mock LLM trong unit test, giới hạn max_iterations của agent.
  - Per-query cap $0.30 > target $0.50 ở PLAN.md — thực ra target $0.50 là cho **production v1**, trong dev ta ép còn 0.30 để có buffer.
- **Hệ quả**:
  - Implement `src/research_assistant/safety/budget.py`: tracker đọc từ response usage, Redis/file state.
  - Unit test PHẢI mock LLM (`anthropic.Anthropic` stub), không gọi thật.
  - Tuần 5 có thể phải downsize/tune trước khi kịp đo full 20 query regression.
  - Ước tính rough: Sonnet 4.5 input ~$3/1M tokens, output ~$15/1M → 1 query 10k input + 2k output ≈ $0.06 → 100 query test ≈ $6. Còn ~$4 cho Haiku + embedding (local = free).

---

## ADR-012: Langfuse instrumentation — shim `observability.py` + `run_research()` là root span

- **Ngày**: 2026-04-22
- **Trạng thái**: Accepted
- **Context**: ADR-009 chốt Langfuse Cloud làm observability layer, nhưng Part B tuần 1 mới chỉ `langfuse_enabled` flag; chưa có span cụ thể. Khi implement full `@observe` cần trả lời:
  1. Làm sao để unit test / CI / run không có key KHÔNG emit auth warnings hay gọi API thật?
  2. Làm sao `trace_id` consistent giữa các node LangGraph (vì LangGraph thread execution có thể phá OpenTelemetry context)?
  3. Streamlit stream events — không decorate được hàm generator → cần context manager thay thế.
  4. `pydantic-settings` load `.env` vào `Settings` nhưng KHÔNG đổ vào `os.environ` → Langfuse SDK `get_client()` đọc env vars → auth fail dù `Settings.langfuse_enabled` = True.
- **Options cân nhắc**:
  - (A) Dùng thẳng `langfuse.observe` khắp nơi + `dotenv.load_dotenv()` early. Nhanh nhưng phụ thuộc implicit env state, và decorator import sớm sẽ trigger auth warning trong test.
  - (B) Viết module shim `observability.py` bọc toàn bộ API của Langfuse, lazy-import SDK, explicit instantiate `Langfuse(public_key=..., secret_key=..., host=...)` từ `Settings`. Thêm `run_research()` helper làm root span và inject `trace_id/url` vào `ResearchState`, `start_agent_span()` context manager cho Streamlit.
  - (C) Bỏ qua tracing chi tiết, chỉ log StepLog.
- **Quyết định**: **B**.
- **Lý do**:
  - Shim tách rõ disabled-path (transparent passthrough) khỏi enabled-path → `pytest` không còn 401 errors, không cần set env vars giả. `conftest.py` autouse clear Langfuse keys củng cố điều này.
  - Explicit `Langfuse(...)` loại được race condition với `os.environ`; không cần `dotenv.load_dotenv()` ở `__main__` (repo-wide quy ước là pydantic-settings chịu trách nhiệm duy nhất về cấu hình).
  - `run_research()` bọc toàn bộ graph thành **1 root agent span** → `trace_id` / `trace_url` bắt ngay tại entry, inject vào state, mọi node đọc cùng giá trị → link trace duy nhất xuất hiện trong report footer.
  - `start_agent_span()` cho phép `ui/app.py` stream từng update mà vẫn gói trong 1 trace (đảm bảo observability parity CLI ↔ UI).
- **Hệ quả**:
  - Rest-of-code chỉ import từ `research_assistant.observability`, Langfuse version bump chỉ đụng shim.
  - `@observe` của shim có cache per-function để chỉ khởi tạo Langfuse decorator một lần.
  - `invoke_llm` / `invoke_structured_llm` capture `model` + `usage_details` + `cost_details` → dashboard đo được cost theo model/node.
  - `reporter_v1.jinja` thêm footer conditional `{% if has_trace %}` link sang `cloud.langfuse.com/.../traces/<id>` — không render khi tracing tắt, đảm bảo smoke test không key vẫn clean.
  - Verify run 2026-04-22: query "What is LangGraph state persistence?" → cost $0.0224, `trace_id=995f6a8874ce7bcf7735711c23e0966a`, footer render đúng, `auth_check()` = True.
  - Bắt buộc gọi `flush()` cuối `run_research` (và cuối UI handler) vì Langfuse Python SDK buffer span; process CLI exit nhanh có thể drop batch cuối nếu không flush.

---

## ADR-013: RAG ingestion stack — Chroma dev, bge-small-en-v1.5 dev, trafilatura-only, abstract-as-prepend

- **Ngày**: 2026-04-22
- **Trạng thái**: Accepted
- **Context**: Bắt đầu RAG pipeline (PLAN §5.1). Trước khi code cần chốt 4 chi tiết không hiển nhiên từ PLAN:
  1. **Vector store dev**: PLAN §3 đã nói "Qdrant prod, Chroma dev" nhưng chưa chốt client mode (Chroma Cloud / HTTP server / PersistentClient local / in-memory). Dev phải chạy được offline, không Docker.
  2. **Embedding model**: PLAN ghi `BAAI/bge-m3` (~2.3 GB, multilingual). Dev corpus Tuần 2 toàn paper/blog tiếng Anh (ADR-010); tải 2.3 GB + embed 5 phút/run vs. bge-small-en-v1.5 (130 MB, embed nhanh 3× trên CPU) → chênh lệch thời gian lặp rất lớn khi iterate chunking/retrieval logic.
  3. **Web/HTML scraping**: PLAN §3 cho `trafilatura` + `playwright` (JS fallback). Playwright cần cài chromium (~200 MB) + chạy browser headless → setup nặng. Corpus Tuần 2 toàn blog static (Anthropic/OpenAI/LangChain/HF), trafilatura đủ dùng.
  4. **Contextual prepend** (ADR-003 nói "1-2 câu tóm tắt doc"): dùng LLM summary (đắt, ~$0.001/doc) hay dùng abstract có sẵn (free, deterministic)? arXiv có abstract; blog có `meta[name="description"]` nhờ trafilatura.
- **Options cân nhắc**:
  - **Vector store**: (A) Chroma PersistentClient local (file-backed sqlite + HNSW bin). (B) Chroma HTTP server (cần chạy daemon). (C) In-memory (mất state mỗi restart → không fit dev loop).
  - **Embedding**: (A) bge-m3 luôn. (B) bge-small-en-v1.5 dev + bge-m3 prod, swap bằng settings. (C) Voyage / OpenAI embedding API (tốn budget ADR-011).
  - **HTML**: (A) trafilatura only. (B) trafilatura + playwright fallback ngay v1.
  - **Prepend**: (A) abstract + first 2 sentences fallback + title fallback. (B) LLM-generated summary 1 lần/doc, cache.
- **Quyết định**:
  - Vector store: **Chroma PersistentClient** tại `data/chroma/` (gitignored), collection `ai_ml_corpus_v1`. `hnsw:space="cosine"` khớp normalised embeddings.
  - Embedding: **bge-small-en-v1.5** (384-dim) làm dev default qua `Settings.embedding_model`; tài liệu trong docstring + PROGRESS cảnh báo swap sang bge-m3 **TRƯỚC** khi thêm VI content hoặc chạy eval factuality sản phẩm. Swap = đổi `.env` + chạy `ingest_seed_corpus.py --rebuild` (dimension khác nhau, không tương thích cùng collection).
  - HTML: **trafilatura only** ở v1 (fetch + extract + metadata). Playwright move sang backlog, trigger khi gặp blog/site JS-heavy trả empty extract.
  - Prepend: **deterministic template** `"[Title] <abstract|first 2 sentences|title>"`, cap `max_chars=400`. Không gọi LLM.
- **Lý do**:
  - PersistentClient = zero ops, vẫn survive restart, sqlite + HNSW file đủ cho 10k–100k chunks; Qdrant swap dễ vì `ChromaStore` giấu API behind `upsert_chunks` / `search(SearchResult)` contract.
  - bge-small EN 384-dim: 3.4 ch/s trên CPU (Windows laptop), 766 chunks = 3.7 phút; bge-m3 1024-dim ước ~1 ch/s + 2.3 GB download lần đầu = blocker iteration. Eval thực chất sẽ chạy trên corpus EN v1 (arXiv + blog EN), không cần multilingual. Khi thêm query/corpus tiếng Việt cho user-facing reports, swap bắt buộc.
  - trafilatura extract OK 5/5 blog trong seed (Anthropic × 2, OpenAI, HF, LangChain) — không có failure; playwright thêm 200 MB và CI phức tạp không đáng lúc này.
  - Abstract có sẵn từ arXiv / `meta description` blog + first-2-sentences fallback = đủ ngữ nghĩa cho chunk context (kiểm tra thực tế: top-1 hit cho query "contextual retrieval" là Anthropic blog dist 0.132, cho thấy prepend đúng nghĩa). LLM summary lại phát sinh variance giữa các lần ingest (model nondeterminism) → khó reproducibility.
- **Hệ quả**:
  - `Settings.embedding_model` + `Settings.embedding_device` là 2 knob chính để swap môi trường. Changelog khi đổi model phải kèm `--rebuild` ingest (auto qua `ChromaStore.reset()`).
  - `ChromaStore` API giữ gọn 4 method: `upsert_chunks`, `search`, `reset`, `count`. Qdrant port sau này chỉ cần implement cùng signature.
  - Các chunk từ PDF arXiv có artifact font (unicode garbage trong `ReAct` Figure 1 hiện thấy) → observation trong PROGRESS; nếu ảnh hưởng retrieval thật sự thì thêm filter theo non-printable-char ratio trước embed.
  - `configs/seed_corpus.yaml` là entry point mở rộng corpus — thêm doc chỉ cần sửa YAML + `--rebuild`, không đụng code.
  - `scripts/ingest_seed_corpus.py` ghi `data/eval/ingest_manifest.json` (commit được — summary minh bạch, không leak data); `data/chroma/*` + `data/raw/*` gitignored.
  - Per-query test budget không đụng đến (ADR-011) vì RAG ingestion zero-LLM-call; embedding là local inference free.

---

## ADR-014: Stage-1 hybrid — `rank_bm25` in-process + 50/50 min-max per leg

- **Ngày**: 2026-04-22
- **Trạng thái**: Accepted
- **Context**: PLAN.md §5.2 yêu cầu stage-1 candidate retrieval: BM25 + cosine embedding, trọng số 0.5/0.5 baseline, top ~50 trước re-rank. Cần cùng `SearchHit` contract như `web_search` để retriever / Synthesizer không phân nhánh theo provider.
- **Options cân nhắc**:
  - (A) Elasticsearch / OpenSearch — full-text tốt nhưng cần daemon hoặc cloud.
  - (B) `rank_bm25` in-process, index rebuild từ cùng text đã lưu trong Chroma `documents`.
  - (C) Chroma full-text nếu bật — phiên bản Chroma dev đang dùng không coi đây là ưu tiên v1.
- **Quyết định**: **(B)** — thư viện `rank_bm25` (Okapi), corpus token từ `body` chunk; index build lần đầu khi gọi `vector_search` (cache theo `collection` + `count()`); trộn union top-50 dense + top-50 BM25 với min--max chuẩn hoá **trong từng chân** rồi cộng `(w_d·d + w_b·b)/(w_d+w_b)`; trọng số mặc định 0.5/0.5. Dense leg vẫn `ChromaStore.search` (ADR-013).
- **Lý do**:
  - Zero thêm dịch vụ ngoài sqlite Chroma; đủ cho seed ~hàng trăm–vài nghìn chunk trên dev laptop.
  - Khớp ADR-002 (hybrid) và roadmap Elasticsearch v2 nếu scale.
  - `tools/vector_search.py` tách rõ: graph chỉ gọi một tool trả `SearchHit` (`source="corpus"`), cùng mẫu Langfuse `@observe` như web search.
- **Hệ quả**:
  - Lần đầu sau ingest / `--rebuild` cần build lại index BM25 (O(n) tokenize) — chấp nhận; expose `clear_vector_search_cache()` cho test.
  - Metadata filter (`filters` → Chroma `where`) chỉ áp dense leg; BM25 leg luôn trên toàn collection (tuần 2 tối thiểu; có thể post-filter trước khi tổng hợp sau nếu cần).

---

## ADR-015: Stage-2 cross-encoder — `BAAI/bge-reranker-v2-m3` trong graph retriever

- **Ngày**: 2026-04-22
- **Trạng thái**: Accepted
- **Context**: ADR-002 + PLAN §5.2 chốt hybrid (stage 1) rồi cross-encoder (stage 2) trước Synthesizer. Cần quyết định: gọi rerank ở đâu, pool bao nhiêu ứng viên, tắt khi nào (CI / máy yếu).
- **Quyết định**:
  - Model: **`BAAI/bge-reranker-v2-m3`** qua `sentence_transformers.CrossEncoder` (lazy cache, lock download), `max_length=1024`, device từ `Settings.reranker_device`.
  - Vị trí: **`graph/research_graph.py` retriever** sau `_corpus_then_web_hits` (pool tối đa `retrieval_candidate_pool`, mặc định 20), cắt xuống `synthesizer_evidence_top_k` (mặc định 5). Một hàm injectable `rerank_fn` + `build_graph(..., rerank_fn=...)` để unit test không tải weights.
  - Passage cho cặp `(query, doc)`: ưu tiên `SearchHit.raw_content` nếu đủ dài (≥48 ký tự sau strip), không thì `snippet`/`title`; cap 8000 ký tự.
  - Điểm trả về: min--max chuẩn hoá trong top-k batch vào `SearchHit.score` ∈ [0,1]. Một hit: `score=1.0`, không gọi `predict`.
  - `Settings.reranker_enabled` (default **True**): `False` để tắt rerank (chỉ `hits[:synthesizer_evidence_top_k]` theo thứ tự pool). Lỗi `predict` / load model: fallback trong `_default_rerank_fn` sang thứ tự gốc (k retriever-level `except` vẫn giữ partial nếu custom `rerank_fn` ném).
- **Hệ quả**: Lần đầu chạy CLI/UI tải thêm ~hundreds MB–1GB+ tùy môi trường; document trong `.env.example`. Web-only evidence (snippet ngắn) vẫn được rerank với passage = snippet.

---

## ADR-016: Retrieval eval 30 câu — qrels theo `source_id` + metrics macro

- **Ngày**: 2026-04-22
- **Trạng thái**: Accepted
- **Context**: ADR-010; Tuần 2 cần Recall@k / NDCG@10 gắn exit criteria.
- **Quyết định**:
  - **`data/eval/retrieval_eval_30.json`**: 30 câu (EN), mỗi câu **một** `relevant_source_ids` khớp `source_id` Chroma sau ingest (kèm hậu tố phiên bản arXiv, blog `h_<hex>`).
  - **Baseline đo lưới (stage-1)**: `hybrid_search_stage1` như `vector_search`, `final_top_k=20`; **recall@10/@20** = macro trung bình theo câu của `|gold ∩ sources(ở top-k chunk)| / |gold|`; **NDCG@10** trên vector nhị phân 10 chunk đầu, IDCG từ sắp lý tưởng cùng multiset.
  - **`research_assistant.eval`**: `metrics` + `retrieval`; **`scripts/run_retrieval_eval.py`**; JSON output tùy chọn.
- **Hệ quả**: Số trên 15 doc seed **không** tổng quát domain khác; cần qrels mới khi mở corpus.

---

<!-- Template cho entry mới:

## ADR-NNN: <Tiêu đề ngắn>

- **Ngày**: YYYY-MM-DD
- **Trạng thái**: Proposed | Accepted | Deprecated | Superseded by ADR-XXX
- **Context**: Vấn đề cần giải quyết
- **Options cân nhắc**: A vs B vs C
- **Quyết định**: Chọn X
- **Lý do**: ...
- **Hệ quả**: Trade-offs
-->
