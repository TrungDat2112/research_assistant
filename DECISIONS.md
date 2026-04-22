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
