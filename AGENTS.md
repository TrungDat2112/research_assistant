# AGENTS.md — Hướng dẫn cho AI assistant

> File này dành cho **AI assistant (Cursor, Claude Code, v.v.)** làm việc trên repo này.
> User đọc cũng được nhưng mục đích chính là định hướng AI.

---

## 1. Bối cảnh dự án

Xây **Research Assistant Agent** — AI agent tự động nghiên cứu một chủ đề và xuất báo cáo có citation. Dựa trên nguyên tắc Stanford "How to Build AI Agents" (xem `AI_building_principles.png`).

**Nguồn sự thật theo thứ tự đọc**:
1. `PROGRESS.md` — trạng thái hiện tại, việc tiếp theo.
2. `PLAN.md` — kiến trúc, tech stack, roadmap đầy đủ.
3. `DECISIONS.md` — lý do các quyết định kỹ thuật.
4. `AGENTS.md` — file này.

---

## 2. Quy tắc bắt buộc khi làm việc

### 2.1. Luôn làm trước khi code
- [ ] Đọc `PROGRESS.md` xem đang ở phase nào.
- [ ] Đối chiếu với roadmap trong `PLAN.md` §10.
- [ ] Nếu task không có trong plan → **hỏi user trước**, đừng tự quyết.

### 2.2. Bám kiến trúc
- Multi-agent theo ReAct (Observe → Plan → Act).
- RAG 2 giai đoạn (hybrid → cross-encoder).
- Tool chuẩn hoá theo 3 loại: retrieval / computation / action.
- Citation bắt buộc trong mọi output có claim.

### 2.3. Code style
- Python 3.11+, type hints đầy đủ, `mypy --strict`.
- `ruff` format + lint, không comment thừa.
- Tool function phải có docstring đầy đủ (LLM đọc docstring để gọi tool) — **name, args, return, khi nào dùng**.
- Prompt template trong `src/research_assistant/prompts/`, versioned bằng filename (VD: `planner_v1.jinja`).
- Config trong `configs/*.yaml`, load bằng `pydantic-settings`.

### 2.4. Testing
- Unit test cho tools, chunking, retrieval.
- Integration test cho graph (dùng LLM stub/mock).
- Eval test (đo metric) chạy offline, không trong CI chính (quá đắt).

### 2.5. Observability
- Mọi agent step phải log qua Langfuse/LangSmith.
- Log redact secret (API key, PII).

### 2.6. Safety first
- Không tự cho phép agent gọi endpoint ngoài whitelist.
- Rate limit + budget cap per query.
- Output filter: reject nếu citation coverage < 90%.

---

## 3. Workflow mỗi session

1. **Start**: đọc `PROGRESS.md` § "Việc tiếp theo".
2. **Confirm với user**: "Tôi sẽ làm task X, Y, Z — ok không?"
3. **Code**: theo quy tắc §2.
4. **Test**: `pytest` + chạy smoke test end-to-end.
5. **Update `PROGRESS.md`**:
   - Check task đã xong.
   - Thêm entry vào § "Log session".
   - Cập nhật § "Việc tiếp theo".
6. **Update `DECISIONS.md`** nếu có quyết định kỹ thuật mới (công nghệ, kiến trúc, trade-off).
7. **Commit**: message theo conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

---

## 4. KHÔNG được làm

- ❌ Đổi tech stack (đã chốt trong `PLAN.md` §3) mà không hỏi + không ghi `DECISIONS.md`.
- ❌ Skip citation / hallucination guard vì "cho nhanh".
- ❌ Commit secret (`.env`, API key). File `.env.example` chỉ chứa placeholder.
- ❌ Tự crawl web quy mô lớn; chỉ dùng search API + fetch theo URL user/agent chỉ định.
- ❌ Tạo feature ngoài roadmap v1 (MCP, multi-user, fine-tune, …) trừ khi user yêu cầu.
- ❌ Viết comment lặp lại code ("// increment counter" kiểu đó).
- ❌ Dùng `git commit` nếu user chưa yêu cầu commit.

---

## 5. Cấu trúc folder (bám `PLAN.md` §12)

```
src/research_assistant/
├── agents/       # planner, synthesizer, critic, reporter
├── tools/        # web_search, academic_search, fetch_*, vector_search
├── rag/          # ingestion, chunking, retrieval, reranking
├── graph/        # LangGraph definition + state schema
├── prompts/      # Jinja2 templates, versioned
├── safety/       # guardrails, validators
├── eval/         # eval harness, metrics, datasets loader
└── config.py     # pydantic-settings
```

---

## 6. Ngôn ngữ giao tiếp

- **Với user**: tiếng Việt (trừ khi user chuyển sang tiếng Anh).
- **Code, docstring, commit message, file names**: tiếng Anh.
- **Comment trong code**: tiếng Anh, ngắn gọn, chỉ giải thích "tại sao", không "cái gì".
- **Prompt LLM**: tiếng Anh (ổn định hơn cho reasoning), trừ khi user yêu cầu prompt tiếng Việt để report ra tiếng Việt.

---

## 7. Checklist trước khi kết thúc session

- [ ] Code đã format (`ruff format`) và lint (`ruff check`).
- [ ] Type check (`mypy src/`).
- [ ] Test pass (`pytest`).
- [ ] `PROGRESS.md` đã update.
- [ ] `DECISIONS.md` đã có entry mới (nếu có quyết định mới).
- [ ] Không có secret trong staged files.
- [ ] User đã confirm trước khi commit.
