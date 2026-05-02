from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "data" / "eval" / "retrieval_eval_100.json"
_EVAL30 = _REPO / "data" / "eval" / "retrieval_eval_30.json"
_MANIFEST = _REPO / "data" / "eval" / "ingest_manifest.json"

_EXTRA_EN: list[dict[str, Any]] = [
    {
        "id": "q31",
        "query": "What papers combine dense retrievers with a generator for open-domain question answering and survey the RAG field?",
        "language": "en",
        "relevant_source_ids": ["2005.11401v4", "2312.10997v5"],
    },
    {
        "id": "q32",
        "query": "Low-rank adapters for LLMs: compare parameter efficiency of LoRA and 4-bit QLoRA fine-tuning",
        "language": "en",
        "relevant_source_ids": ["2106.09685v2", "2305.14314v1"],
    },
    {
        "id": "q33",
        "query": "Learning augmentation policies and strategies for object detection on visual datasets",
        "language": "en",
        "relevant_source_ids": ["1906.11172v1"],
    },
    {
        "id": "q34",
        "query": "Scalable network representation learning with Metropolis-Hastings and random walks",
        "language": "en",
        "relevant_source_ids": ["2010.04895v2"],
    },
    {
        "id": "q35",
        "query": "Reinforcement learning to walk over knowledge base relations for question answering",
        "language": "en",
        "relevant_source_ids": ["1711.05851v2"],
    },
    {
        "id": "q36",
        "query": "Harmlessness and preference learning from AI feedback: Constitutional AI",
        "language": "en",
        "relevant_source_ids": ["2212.08073v1"],
    },
    {
        "id": "q37",
        "query": "Distilling a smaller, faster, lighter transformer from BERT: DistilBERT",
        "language": "en",
        "relevant_source_ids": ["1910.01108v4"],
    },
    {
        "id": "q38",
        "query": "Incentivizing chain-of-thought and reasoning in LLMs with reinforcement learning and self-refine loops",
        "language": "en",
        "relevant_source_ids": ["2501.12948v2", "2303.17651v2"],
    },
    {
        "id": "q39",
        "query": "From local community summaries to global answers: graph-based RAG and survey coverage of RAG",
        "language": "en",
        "relevant_source_ids": ["2404.16130v2", "2312.10997v5"],
    },
    {
        "id": "q40",
        "query": "Contextual text embeddings for each chunk in RAG and survey of retrieval-augmented LLMs",
        "language": "en",
        "relevant_source_ids": ["h_a9cde9583a70d78b", "2312.10997v5"],
    },
    {
        "id": "q41",
        "query": "Practical RAG with Hugging Face transformers and distributed Ray data pipelines",
        "language": "en",
        "relevant_source_ids": ["h_335ba6bcb0ab0a0f"],
    },
    {
        "id": "q42",
        "query": "State machines, checkpoints, and cyclic workflows for long-horizon LLM applications with LangGraph",
        "language": "en",
        "relevant_source_ids": ["h_3b5b0e78c657a8b3"],
    },
    {
        "id": "q43",
        "query": "Planning, tool use, and multi-step agents in production: Anthropic and ReAct",
        "language": "en",
        "relevant_source_ids": ["h_7d24e5faa28b319b", "2210.03629v3"],
    },
    {
        "id": "q44",
        "query": "JSON schema and strict function calling outputs in the OpenAI API and structured tool use",
        "language": "en",
        "relevant_source_ids": ["h_9606a2cdf1b1c87e"],
    },
    {
        "id": "q45",
        "query": "Self-RAG and Self-Refine: self-reflection and iterative refinement in language models",
        "language": "en",
        "relevant_source_ids": ["2310.11511v1", "2303.17651v2"],
    },
    {
        "id": "q46",
        "query": "Llama 2 pre-training, safety tuning, and open foundation release details",
        "language": "en",
        "relevant_source_ids": ["2307.09288v2"],
    },
    {
        "id": "q47",
        "query": "ReAct: trajectories of reasoning and acting with external tools in one loop",
        "language": "en",
        "relevant_source_ids": ["2210.03629v3"],
    },
    {
        "id": "q48",
        "query": "QLoRA: NF4 quantisation and paged optimisers to fine-tune large models on one GPU",
        "language": "en",
        "relevant_source_ids": ["2305.14314v1"],
    },
    {
        "id": "q49",
        "query": "RAG for knowledge-intensive NLP tasks: retrieve then generate with dense passages",
        "language": "en",
        "relevant_source_ids": ["2005.11401v4"],
    },
    {
        "id": "q50",
        "query": "Survey: taxonomy, benchmarks, and future directions of retrieval-augmented generation for LLMs",
        "language": "en",
        "relevant_source_ids": ["2312.10997v5"],
    },
    {
        "id": "q51",
        "query": "What is self-reflection, critique, and adaptive retrieval in Self-RAG?",
        "language": "en",
        "relevant_source_ids": ["2310.11511v1"],
    },
    {
        "id": "q52",
        "query": "How does the Graph RAG pipeline build entity graphs and community summaries before answering?",
        "language": "en",
        "relevant_source_ids": ["2404.16130v2"],
    },
    {
        "id": "q53",
        "query": "Reinforcement learning to improve LLM reasoning: DeepSeek-R1 style incentives",
        "language": "en",
        "relevant_source_ids": ["2501.12948v2"],
    },
    {
        "id": "q54",
        "query": "Constitutional classifiers, revision, and self-critique for harmlessness in assistants",
        "language": "en",
        "relevant_source_ids": ["2212.08073v1"],
    },
    {
        "id": "q55",
        "query": "LoRA: low-rank update matrices and trainable parameters relative to full fine-tuning",
        "language": "en",
        "relevant_source_ids": ["2106.09685v2"],
    },
    {
        "id": "q56",
        "query": "Knowledge base path reasoning with reinforcement learning: Go for a Walk style agents",
        "language": "en",
        "relevant_source_ids": ["1711.05851v2"],
    },
    {
        "id": "q57",
        "query": "Object detection: learning data augmentation policies from labeled detection datasets",
        "language": "en",
        "relevant_source_ids": ["1906.11172v1"],
    },
    {
        "id": "q58",
        "query": "Network representation learning: UniNet and graph embedding scalability",
        "language": "en",
        "relevant_source_ids": ["2010.04895v2"],
    },
    {
        "id": "q59",
        "query": "How does contextual chunk enrichment reduce retrieval failures in RAG (Anthropic article)?",
        "language": "en",
        "relevant_source_ids": ["h_a9cde9583a70d78b"],
    },
    {
        "id": "q60",
        "query": "Design patterns for long-running software agents, routing, and failure recovery",
        "language": "en",
        "relevant_source_ids": ["h_7d24e5faa28b319b"],
    },
    {
        "id": "q61",
        "query": "Retrieval augmented generation at scale: Ray actors and multi-node embedding",
        "language": "en",
        "relevant_source_ids": ["h_335ba6bcb0ab0a0f"],
    },
    {
        "id": "q62",
        "query": "How does LangGraph support persistence, interrupts, and human-in-the-loop steps?",
        "language": "en",
        "relevant_source_ids": ["h_3b5b0e78c657a8b3"],
    },
    {
        "id": "q63",
        "query": "OpenAI: enforcing JSON that matches a strict schema in production APIs",
        "language": "en",
        "relevant_source_ids": ["h_9606a2cdf1b1c87e"],
    },
    {
        "id": "q64",
        "query": "Self-Refine: improve drafts through multiple refinement rounds without new labels",
        "language": "en",
        "relevant_source_ids": ["2303.17651v2"],
    },
    {
        "id": "q65",
        "query": "Differences between BERT, DistilBERT, and distillation for latency-sensitive deployment",
        "language": "en",
        "relevant_source_ids": ["1910.01108v4"],
    },
    {
        "id": "q66",
        "query": "RAG and Self-RAG: when to retrieve, generate, and score intermediate outputs",
        "language": "en",
        "relevant_source_ids": ["2005.11401v4", "2310.11511v1"],
    },
    {
        "id": "q67",
        "query": "Query-focused summarization over long documents: graph RAG and survey-style coverage",
        "language": "en",
        "relevant_source_ids": ["2404.16130v2", "2005.11401v4"],
    },
    {
        "id": "q68",
        "query": "Sub-goal decomposition and tool use in ReAct compared to long-horizon agent runbooks (Anthropic)",
        "language": "en",
        "relevant_source_ids": ["2210.03629v3", "h_7d24e5faa28b319b"],
    },
    {
        "id": "q69",
        "query": "RAG with Ray, structured outputs for tools, and LangGraph state machines for orchestration",
        "language": "en",
        "relevant_source_ids": ["h_335ba6bcb0ab0a0f", "h_9606a2cdf1b1c87e", "h_3b5b0e78c657a8b3"],
    },
    {
        "id": "q70",
        "query": "RLHF, constitutional principles, and preference modelling for safe assistants",
        "language": "en",
        "relevant_source_ids": ["2212.08073v1", "2307.09288v2"],
    },
]

# 30 Vietnamese items (q71-q100) - bge-m3 multilingual; qrels point at English+ blog chunks in Chroma
_EXTRA_VI: list[dict[str, Any]] = [
    {
        "id": "q71",
        "query": "RAG truy hồi tăng cường (retrieval-augmented generation) dùng cho tác vụ NLP cần nhiều tri thức là gì?",
        "language": "vi",
        "relevant_source_ids": ["2005.11401v4"],
    },
    {
        "id": "q72",
        "query": "Self-RAG: mô hình tự lấy tài liệu, tự sinh câu trả lời và tự phê bình thế nào?",
        "language": "vi",
        "relevant_source_ids": ["2310.11511v1"],
    },
    {
        "id": "q73",
        "query": "Tổng quan và phân loại phương pháp RAG cho mô hình ngôn ngữ lớn (LLM)",
        "language": "vi",
        "relevant_source_ids": ["2312.10997v5"],
    },
    {
        "id": "q74",
        "query": "LoRA: thích ứng hạng thấp (low-rank adaptation) để fine-tune LLM tiết kiệm tham số",
        "language": "vi",
        "relevant_source_ids": ["2106.09685v2"],
    },
    {
        "id": "q75",
        "query": "QLoRA: fine-tune LLM 4-bit lượng tử hóa trên GPU nhỏ với paged optimiser",
        "language": "vi",
        "relevant_source_ids": ["2305.14314v1"],
    },
    {
        "id": "q76",
        "query": "RAG tập trung theo câu hỏi: từ tóm tắt cộng đồng cục bộ tới tóm tắt toàn cục (GraphRAG)",
        "language": "vi",
        "relevant_source_ids": ["2404.16130v2"],
    },
    {
        "id": "q77",
        "query": "ReAct: vòng lặp suy luận và hành động dùng công cụ bên ngoài trong mô hình ngôn ngữ",
        "language": "vi",
        "relevant_source_ids": ["2210.03629v3"],
    },
    {
        "id": "q78",
        "query": "Self-Refine: cải thiện bản thảo bằng nhiều vòng tự phản hồi mà không cần dữ liệu mới",
        "language": "vi",
        "relevant_source_ids": ["2303.17651v2"],
    },
    {
        "id": "q79",
        "query": "Llama 2: quy mô, huấn luyện trước, và căn chỉnh an toàn theo phản hồi con người",
        "language": "vi",
        "relevant_source_ids": ["2307.09288v2"],
    },
    {
        "id": "q80",
        "query": "DeepSeek-R1: khuyến khích suy luận chuỗi trong LLM bằng học tăng cường",
        "language": "vi",
        "relevant_source_ids": ["2501.12948v2"],
    },
    {
        "id": "q81",
        "query": "Bài toán tăng cường dữ liệu cho phát hiện đối tượng: học chính sách augmentation",
        "language": "vi",
        "relevant_source_ids": ["1906.11172v1"],
    },
    {
        "id": "q82",
        "query": "Học biểu diễn mạng quy mô lớn với lấy mẫu Metropolis-Hastings",
        "language": "vi",
        "relevant_source_ids": ["2010.04895v2"],
    },
    {
        "id": "q83",
        "query": "Đi bộ trên đồ thị tri thức: học tăng cường trả lời câu hỏi đa bước trên cơ sở tri thức",
        "language": "vi",
        "relevant_source_ids": ["1711.05851v2"],
    },
    {
        "id": "q84",
        "query": "Constitutional AI: giảm tác hại từ phản hồi của mô hình AI và lớp bảo vệ ưu tiên",
        "language": "vi",
        "relevant_source_ids": ["2212.08073v1"],
    },
    {
        "id": "q85",
        "query": "DistilBERT: cách cô đặc BERT thành mô hình nhỏ hơn, nhanh hơn cho triển khai",
        "language": "vi",
        "relevant_source_ids": ["1910.01108v4"],
    },
    {
        "id": "q86",
        "query": "Tìm thấy tài liệu theo bối cảnh (contextual retrieval) của Anthropic: giảm bỏ sót khi RAG",
        "language": "vi",
        "relevant_source_ids": ["h_a9cde9583a70d78b"],
    },
    {
        "id": "q87",
        "query": "Cách xây agent hiệu quả: lập kế hoạch, công cụ, và tác tử đa bước theo bài viết kỹ thuật Anthropic",
        "language": "vi",
        "relevant_source_ids": ["h_7d24e5faa28b319b"],
    },
    {
        "id": "q88",
        "query": "OpenAI: định dạng JSON đúng lược đồ ràng buộc chặt chẽ khi dùng API",
        "language": "vi",
        "relevant_source_ids": ["h_9606a2cdf1b1c87e"],
    },
    {
        "id": "q89",
        "query": "RAG tại quy mô lớn với Ray và bộ dữ liệu Hugging Face Transformers",
        "language": "vi",
        "relevant_source_ids": ["h_335ba6bcb0ab0a0f"],
    },
    {
        "id": "q90",
        "query": "LangGraph là gì: trạng thái, vòng lặp, kiểm tra điểm cho ứng dụng nhiều bước với LLM",
        "language": "vi",
        "relevant_source_ids": ["h_3b5b0e78c657a8b3"],
    },
    {
        "id": "q91",
        "query": "RAG cơ bản và tổng quan mô tả: kết hợp tìm kiếm dày đặc với tạo văn bản theo tài liệu",
        "language": "vi",
        "relevant_source_ids": ["2005.11401v4", "2312.10997v5"],
    },
    {
        "id": "q92",
        "query": "So sánh nhanh LoRA và QLoRA khi fine-tune LLM: bộ nhớ, độ ổn định, và khi nào dùng cái nào",
        "language": "vi",
        "relevant_source_ids": ["2106.09685v2", "2305.14314v1"],
    },
    {
        "id": "q93",
        "query": "Tự phản ánh: Self-RAG kết hợp tự tạo, tự truy xuất, và tự cải thiện như thế nào?",
        "language": "vi",
        "relevant_source_ids": ["2310.11511v1", "2005.11401v4"],
    },
    {
        "id": "q94",
        "query": "RAG: khảo sát phương pháp và ứng dụng cho LLM, kèm tài liệu mật độ truy hồi",
        "language": "vi",
        "relevant_source_ids": ["2312.10997v5"],
    },
    {
        "id": "q95",
        "query": "Giao diện suy luận-hành động (ReAct) so với thiết kế tác tử theo tài liệu Anthropic",
        "language": "vi",
        "relevant_source_ids": ["2210.03629v3", "h_7d24e5faa28b319b"],
    },
    {
        "id": "q96",
        "query": "An toàn và căn chỉnh: so sánh cách tiếp cận Constitutional AI với RLHF trên LLaMA 2",
        "language": "vi",
        "relevant_source_ids": ["2212.08073v1", "2307.09288v2"],
    },
    {
        "id": "q97",
        "query": "Tóm tắt theo truy vấn từ đồ thị: GraphRAG trên tập tài liệu riêng tư, khớp tài liệu khảo sát tổng RAG",
        "language": "vi",
        "relevant_source_ids": ["2404.16130v2", "2312.10997v5"],
    },
    {
        "id": "q98",
        "query": "Cải thiện từng bản thảo: Self-Refine và cơ chế tự tạo, tự phê bình trong thế hệ gần đây",
        "language": "vi",
        "relevant_source_ids": ["2303.17651v2", "2310.11511v1"],
    },
    {
        "id": "q99",
        "query": "Hệ thống nhiều tác tử, Ray RAG, và bước kiểm soát cấu trúc: pipeline triển khai thực tế",
        "language": "vi",
        "relevant_source_ids": ["h_335ba6bcb0ab0a0f", "h_9606a2cdf1b1c87e", "h_3b5b0e78c657a8b3"],
    },
    {
        "id": "q100",
        "query": "Học sâu: đồ thị, phát hiện đối tượng, biểu diễn mạng — ba hướng từ tập bài mẫu trong corpus thử nghiệm",
        "language": "vi",
        "relevant_source_ids": ["1906.11172v1", "2010.04895v2", "1910.01108v4"],
    },
]


def _load_manifest_ids() -> set[str]:
    if not _MANIFEST.is_file():
        return set()
    data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return {s["source_id"] for s in data.get("sources", [])}


def _validate_ids(items: list[dict[str, Any]], known: set[str]) -> list[str]:
    errors: list[str] = []
    for it in items:
        for sid in it.get("relevant_source_ids", []):
            if sid not in known:
                errors.append(f"{it.get('id')}: unknown source_id {sid!r} (ingest or typo)")
    return errors


def _build_full_payload() -> dict[str, Any]:
    if not _EVAL30.is_file():
        raise FileNotFoundError(f"Base eval not found: {_EVAL30}")
    base = json.loads(_EVAL30.read_text(encoding="utf-8"))
    first: list[dict[str, Any]] = []
    for row in base["items"]:
        first.append({**row, "language": "en"})
    items = first + _EXTRA_EN + _EXTRA_VI
    if len(items) != 100:
        raise RuntimeError(f"expected 100 items, got {len(items)}")
    return {
        "version": 2,
        "description": (
            "100 retrieval queries: 70 EN (q01-q70) + 30 VI (q71-q100). "
            "relevant_source_ids are multi-gold: union must intersect ranked chunks. "
            "source_id values match data/eval/ingest_manifest.json after corpus ingest."
        ),
        "counts": {"en": 70, "vi": 30, "items": 100},
        "items": items,
    }


def _write(out: Path) -> None:
    known = _load_manifest_ids()
    payload = _build_full_payload()
    errs = _validate_ids(payload["items"], known)
    if errs:
        msg = "Invalid qrels (unknown source_id):\n" + "\n".join(errs[:20])
        if len(errs) > 20:
            msg += f"\n... and {len(errs) - 20} more"
        raise SystemExit(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} (100 items, multi-relevant where noted)")


def _skeleton() -> None:
    known = _load_manifest_ids()
    if not known:
        print("No manifest found; cannot list sources", file=sys.stderr)
        sys.exit(2)
    items = [
        {
            "id": f"skel_{i:03d}",
            "query": f"TBD: natural question about {sid}",
            "language": "en",
            "relevant_source_ids": [sid],
        }
        for i, sid in enumerate(sorted(known), start=1)
    ]
    print(json.dumps({"version": 0, "items": items}, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build retrieval_eval_100.json from base eval + extras."
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help=f"Write {_DEFAULT_OUT} (fails if qrels not in current ingest_manifest).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output path for --write.",
    )
    ap.add_argument(
        "--skeleton",
        action="store_true",
        help="Print one TBD item per source_id (JSON to stdout) for manual authoring.",
    )
    args = ap.parse_args()
    if args.write:
        _write(args.out)
        return 0
    if args.skeleton:
        _skeleton()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
