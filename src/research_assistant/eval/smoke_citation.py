from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from research_assistant.agents.critic import paragraph_citation_stats

_QUERY_HEADER_RE = re.compile(
    r"^# Query (?P<idx>\d+) \((?P<lang>VI|EN)\): (?P<title>.+)$",
    re.MULTILINE,
)
_REF_SECTION_RE = re.compile(
    r"\n##\s*(?:Tài liệu tham khảo|References)\s*\n",
    re.IGNORECASE,
)


def _is_numbered_section_header(p: str) -> bool:
    return bool(re.match(r"^##\s+\d+\.\s", p.strip()))


def _is_markdown_heading_only(p: str) -> bool:
    lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    return bool(re.match(r"^#{1,6}\s+\S", lines[0]))


def _is_bold_heading_only(p: str) -> bool:
    lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    return bool(re.match(r"^\*\*.+\*\*\s*:?\s*$", lines[0]))


def _is_no_synthesized_placeholder(p: str) -> bool:
    low = p.lower()
    return "no synthesized answer" in low


def _is_insufficient_paragraph(p: str) -> bool:
    t = p.lower()
    return (
        "insufficient evidence" in t
        or "chưa đủ dữ liệu" in t
        or "không đủ dữ liệu để kết luận" in t
    )


def skip_smoke_report_paragraph(p: str) -> bool:
    return (
        _is_numbered_section_header(p)
        or _is_markdown_heading_only(p)
        or _is_bold_heading_only(p)
        or _is_no_synthesized_placeholder(p)
        or _is_insufficient_paragraph(p)
    )


def extract_report_body(query_block: str) -> str:
    m_ref = _REF_SECTION_RE.search(query_block)
    body = query_block[: m_ref.start()] if m_ref else query_block
    m_first = re.search(r"^##\s+\d+\.\s", body, re.MULTILINE)
    if not m_first:
        return body.strip()
    return body[m_first.start() :].strip()


@dataclass(frozen=True)
class SmokeQueryCitation:
    query_index: int
    language: Literal["vi", "en"]
    title: str
    paragraph_citation_coverage: float
    substantive_paragraphs: int
    cited_paragraphs: int


def parse_smoke_markdown(md: str) -> list[SmokeQueryCitation]:
    matches = list(_QUERY_HEADER_RE.finditer(md))
    if not matches:
        msg = "no '# Query N (VI|EN):' headers found — expected smoke_outputs.md"
        raise ValueError(msg)

    out: list[SmokeQueryCitation] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        block = md[start:end]
        idx = int(m.group("idx"))
        lang = cast(Literal["vi", "en"], m.group("lang").lower())
        title = m.group("title").strip()
        body = extract_report_body(block)
        cov, cited, n_sub = paragraph_citation_stats(
            body,
            skip_paragraph=skip_smoke_report_paragraph,
            apply_full_body_insufficient_guard=False,
        )
        out.append(
            SmokeQueryCitation(
                query_index=idx,
                language=lang,
                title=title,
                paragraph_citation_coverage=round(cov, 6),
                substantive_paragraphs=n_sub,
                cited_paragraphs=cited,
            ),
        )
    return out


def run_smoke_citation_eval(
    smoke_md_path: Path,
    *,
    target_mean_coverage: float,
) -> dict[str, Any]:
    text = smoke_md_path.read_text(encoding="utf-8")
    rows = parse_smoke_markdown(text)
    mean_cov = sum(r.paragraph_citation_coverage for r in rows) / len(rows)
    return {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "metric": "paragraph_citation_coverage",
        "smoke_markdown": str(smoke_md_path.as_posix()),
        "n_queries": len(rows),
        "target_mean_coverage": target_mean_coverage,
        "mean_coverage": round(mean_cov, 6),
        "meets_target": mean_cov >= target_mean_coverage,
        "queries": [
            {
                "query_index": r.query_index,
                "language": r.language,
                "title": r.title,
                "paragraph_citation_coverage": r.paragraph_citation_coverage,
                "substantive_paragraphs": r.substantive_paragraphs,
                "cited_paragraphs": r.cited_paragraphs,
            }
            for r in rows
        ],
    }
