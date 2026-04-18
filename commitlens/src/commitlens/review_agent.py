"""Code review agent using LLM + bandit static analysis."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from commitlens.config import settings
from commitlens.diff_parser import get_diff_text, truncate_diff
from commitlens.models import DiffResult, Finding, ReviewResult
from commitlens.security import run_bandit

SYSTEM_PROMPT = """\
You are an expert code reviewer. Review the following code changes and \
identify issues.

Focus on these categories (in priority order):
1. **bug**: Logic errors, null references, off-by-one, race conditions
2. **security**: Hardcoded secrets, injection risks, unsafe operations
3. **performance**: N+1 queries, unnecessary loops, memory leaks
4. **style**: Naming, complexity, missing error handling (only if significant)

Rules:
- Only report REAL issues, not style nitpicks
- Each finding must have a specific file and line number when possible
- Provide actionable suggestions
- If no issues found, return empty findings with a positive summary
- Set risk_level based on worst finding severity
- Do NOT repeat issues already found by the static analyzer (listed below)
"""

USER_PROMPT_TEMPLATE = """\
Review these code changes:

Files changed: {file_summary}
Total: +{additions} -{deletions} across {total_files} file(s)

{bandit_section}

Diff:
```
{diff_text}
```
"""


def _get_llm() -> ChatOpenAI | ChatAnthropic:
    """Return configured LLM instance based on provider setting."""
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.model_name,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def review_code(
    diff_result: DiffResult,
    repo_path: str = ".",
) -> ReviewResult:
    """Review staged changes using LLM + bandit.

    Args:
        diff_result: Parsed diff from diff_parser.
        repo_path: Path to git repo for raw diff text.

    Returns:
        ReviewResult with merged findings from LLM and bandit.
    """
    changed_files = [f.file_path for f in diff_result.files]

    bandit_findings: list[Finding] = []
    if settings.bandit_enabled:
        bandit_findings = run_bandit(changed_files)

    bandit_section = _build_bandit_section(bandit_findings)

    raw_diff = get_diff_text(repo_path)
    raw_diff = truncate_diff(raw_diff, settings.max_diff_lines)
    file_summary = ", ".join(f.file_path for f in diff_result.files)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])

    llm = _get_llm()
    chain = prompt | llm.with_structured_output(ReviewResult)

    llm_result: ReviewResult = chain.invoke({  # type: ignore[assignment]
        "file_summary": file_summary,
        "additions": diff_result.total_additions,
        "deletions": diff_result.total_deletions,
        "total_files": diff_result.total_files,
        "bandit_section": bandit_section,
        "diff_text": raw_diff,
    })

    all_findings = _merge_findings(llm_result.findings, bandit_findings)

    return ReviewResult(
        findings=all_findings,
        summary=llm_result.summary,
        risk_level=llm_result.risk_level,
        files_reviewed=diff_result.total_files,
    )


def review_from_raw_diff(diff_text: str) -> ReviewResult:
    """Review raw diff text without git repo (for eval & Streamlit).

    Args:
        diff_text: Raw unified diff string.

    Returns:
        ReviewResult with LLM findings (no bandit — no real files).
    """
    diff_text = truncate_diff(diff_text, settings.max_diff_lines)
    n_files = max(diff_text.count("diff --git"), 1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT_TEMPLATE),
    ])

    llm = _get_llm()
    chain = prompt | llm.with_structured_output(ReviewResult)

    result: ReviewResult = chain.invoke({  # type: ignore[assignment]
        "file_summary": f"{n_files} file(s)",
        "additions": diff_text.count("\n+") - diff_text.count("\n+++"),
        "deletions": diff_text.count("\n-") - diff_text.count("\n---"),
        "total_files": n_files,
        "bandit_section": "Static analyzer not available (raw diff mode).",
        "diff_text": diff_text,
    })

    return result


def _build_bandit_section(findings: list[Finding]) -> str:
    """Format bandit findings as context for the LLM prompt."""
    if not findings:
        return "Static analyzer found no issues."
    lines = [
        "Static analyzer already found these issues (do NOT repeat):",
    ]
    for f in findings:
        lines.append(f"- [{f.severity}] {f.file}:{f.line} — {f.message}")
    return "\n".join(lines)


def _merge_findings(
    llm_findings: list[Finding],
    bandit_findings: list[Finding],
) -> list[Finding]:
    """Merge and deduplicate findings from LLM and bandit.

    If both flag the same (file, line), keep bandit's version
    since it has lower false-positive rates for known patterns.
    """
    bandit_keys = {(f.file, f.line) for f in bandit_findings}
    unique_llm = [
        f for f in llm_findings
        if (f.file, f.line) not in bandit_keys
    ]

    severity_order = {"error": 0, "warning": 1, "info": 2}
    merged = bandit_findings + unique_llm
    merged.sort(key=lambda f: severity_order.get(f.severity, 3))

    return merged
