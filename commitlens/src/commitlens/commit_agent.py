"""Commit message generation agent using LLM with structured output."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from commitlens.config import settings
from commitlens.diff_parser import get_diff_text, truncate_diff
from commitlens.models import CommitMessage, DiffResult

SYSTEM_PROMPT = """\
You are a commit message generator. Generate a single commit message following \
Conventional Commits specification.

Rules:
- type: one of feat, fix, refactor, docs, test, chore, style, perf, ci, build
- scope: the module or area affected (optional, lowercase, short)
- subject: imperative mood, lowercase start, no period, max 72 chars
- body: bullet points of key changes (optional, only for complex changes)
- breaking: true only if this introduces a breaking change

Analyze the diff carefully:
- feat: new feature or capability added
- fix: bug fix
- refactor: code restructuring without behavior change
- docs: documentation only
- test: adding or updating tests
- chore: maintenance tasks (deps, configs)
- style: formatting, whitespace
- perf: performance improvement
- ci: CI/CD changes
- build: build system changes
"""

FEW_SHOT_EXAMPLES = """
Example 1:
Diff: Added new /users endpoint with GET and POST handlers in routes/users.py
Message: feat(users): add CRUD endpoints for user management

Example 2:
Diff: Fixed off-by-one error in pagination logic in utils/paginator.py
Message: fix(pagination): correct off-by-one error in page calculation

Example 3:
Diff: Moved database connection setup from main.py to db/connection.py
Message: refactor(db): extract database connection setup to dedicated module

Example 4:
Diff: Updated pytest fixtures in tests/conftest.py, added tests for auth module
Message: test(auth): add unit tests for token validation

Example 5:
Diff: Bumped fastapi from 0.109 to 0.115 in pyproject.toml
Message: chore(deps): bump fastapi to 0.115
"""

USER_PROMPT_TEMPLATE = """\
Generate a commit message for the following changes:

Files changed: {file_summary}
Total: +{additions} -{deletions} across {total_files} file(s)

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


def generate_commit_message(
    diff_result: DiffResult,
    repo_path: str = ".",
) -> CommitMessage:
    """Generate a conventional commit message from staged diff.

    Args:
        diff_result: Parsed diff from diff_parser.
        repo_path: Path to git repo for raw diff text.

    Returns:
        CommitMessage with structured commit data.
    """
    raw_diff = get_diff_text(repo_path)
    raw_diff = truncate_diff(raw_diff, settings.max_diff_lines)

    file_summary = ", ".join(f.file_path for f in diff_result.files)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + FEW_SHOT_EXAMPLES),
        ("human", USER_PROMPT_TEMPLATE),
    ])

    llm = _get_llm()
    chain = prompt | llm.with_structured_output(CommitMessage)

    result: CommitMessage = chain.invoke({  # type: ignore[assignment]
        "file_summary": file_summary,
        "additions": diff_result.total_additions,
        "deletions": diff_result.total_deletions,
        "total_files": diff_result.total_files,
        "diff_text": raw_diff,
    })

    return result


def generate_from_raw_diff(diff_text: str) -> CommitMessage:
    """Generate commit message from raw diff text (for eval & Streamlit).

    Unlike ``generate_commit_message``, this does NOT require a git repo.

    Args:
        diff_text: Raw unified diff string.

    Returns:
        CommitMessage with structured commit data.
    """
    diff_text = truncate_diff(diff_text, settings.max_diff_lines)

    n_files = diff_text.count("diff --git")
    additions = diff_text.count("\n+") - diff_text.count("\n+++")
    deletions = diff_text.count("\n-") - diff_text.count("\n---")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + FEW_SHOT_EXAMPLES),
        ("human", USER_PROMPT_TEMPLATE),
    ])

    llm = _get_llm()
    chain = prompt | llm.with_structured_output(CommitMessage)

    result: CommitMessage = chain.invoke({  # type: ignore[assignment]
        "file_summary": f"{n_files} file(s)",
        "additions": additions,
        "deletions": deletions,
        "total_files": n_files or 1,
        "diff_text": diff_text,
    })

    return result
