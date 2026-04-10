"""Pydantic models — the data contract between all components."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field  # noqa: I001

# === Diff Models ===


class DiffHunk(BaseModel):
    """A single change hunk within a file."""

    header: str
    added_lines: list[str]
    removed_lines: list[str]
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)


class FileDiff(BaseModel):
    """Diff of a single file."""

    file_path: str
    change_type: Literal["added", "modified", "deleted", "renamed"]
    old_path: str | None = None
    hunks: list[DiffHunk] = Field(default_factory=list)
    language: str | None = None
    additions: int = 0
    deletions: int = 0


class DiffResult(BaseModel):
    """Complete diff result from git."""

    files: list[FileDiff]
    total_additions: int = 0
    total_deletions: int = 0
    total_files: int = 0


# === Commit Models ===


class CommitMessage(BaseModel):
    """Structured conventional commit message."""

    type: Literal[
        "feat", "fix", "refactor", "docs", "test",
        "chore", "style", "perf", "ci", "build",
    ]
    scope: str | None = None
    subject: str = Field(..., max_length=72)
    body: str | None = None
    breaking: bool = False

    def format(self) -> str:
        """Format into conventional commit string.

        Examples:
            >>> m = CommitMessage(type="feat", scope="auth", subject="add login")
            >>> m.format()
            'feat(auth): add login'
            >>> m = CommitMessage(type="fix", subject="crash", breaking=True)
            >>> m.format()
            'fix!: crash'
        """
        prefix = self.type
        if self.scope:
            prefix += f"({self.scope})"
        if self.breaking:
            prefix += "!"
        msg = f"{prefix}: {self.subject}"
        if self.body:
            msg += f"\n\n{self.body}"
        return msg


# === Review Models ===


class Finding(BaseModel):
    """A single review finding."""

    severity: Literal["error", "warning", "info"]
    category: Literal["bug", "security", "style", "performance"]
    file: str
    line: int | None = None
    message: str
    suggestion: str | None = None
    source: Literal["llm", "bandit"] = "llm"


class ReviewResult(BaseModel):
    """Complete review result."""

    findings: list[Finding] = Field(default_factory=list)
    summary: str
    risk_level: Literal["low", "medium", "high"]
    files_reviewed: int = 0
