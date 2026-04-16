"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from commitlens.models import (
    CommitMessage,
    DiffHunk,
    DiffResult,
    FileDiff,
    Finding,
    ReviewResult,
)

# === CommitMessage.format() ===


class TestCommitMessageFormat:
    def test_simple_with_scope(self) -> None:
        msg = CommitMessage(
            type="feat", scope="auth", subject="add login endpoint",
        )
        assert msg.format() == "feat(auth): add login endpoint"

    def test_no_scope(self) -> None:
        msg = CommitMessage(
            type="fix", subject="resolve crash on startup",
        )
        assert msg.format() == "fix: resolve crash on startup"

    def test_breaking_change(self) -> None:
        msg = CommitMessage(
            type="feat",
            scope="api",
            subject="change response format",
            breaking=True,
        )
        assert msg.format() == "feat(api)!: change response format"

    def test_breaking_no_scope(self) -> None:
        msg = CommitMessage(
            type="fix", subject="drop legacy endpoint", breaking=True,
        )
        assert msg.format() == "fix!: drop legacy endpoint"

    def test_with_body(self) -> None:
        msg = CommitMessage(
            type="feat",
            scope="auth",
            subject="add login",
            body="- Add endpoint\n- Add validation",
        )
        result = msg.format()
        assert result.startswith("feat(auth): add login")
        assert "- Add endpoint" in result
        assert "- Add validation" in result
        assert "\n\n" in result

    def test_all_commit_types(self) -> None:
        valid_types = [
            "feat", "fix", "refactor", "docs", "test",
            "chore", "style", "perf", "ci", "build",
        ]
        for t in valid_types:
            msg = CommitMessage(type=t, subject="some change")
            assert msg.format().startswith(f"{t}: ")

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CommitMessage(type="invalid", subject="some change")

    def test_subject_max_length(self) -> None:
        with pytest.raises(ValidationError):
            CommitMessage(type="feat", subject="x" * 73)

    def test_subject_at_max_length(self) -> None:
        msg = CommitMessage(type="feat", subject="x" * 72)
        assert len(msg.subject) == 72


# === DiffHunk defaults ===


class TestDiffHunk:
    def test_defaults(self) -> None:
        hunk = DiffHunk(
            header="@@ -1,3 +1,5 @@",
            added_lines=["a"],
            removed_lines=["b"],
        )
        assert hunk.context_before == []
        assert hunk.context_after == []


# === FileDiff ===


class TestFileDiff:
    def test_minimal(self) -> None:
        fd = FileDiff(file_path="main.py", change_type="modified")
        assert fd.hunks == []
        assert fd.language is None
        assert fd.additions == 0

    def test_renamed(self) -> None:
        fd = FileDiff(
            file_path="new_name.py",
            change_type="renamed",
            old_path="old_name.py",
        )
        assert fd.old_path == "old_name.py"


# === DiffResult ===


class TestDiffResult:
    def test_totals(self) -> None:
        files = [
            FileDiff(
                file_path="a.py", change_type="modified",
                additions=5, deletions=2,
            ),
            FileDiff(
                file_path="b.py", change_type="added",
                additions=10, deletions=0,
            ),
        ]
        dr = DiffResult(
            files=files,
            total_additions=15,
            total_deletions=2,
            total_files=2,
        )
        assert dr.total_files == 2
        assert dr.total_additions == 15


# === Finding ===


class TestFinding:
    def test_defaults(self) -> None:
        f = Finding(
            severity="error",
            category="bug",
            file="main.py",
            message="null ref",
        )
        assert f.source == "llm"
        assert f.line is None
        assert f.suggestion is None

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Finding(
                severity="critical",
                category="bug",
                file="x.py",
                message="bad",
            )


# === ReviewResult ===


class TestReviewResult:
    def test_empty_findings(self) -> None:
        r = ReviewResult(
            summary="All good", risk_level="low",
        )
        assert r.findings == []
        assert r.files_reviewed == 0
