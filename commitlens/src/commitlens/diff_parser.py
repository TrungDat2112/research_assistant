"""Git diff parsing — extract staged changes into structured format."""

from __future__ import annotations

from pathlib import Path

import git

from commitlens.models import DiffHunk, DiffResult, FileDiff

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".php": "php",
}


def get_staged_diff(repo_path: str = ".") -> DiffResult:
    """Get structured diff of staged changes.

    Args:
        repo_path: Path to git repository root.

    Returns:
        DiffResult with parsed file diffs and statistics.

    Raises:
        git.InvalidGitRepositoryError: If path is not inside a git repo.
        ValueError: If there are no staged changes.
    """
    repo = git.Repo(repo_path)

    if repo.head.is_valid():
        staged = repo.index.diff("HEAD")
    else:
        staged = repo.index.diff(git.NULL_TREE)

    if not staged:
        raise ValueError("No staged changes found. Run 'git add' first.")

    files: list[FileDiff] = []
    for diff_item in staged:
        file_diff = _parse_diff_item(repo, diff_item)
        files.append(file_diff)

    return DiffResult(
        files=files,
        total_additions=sum(f.additions for f in files),
        total_deletions=sum(f.deletions for f in files),
        total_files=len(files),
    )


def get_diff_text(repo_path: str = ".") -> str:
    """Get raw diff text of staged changes (for LLM prompt).

    Returns the unified diff string with 3 lines of context.
    """
    repo = git.Repo(repo_path)
    return repo.git.diff("--cached", "--unified=3")


def truncate_diff(diff_text: str, max_lines: int = 500) -> str:
    """Truncate diff to *max_lines*, keeping the first N lines."""
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    truncated = lines[:max_lines]
    remaining = len(lines) - max_lines
    truncated.append(f"\n... (truncated {remaining} lines)")
    return "\n".join(truncated)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_diff_item(repo: git.Repo, diff_item: git.Diff) -> FileDiff:
    """Parse a single ``git.Diff`` object into a ``FileDiff`` model."""
    if diff_item.new_file:
        change_type = "added"
    elif diff_item.deleted_file:
        change_type = "deleted"
    elif diff_item.renamed_file:
        change_type = "renamed"
    else:
        change_type = "modified"

    file_path: str = diff_item.b_path or diff_item.a_path or ""
    ext = Path(file_path).suffix
    language = LANGUAGE_MAP.get(ext)

    raw_diff: str = repo.git.diff(
        "--cached", "--unified=3", "--", file_path,
    )
    hunks = _parse_hunks(raw_diff)

    additions = sum(len(h.added_lines) for h in hunks)
    deletions = sum(len(h.removed_lines) for h in hunks)

    return FileDiff(
        file_path=file_path,
        change_type=change_type,
        old_path=diff_item.a_path if change_type == "renamed" else None,
        hunks=hunks,
        language=language,
        additions=additions,
        deletions=deletions,
    )


def _parse_hunks(raw_diff: str) -> list[DiffHunk]:
    """Parse unified diff text into a list of ``DiffHunk``."""
    hunks: list[DiffHunk] = []
    current_hunk: dict[str, object] | None = None

    for line in raw_diff.splitlines():
        if line.startswith("@@"):
            if current_hunk is not None:
                hunks.append(DiffHunk(**current_hunk))  # type: ignore[arg-type]
            current_hunk = {
                "header": line,
                "added_lines": [],
                "removed_lines": [],
                "context_before": [],
                "context_after": [],
            }
        elif current_hunk is not None:
            added: list[str] = current_hunk["added_lines"]  # type: ignore[assignment]
            removed: list[str] = current_hunk["removed_lines"]  # type: ignore[assignment]
            ctx_before: list[str] = current_hunk["context_before"]  # type: ignore[assignment]
            ctx_after: list[str] = current_hunk["context_after"]  # type: ignore[assignment]

            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:])
            elif line.startswith(" "):
                if not added and not removed:
                    ctx_before.append(line[1:])
                else:
                    ctx_after.append(line[1:])

    if current_hunk is not None:
        hunks.append(DiffHunk(**current_hunk))  # type: ignore[arg-type]

    return hunks
