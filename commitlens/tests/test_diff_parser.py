"""Tests for git diff parsing logic."""

from __future__ import annotations

from commitlens.diff_parser import LANGUAGE_MAP, _parse_hunks, truncate_diff


class TestParseHunks:
    def test_single_hunk(self) -> None:
        raw = (
            "@@ -1,3 +1,5 @@\n"
            " def greet(name):\n"
            '-    print("hello")\n'
            "+    if not name:\n"
            '+        raise ValueError("name required")\n'
            '+    print(f"hello {name}")\n'
        )
        hunks = _parse_hunks(raw)
        assert len(hunks) == 1
        assert len(hunks[0].added_lines) == 3
        assert len(hunks[0].removed_lines) == 1
        assert hunks[0].context_before == ["def greet(name):"]

    def test_multiple_hunks(self) -> None:
        raw = (
            "@@ -1,2 +1,2 @@\n"
            "-old_line_1\n"
            "+new_line_1\n"
            "@@ -10,2 +10,2 @@\n"
            "-old_line_2\n"
            "+new_line_2\n"
        )
        hunks = _parse_hunks(raw)
        assert len(hunks) == 2
        assert hunks[0].added_lines == ["new_line_1"]
        assert hunks[1].removed_lines == ["old_line_2"]

    def test_empty_diff(self) -> None:
        assert _parse_hunks("") == []

    def test_diff_header_lines_ignored(self) -> None:
        raw = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        hunks = _parse_hunks(raw)
        assert len(hunks) == 1
        assert hunks[0].added_lines == ["new"]
        assert hunks[0].removed_lines == ["old"]

    def test_context_after(self) -> None:
        raw = (
            "@@ -1,4 +1,4 @@\n"
            " before\n"
            "-old\n"
            "+new\n"
            " after\n"
        )
        hunks = _parse_hunks(raw)
        assert hunks[0].context_before == ["before"]
        assert hunks[0].context_after == ["after"]


class TestTruncateDiff:
    def test_short_diff_unchanged(self) -> None:
        diff = "line1\nline2\nline3"
        assert truncate_diff(diff, 10) == diff

    def test_long_diff_truncated(self) -> None:
        diff = "\n".join(f"line{i}" for i in range(100))
        result = truncate_diff(diff, 10)
        assert result.startswith("line0\n")
        assert "truncated 90 lines" in result
        assert "line10" not in result

    def test_exact_limit_unchanged(self) -> None:
        diff = "\n".join(f"line{i}" for i in range(10))
        assert truncate_diff(diff, 10) == diff

    def test_empty_diff(self) -> None:
        assert truncate_diff("", 10) == ""


class TestLanguageMap:
    def test_python(self) -> None:
        assert LANGUAGE_MAP[".py"] == "python"

    def test_javascript(self) -> None:
        assert LANGUAGE_MAP[".js"] == "javascript"

    def test_typescript(self) -> None:
        assert LANGUAGE_MAP[".ts"] == "typescript"

    def test_unknown_extension_not_in_map(self) -> None:
        assert ".xyz" not in LANGUAGE_MAP
