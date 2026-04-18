"""Bandit security scanner wrapper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from commitlens.models import Finding

_SEVERITY_MAP: dict[str, str] = {
    "HIGH": "error",
    "MEDIUM": "warning",
    "LOW": "info",
}


def run_bandit(file_paths: list[str]) -> list[Finding]:
    """Run bandit on given Python files, return structured findings.

    Gracefully returns empty list if bandit is not installed,
    no Python files are given, or bandit times out.
    """
    python_files = [f for f in file_paths if f.endswith(".py")]
    if not python_files:
        return []

    existing = [f for f in python_files if Path(f).exists()]
    if not existing:
        return []

    try:
        result = subprocess.run(
            ["bandit", "-f", "json", "-ll", "--", *existing],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []

    if not result.stdout:
        return []

    try:
        bandit_output = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    findings: list[Finding] = []
    for issue in bandit_output.get("results", []):
        sev = issue.get("issue_severity", "LOW")
        findings.append(Finding(
            severity=_SEVERITY_MAP.get(sev, "info"),
            category="security",
            file=issue.get("filename", "unknown"),
            line=issue.get("line_number"),
            message=(
                f"{issue.get('issue_text', '')} "
                f"[{issue.get('test_id', '')}]"
            ),
            suggestion=issue.get("more_info"),
            source="bandit",
        ))

    return findings
