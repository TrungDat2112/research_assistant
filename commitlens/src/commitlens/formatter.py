"""Rich console output formatting for CLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from commitlens.models import CommitMessage, ReviewResult

console = Console()

_UNICODE_OK = console.encoding != "cp1252" if console.encoding else True

SEVERITY_STYLES: dict[str, tuple[str, str]] = {
    "error": ("bold red", "X" if not _UNICODE_OK else "✗"),
    "warning": ("yellow", "!" if not _UNICODE_OK else "⚠"),
    "info": ("cyan", "i" if not _UNICODE_OK else "ℹ"),
}

CATEGORY_LABELS: dict[str, str] = {
    "bug": "Bug",
    "security": "Security",
    "style": "Style",
    "performance": "Performance",
}


def display_commit(commit: CommitMessage) -> None:
    """Display formatted commit message in a green bordered panel."""
    panel = Panel(
        commit.format(),
        title="Generated Commit Message",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)

    if commit.breaking:
        icon = "!" if not _UNICODE_OK else "⚠"
        console.print(f"[bold red]{icon} BREAKING CHANGE[/bold red]")


def display_review(result: ReviewResult) -> None:
    """Display formatted review results with findings table."""
    risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
    risk_color = risk_colors.get(result.risk_level, "white")

    console.print(
        f"\n[bold]Code Review[/bold] — "
        f"[{risk_color}]{result.risk_level.upper()} risk"
        f"[/{risk_color}] — "
        f"{result.files_reviewed} file(s) reviewed\n"
    )

    if not result.findings:
        check = "OK" if not _UNICODE_OK else "✓"
        console.print(
            f"[green]{check} No issues found. Code looks good![/green]"
        )
    else:
        _print_findings_table(result)
        _print_suggestions(result)

    console.print(f"\n[dim]{result.summary}[/dim]\n")


def _print_findings_table(result: ReviewResult) -> None:
    """Render the findings as a Rich table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=3)
    table.add_column("Category", width=12)
    table.add_column("Location", width=30)
    table.add_column("Issue", ratio=1)

    for finding in result.findings:
        style, icon = SEVERITY_STYLES.get(
            finding.severity, ("white", "?"),
        )
        location = finding.file
        if finding.line:
            location += f":{finding.line}"
        cat_label = CATEGORY_LABELS.get(
            finding.category, finding.category,
        )
        table.add_row(
            Text(icon, style=style),
            cat_label,
            location,
            finding.message,
        )

    console.print(table)


def _print_suggestions(result: ReviewResult) -> None:
    """Print actionable suggestions below the findings table."""
    suggestions = [f for f in result.findings if f.suggestion]
    if not suggestions:
        return

    console.print("\n[bold]Suggestions:[/bold]")
    for finding in suggestions:
        loc = finding.file
        if finding.line:
            loc += f":{finding.line}"
        arrow = "->" if not _UNICODE_OK else "→"
        console.print(f"  {arrow} {loc}: {finding.suggestion}")
