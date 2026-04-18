"""CommitLens CLI — AI-Powered Commit & Code Analyzer."""

from __future__ import annotations

import git as gitmodule
import typer
from rich.console import Console
from rich.prompt import Confirm

from commitlens import __version__
from commitlens.commit_agent import generate_commit_message
from commitlens.diff_parser import get_staged_diff
from commitlens.formatter import display_commit, display_review
from commitlens.review_agent import review_code

app = typer.Typer(
    name="commitlens",
    help="AI-Powered Commit & Code Analyzer",
    no_args_is_help=True,
)
console = Console()

_API_KEY_HELP = (
    "\n[bold]How to fix:[/bold]"
    "\n  1. Copy .env.example to .env"
    "\n  2. Set your API key: COMMITLENS_OPENAI_API_KEY=sk-..."
    "\n  3. Or set env var directly: "
    "$env:COMMITLENS_OPENAI_API_KEY='sk-...'"
)


def _handle_llm_error(e: Exception) -> None:
    """Print a user-friendly message for common LLM errors."""
    err_str = str(e).lower()
    if "api key" in err_str or "authenticate" in err_str or "401" in err_str:
        console.print("[red]API key missing or invalid.[/red]")
        console.print(_API_KEY_HELP)
    elif "rate limit" in err_str or "429" in err_str:
        console.print(
            "[red]Rate limited by LLM provider. Wait and retry.[/red]",
        )
    elif "timeout" in err_str:
        console.print("[red]LLM request timed out. Check your network and retry.[/red]")
    else:
        console.print(f"[red]LLM error: {e}[/red]")


@app.command()
def commit(
    repo_path: str = typer.Option(
        ".", "--repo", "-r", help="Path to git repository",
    ),
    auto_commit: bool = typer.Option(
        False, "--yes", "-y", help="Auto-commit without confirmation",
    ),
) -> None:
    """Generate a conventional commit message from staged changes."""
    try:
        diff_result = get_staged_diff(repo_path)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None

    console.print(
        f"[dim]Analyzing {diff_result.total_files} file(s)...[/dim]",
    )

    try:
        commit_msg = generate_commit_message(diff_result, repo_path)
    except Exception as e:  # noqa: BLE001
        _handle_llm_error(e)
        raise typer.Exit(code=1) from None

    display_commit(commit_msg)

    should_commit = auto_commit or Confirm.ask("Apply this commit?")
    if should_commit:
        repo = gitmodule.Repo(repo_path)
        repo.index.commit(commit_msg.format())
        console.print("[green]Committed successfully![/green]")
    else:
        console.print("[dim]Commit cancelled.[/dim]")


@app.command()
def review(
    repo_path: str = typer.Option(
        ".", "--repo", "-r", help="Path to git repository",
    ),
) -> None:
    """Review staged code changes for bugs, security issues, and style."""
    try:
        diff_result = get_staged_diff(repo_path)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None

    console.print(
        f"[dim]Reviewing {diff_result.total_files} file(s)...[/dim]",
    )

    try:
        result = review_code(diff_result, repo_path)
    except Exception as e:  # noqa: BLE001
        _handle_llm_error(e)
        raise typer.Exit(code=1) from None

    display_review(result)


@app.command()
def version() -> None:
    """Show CommitLens version."""
    console.print(f"CommitLens v{__version__}")


if __name__ == "__main__":
    app()
