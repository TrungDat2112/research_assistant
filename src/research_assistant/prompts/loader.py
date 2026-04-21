"""Versioned Jinja2 prompt loader.

Templates live next to this module (``*.jinja``) and are referenced by
their full filename (e.g. ``"planner_v1.jinja"``). Using the filename as
the version key is intentional — diffs are explicit and older versions
remain renderable for replay experiments (PLAN.md §8).

The ``Environment`` is constructed once with ``StrictUndefined`` so missing
context variables raise immediately instead of silently producing empty
strings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

_TEMPLATE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(enabled_extensions=()),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render(template_name: str, /, **context: Any) -> str:
    """Render ``template_name`` with the given ``context``.

    Raises
    ------
    jinja2.TemplateNotFound
        If the template file does not exist under ``prompts/``.
    jinja2.UndefinedError
        If a variable referenced in the template is missing from
        ``context`` (strict mode prevents silent typos).
    """
    template = _env().get_template(template_name)
    return template.render(**context)


def available_templates() -> list[str]:
    """Return the sorted list of ``*.jinja`` files available on disk."""
    return sorted(p.name for p in _TEMPLATE_DIR.glob("*.jinja"))
