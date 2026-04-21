"""Smoke test — verifies the package imports and exposes __version__."""

from __future__ import annotations

import research_assistant


def test_package_version_is_exposed() -> None:
    assert isinstance(research_assistant.__version__, str)
    assert research_assistant.__version__ != ""
