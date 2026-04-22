"""Tests for eval JSON loading and schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant.eval.retrieval import load_retrieval_eval

_REPO = Path(__file__).resolve().parents[2]
_EVAL_PATH = _REPO / "data" / "eval" / "retrieval_eval_30.json"


def test_loads_thirty_items() -> None:
    if not _EVAL_PATH.is_file():
        pytest.skip("retrieval_eval_30.json not in workspace")
    items = load_retrieval_eval(_EVAL_PATH)
    assert len(items) == 30
    assert all(i.id and i.query and i.relevant_source_ids for i in items)
    # Single-doc labels in our eval set
    assert all(len(i.relevant_source_ids) == 1 for i in items)
