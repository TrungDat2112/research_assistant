"""Tests for eval JSON loading and schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_assistant.eval.retrieval import load_retrieval_eval

_REPO = Path(__file__).resolve().parents[2]
_EVAL100 = _REPO / "data" / "eval" / "retrieval_eval_100.json"
_EVAL30 = _REPO / "data" / "eval" / "retrieval_eval_30.json"


def test_loads_hundred_item_eval() -> None:
    if not _EVAL100.is_file():
        pytest.skip("retrieval_eval_100.json not in workspace")
    items = load_retrieval_eval(_EVAL100)
    assert len(items) == 100
    assert all(i.id and i.query and i.relevant_source_ids for i in items)
    n_en = sum(1 for i in items if i.language == "en")
    n_vi = sum(1 for i in items if i.language == "vi")
    assert n_en == 70
    assert n_vi == 30
    assert any(len(i.relevant_source_ids) > 1 for i in items), "expect multi-gold qrels"
    # q01-30: legacy v1 (implicit language = en in file or explicit)
    assert items[0].id == "q01" and items[0].language == "en"


def test_loads_legacy_thirty_file_without_language_key() -> None:
    if not _EVAL30.is_file():
        pytest.skip("retrieval_eval_30.json not in workspace")
    items = load_retrieval_eval(_EVAL30)
    assert len(items) == 30
    assert all(i.language == "en" for i in items)
    assert all(len(i.relevant_source_ids) == 1 for i in items)
