"""Tests for :mod:`research_assistant.tools.fetch_pdf`.

HTTP is exercised via ``httpx.MockTransport``; pymupdf is mocked where the
exercise is fetch/cache logic, and patched once to satisfy the "mock pymupdf"
contract without requiring a real PDF fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from research_assistant.tools.fetch_pdf import (
    MAX_PDF_BYTES,
    FetchPdfError,
    _cache_key_url,
    _parse_arxiv_id,
    _parse_direct_pdf_url,
    fetch_pdf,
)

_MIN_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj trailer<<>>\n%%EOF\n"


def _transport_pdf(content: bytes = _MIN_PDF) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    return httpx.MockTransport(handler)


def test_parse_arxiv_bare_id() -> None:
    assert _parse_arxiv_id("2404.16130v2") == "2404.16130v2"


def test_parse_arxiv_abs_url() -> None:
    assert _parse_arxiv_id("https://arxiv.org/abs/2404.16130") == "2404.16130"


def test_parse_arxiv_pdf_url() -> None:
    assert _parse_arxiv_id("https://arxiv.org/pdf/2404.16130.pdf") == "2404.16130"


def test_parse_direct_pdf_url() -> None:
    u = "https://example.com/paper.pdf"
    assert _parse_direct_pdf_url(u) == u


def test_parse_direct_pdf_rejects_non_pdf_path() -> None:
    assert _parse_direct_pdf_url("https://example.com/page.html") is None


def test_cache_key_stable() -> None:
    u = "https://arxiv.org/pdf/2404.16130.pdf"
    assert _cache_key_url(u) == _cache_key_url(u)


def test_fetch_pdf_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(FetchPdfError, match="Empty"):
        fetch_pdf("  ", cache_dir=tmp_path, client=httpx.Client(transport=_transport_pdf()))


def test_fetch_pdf_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(FetchPdfError, match="Expected an arXiv"):
        fetch_pdf(
            "https://example.com/doc.html",
            cache_dir=tmp_path,
            client=httpx.Client(transport=_transport_pdf()),
        )


def test_fetch_pdf_arxiv_happy_path(tmp_path: Path) -> None:
    def extract(_path: Path, max_c: int) -> str:
        return "extracted body"[:max_c]

    client = httpx.Client(transport=_transport_pdf(), timeout=30.0)
    doc = fetch_pdf(
        "2404.16130",
        cache_dir=tmp_path,
        client=client,
        extract_pdf_text_fn=extract,
    )
    assert doc.source_id == "2404.16130"
    assert doc.doc_type == "pdf"
    assert "arxiv.org/abs/2404.16130" in doc.url
    assert doc.text == "extracted body"
    cached = tmp_path / f"{_cache_key_url('https://arxiv.org/pdf/2404.16130.pdf')}.pdf"
    assert cached.is_file()


def test_fetch_pdf_cache_skips_second_download(tmp_path: Path) -> None:
    calls: dict[str, int] = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=_MIN_PDF)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)

    def extract(_p: Path, _m: int) -> str:
        return "x"

    fetch_pdf("2404.16130", cache_dir=tmp_path, client=client, extract_pdf_text_fn=extract)
    fetch_pdf("2404.16130", cache_dir=tmp_path, client=client, extract_pdf_text_fn=extract)
    assert calls["n"] == 1


def test_fetch_pdf_oversize_raises(tmp_path: Path) -> None:
    big = b"%PDF" + b"x" * (MAX_PDF_BYTES + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    with pytest.raises(FetchPdfError, match="exceeds max size"):
        fetch_pdf(
            "2404.16130",
            cache_dir=tmp_path,
            client=client,
            extract_pdf_text_fn=lambda _p, _m: "ok",
        )


def test_fetch_pdf_non_pdf_magic_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"NOTPDF")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    with pytest.raises(FetchPdfError, match="not a PDF"):
        fetch_pdf(
            "2404.16130",
            cache_dir=tmp_path,
            client=client,
            extract_pdf_text_fn=lambda _p, _m: "ok",
        )


def test_fetch_pdf_http_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"nope")

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    with pytest.raises(FetchPdfError, match="404"):
        fetch_pdf(
            "2404.16130",
            cache_dir=tmp_path,
            client=client,
            extract_pdf_text_fn=lambda _p, _m: "ok",
        )


def test_fetch_pdf_uses_pymupdf_when_not_injected(tmp_path: Path) -> None:
    """Patch pymupdf.open so extraction runs without a real PDF on disk."""

    class _FakePage:
        def get_text(self, mode: str) -> str:
            return "synthetic page"

    class _FakeDoc:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {"title": " Titled "}

        def __iter__(self) -> Any:
            return iter([_FakePage()])

        def close(self) -> None:
            return None

    fake_doc = _FakeDoc()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_MIN_PDF)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)

    with patch("pymupdf.open", return_value=fake_doc) as m_open:
        doc = fetch_pdf("2404.16130", cache_dir=tmp_path, client=client)
    m_open.assert_called_once()
    assert doc.text == "synthetic page"
    assert doc.title.strip() == "Titled"


def test_fetch_pdf_truncates_via_mock_pymupdf(tmp_path: Path) -> None:
    class _FakePage:
        def get_text(self, mode: str) -> str:
            return "abcde"

    class _FakeDoc:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}

        def __iter__(self) -> Any:
            return iter([_FakePage(), _FakePage()])

        def close(self) -> None:
            return None

    client = httpx.Client(transport=_transport_pdf(), timeout=30.0)
    with patch("pymupdf.open", return_value=_FakeDoc()):
        doc = fetch_pdf("2404.16130", cache_dir=tmp_path, client=client, max_chars=7)
    assert doc.text == "abcde\na"


def test_fetch_pdf_empty_extract_raises(tmp_path: Path) -> None:
    class _FakeDoc:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}

        def __iter__(self) -> Any:
            return iter([])

        def close(self) -> None:
            return None

    client = httpx.Client(transport=_transport_pdf(), timeout=30.0)
    with (
        patch("pymupdf.open", return_value=_FakeDoc()),
        pytest.raises(FetchPdfError, match="No text"),
    ):
        fetch_pdf("2404.16130", cache_dir=tmp_path, client=client)


def test_fetch_pdf_direct_url(tmp_path: Path) -> None:
    url = "https://files.example.com/pub/paper.pdf"
    key = _cache_key_url(url)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == url
        return httpx.Response(200, content=_MIN_PDF)

    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=30.0)
    doc = fetch_pdf(url, cache_dir=tmp_path, client=client, extract_pdf_text_fn=lambda _p, _m: "hi")
    assert doc.url == url
    assert doc.source_id.startswith("h_")
    assert (tmp_path / f"{key}.pdf").is_file()


def test_mock_transport_uses_timeout_compat() -> None:
    """Smoke: MockTransport works with the same client shape as production."""
    client = httpx.Client(transport=_transport_pdf(), timeout=httpx.Timeout(30.0))
    try:
        r = client.get("https://example.com/x.pdf")
        assert r.status_code == 200
        assert r.content.startswith(b"%PDF")
    finally:
        client.close()
