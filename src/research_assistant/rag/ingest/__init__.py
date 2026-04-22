"""Source adapters for ingesting raw documents into the RAG pipeline."""

from research_assistant.rag.ingest.arxiv_source import fetch_arxiv_doc, search_arxiv
from research_assistant.rag.ingest.html_source import fetch_html_doc
from research_assistant.rag.ingest.loader import IngestResult, SeedConfig, load_seed_corpus

__all__ = [
    "IngestResult",
    "SeedConfig",
    "fetch_arxiv_doc",
    "fetch_html_doc",
    "load_seed_corpus",
    "search_arxiv",
]
