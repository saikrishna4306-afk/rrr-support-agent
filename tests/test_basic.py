from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.ingest import chunk_text, ingest, parse_doc
from backend.language import detect_language
from backend.main import app
from backend.rag import get_retriever, reset_retriever


@pytest.fixture(scope="module", autouse=True)
def indexed():
    count = ingest()
    reset_retriever()
    return count


def test_detect_language():
    assert detect_language("my thread is breaking") == "en"
    assert detect_language("పై దారం తెగిపోతోంది") == "te"
    assert detect_language("धागा टूट रहा है") == "hi"


def test_parse_doc_and_chunking():
    doc = parse_doc(Path(settings.docs_path) / "thread_breaking_en.md")
    assert doc["meta"]["title"] == "Upper thread keeps breaking"
    chunks = chunk_text(doc["body"], max_chars=200)
    assert len(chunks) > 1 and all(len(c) <= 400 for c in chunks)


def test_ingest_indexed_every_document(indexed):
    assert indexed >= len(list(Path(settings.docs_path).glob("*.md")))


def test_search_finds_the_right_guide():
    hits = get_retriever().search("my upper thread keeps breaking")
    assert hits[0].meta["source"] == "thread_breaking_en.md"


def test_error_code_found_by_keyword_search():
    hits = get_retriever().search("E21 on screen")
    assert hits[0].meta["source"] == "error_e21_en.md"


def test_chat_answers_and_hands_off():
    with TestClient(app) as client:
        ok = client.post("/chat", json={"session_id": "test-session-1", "message": "needle keeps breaking"})
        assert ok.status_code == 200
        assert ok.json()["handoff"] is False
        assert ok.json()["sources"][0]["source"] == "needle_breaking_en.md"

        unknown = client.post("/chat", json={"session_id": "test-session-1", "message": "xqzv blorp"})
        body = unknown.json()
        assert body["handoff"] is True and body["ticket_id"] is not None
