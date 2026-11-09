import pytest

from fastapi.testclient import TestClient

from backend import db
from backend.config import settings
from backend.main import app


@pytest.fixture(scope="module", autouse=True)
def initialized_db():
    db.init_db()


def test_diagnose_creates_ticket_for_photo(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "evidence_path", str(tmp_path))
    with TestClient(app) as client:
        response = client.post("/diagnose", data={"session_id": "diagnose-1", "message": "What is this part?"},
                               files={"evidence": ("part.jpg", b"image-data", "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] and body["handoff"]
    ticket = db.get_ticket(body["ticket_id"])
    assert ticket["has_evidence"] and ticket["evidence_type"] == "image/jpeg"


def test_technician_endpoints_require_token(monkeypatch):
    monkeypatch.setattr(settings, "technician_token", "test-token")
    with TestClient(app) as client:
        assert client.get("/technician/tickets").status_code == 401
        response = client.get("/technician/tickets", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200


def test_ticket_status_update():
    ticket_id = db.create_ticket("technician-test", "Need help")
    updated = db.update_ticket(ticket_id, "resolved", "Replaced needle")
    assert updated["status"] == "resolved"
    assert updated["technician_note"] == "Replaced needle"
