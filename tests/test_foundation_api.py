from fastapi.testclient import TestClient

from backend.main import app


def test_chat_keeps_contract_and_adds_timings():
    with TestClient(app) as client:
        response = client.post("/chat", json={
            "session_id": "foundation-session", "message": "needle keeps breaking",
        })
    assert response.status_code == 200
    body = response.json()
    assert {"answer", "language", "confidence", "handoff", "ticket_id", "sources"} <= set(body)
    assert body["timings"]["chat"] >= 0
