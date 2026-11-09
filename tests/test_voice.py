import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend.ingest import ingest
from backend.main import app
from backend.rag import reset_retriever


@pytest.fixture(scope="module")
def indexed():
    ingest()
    reset_retriever()


def wav(seconds: float = 0.1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\0\0" * int(seconds * 16000))
    return output.getvalue()


def test_voice_returns_transcript_answer_and_wav(indexed):
    with TestClient(app) as client:
        response = client.post("/voice", data={"session_id": "voice-test-1"},
                               files={"audio": ("recording.wav", wav(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "My needle keeps breaking"
    assert body["answer_text"]
    assert body["audio_mime_type"] == "audio/wav"
    assert body["audio"]


def test_voice_rejects_non_wav(indexed):
    with TestClient(app) as client:
        response = client.post("/voice", data={"session_id": "voice-test-1"},
                               files={"audio": ("recording.mp3", b"not-a-wav", "audio/mpeg")})
    assert response.status_code == 415
