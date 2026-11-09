import asyncio
import json

import httpx

from backend.config import settings
from backend.providers.gnani import GnaniSTTProvider, GnaniTTSProvider


def test_gnani_stt_request_payload(monkeypatch):
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        captured["body"] = request.content
        return httpx.Response(200, json={"success": True, "transcript": "నమస్కారం", "detected_language_code": "te-IN"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(settings, "gnani_api_key", "private-test-key")
    monkeypatch.setattr(settings, "gnani_base_url", "https://gnani.test")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    result = asyncio.run(GnaniSTTProvider().transcribe(b"RIFF fake wav"))

    assert result.text == "నమస్కారం" and result.language == "te"
    assert captured["headers"]["x-api-key-id"] == "private-test-key"
    assert b'name="audio_file"' in captured["body"] and b'name="model"' in captured["body"]


def test_gnani_tts_json_payload(monkeypatch):
    captured = {}

    def handler(request):
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, content=b"RIFF generated wav")

    original_client = httpx.AsyncClient
    monkeypatch.setattr(settings, "gnani_api_key", "private-test-key")
    monkeypatch.setattr(settings, "gnani_base_url", "https://gnani.test")
    monkeypatch.setattr(settings, "gnani_tts_voice_en", "en-test-voice")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    result = asyncio.run(GnaniTTSProvider().synthesize("Help me", "en"))

    assert result.data == b"RIFF generated wav" and result.voice == "en-test-voice"
    assert captured["json"]["text"] == "Help me"
    assert captured["json"]["audio_config"]["container"] == "wav"
