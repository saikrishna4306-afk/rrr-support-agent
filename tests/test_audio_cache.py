from backend import audio_cache
from backend.providers.base import SynthesizedAudio


def test_exact_audio_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_cache.settings, "audio_cache_path", str(tmp_path))
    source = SynthesizedAudio(b"wav-data", "audio/wav", "test-voice")
    audio_cache.save("answer", "en", "mock", source)

    assert audio_cache.load("answer", "test-voice", "en", "mock") == source
    assert audio_cache.load("different", "test-voice", "en", "mock") is None
