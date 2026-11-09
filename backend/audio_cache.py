"""Exact local cache for generated demo TTS audio."""
import hashlib
import json
from pathlib import Path

from .config import settings
from .providers.base import SynthesizedAudio


def _key(text: str, voice: str, language: str, provider: str) -> str:
    value = "\x1f".join((text, voice, language, provider)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load(text: str, voice: str, language: str, provider: str) -> SynthesizedAudio | None:
    """Return exactly matching cached bytes, or None when this speech is new."""
    stem = Path(settings.audio_cache_path) / _key(text, voice, language, provider)
    metadata = stem.with_suffix(".json")
    audio = stem.with_suffix(".audio")
    if not (metadata.exists() and audio.exists()):
        return None
    data = json.loads(metadata.read_text(encoding="utf-8"))
    return SynthesizedAudio(audio.read_bytes(), data["mime_type"], data["voice"])


def save(text: str, language: str, provider: str, audio: SynthesizedAudio) -> None:
    """Write exact result metadata and bytes after a successful synthesis."""
    stem = Path(settings.audio_cache_path) / _key(text, audio.voice, language, provider)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".audio").write_bytes(audio.data)
    stem.with_suffix(".json").write_text(json.dumps({"mime_type": audio.mime_type, "voice": audio.voice}),
                                          encoding="utf-8")
