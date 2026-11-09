"""Deterministic offline providers used by tests, CI, and local demos."""
import io
import wave

from .base import EmbeddingProvider, LLMProvider, STTProvider, TTSProvider, SynthesizedAudio, Transcription
from ..embeddings import _hash_embed


class MockEmbeddingProvider(EmbeddingProvider):
    """Uses the existing stable hash vectors without network or model downloads."""

    name = "hash"
    model_name = "hash-512-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return _hash_embed(texts)


class MockSTTProvider(STTProvider):
    """Stable fake speech recognition so tests never need microphone audio or a key."""

    name = "mock"

    async def transcribe(self, audio: bytes) -> Transcription:
        return Transcription("My needle keeps breaking", "en")


class MockTTSProvider(TTSProvider):
    """Produces a valid, silent 16 kHz WAV that browsers can decode in tests."""

    name = "mock"

    async def synthesize(self, text: str, language: str) -> SynthesizedAudio:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 1600)  # 100 ms of deterministic silence
        return SynthesizedAudio(output.getvalue(), "audio/wav", "mock-silent")


class MockLLMProvider(LLMProvider):
    """Returns a marker; RAG uses its deterministic offline answer instead."""

    name = "mock"
    model_name = "mock-v1"

    async def generate(self, system: str, question: str) -> str:
        return "HANDOFF"
