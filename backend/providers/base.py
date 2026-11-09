"""Small async contracts used by the support pipeline."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcription:
    """Transcript plus the language reported or inferred by the STT provider."""
    text: str
    language: str | None = None


@dataclass(frozen=True)
class SynthesizedAudio:
    """Audio bytes and the MIME type the browser needs to play them."""
    data: bytes
    mime_type: str
    voice: str


class LLMProvider(ABC):
    """Creates a grounded answer from a prepared system prompt and question."""

    name: str
    model_name: str

    @abstractmethod
    async def generate(self, system: str, question: str) -> str: ...


class EmbeddingProvider(ABC):
    """Converts text to same-length dense vectors."""

    name: str
    model_name: str

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VisionProvider(ABC):
    """Reserved for Milestone D's image diagnosis provider."""

    @abstractmethod
    async def diagnose(self, image: bytes) -> dict: ...


class STTProvider(ABC):
    """Reserved for Milestone B's speech-to-text provider."""

    @abstractmethod
    async def transcribe(self, audio: bytes) -> Transcription: ...


class TTSProvider(ABC):
    """Reserved for Milestone B's text-to-speech provider."""

    @abstractmethod
    async def synthesize(self, text: str, language: str) -> SynthesizedAudio: ...
