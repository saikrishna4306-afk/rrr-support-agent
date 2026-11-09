"""Edge TTS adapter. Voice names are entirely configured through .env."""
import asyncio
import os
import tempfile
from pathlib import Path

from .base import STTProvider, TTSProvider, SynthesizedAudio
from ..config import settings


def edge_voice(language: str) -> str:
    """Return the configured language voice, falling back to English for unknown text."""
    return {"te": settings.edge_tts_voice_te, "hi": settings.edge_tts_voice_hi}.get(
        language, settings.edge_tts_voice_en)


class EdgeTTSProvider(TTSProvider):
    name = "edge"

    async def synthesize(self, text: str, language: str) -> SynthesizedAudio:
        voice = edge_voice(language)

        def create() -> bytes:
            import edge_tts
            fd, filename = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                asyncio.run(edge_tts.Communicate(text, voice).save(filename))
                return Path(filename).read_bytes()
            finally:
                Path(filename).unlink(missing_ok=True)
        # edge-tts exposes an async API; isolate its own event loop in a worker thread.
        data = await asyncio.to_thread(create)
        return SynthesizedAudio(data, "audio/mpeg", voice)
