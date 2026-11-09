"""Gnani Vachana REST STT/TTS adapters using documented httpx requests only."""
import httpx

from .base import STTProvider, TTSProvider, SynthesizedAudio, Transcription
from ..config import settings


def _headers() -> dict[str, str]:
    if not settings.gnani_api_key:
        raise ValueError("GNANI_API_KEY is required when a Gnani provider is selected")
    return {"X-API-Key-ID": settings.gnani_api_key}


def _voice(language: str) -> str:
    voice = {"te": settings.gnani_tts_voice_te, "hi": settings.gnani_tts_voice_hi}.get(
        language, settings.gnani_tts_voice_en)
    if not voice:
        raise ValueError("Set the matching GNANI_TTS_VOICE_TE, _HI, or _EN in .env")
    return voice


class GnaniSTTProvider(STTProvider):
    name = "gnani"

    async def transcribe(self, audio: bytes) -> Transcription:
        files = {"audio_file": ("recording.wav", audio, "audio/wav")}
        data = {"model": settings.gnani_stt_model, "language_code": "indic-auto"}
        async with httpx.AsyncClient(timeout=settings.provider_timeout_seconds) as client:
            response = await client.post(f"{settings.gnani_base_url}/stt/v3", headers=_headers(),
                                         files=files, data=data)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success") or not payload.get("transcript"):
            raise ValueError("Gnani STT response did not contain a successful transcript")
        code = payload.get("detected_language_code", "")
        language = {"te": "te", "hi": "hi", "en": "en"}.get(code[:2])
        return Transcription(str(payload["transcript"]), language)


class GnaniTTSProvider(TTSProvider):
    name = "gnani"

    async def synthesize(self, text: str, language: str) -> SynthesizedAudio:
        voice = _voice(language)
        body = {"text": text, "voice": voice, "model": settings.gnani_tts_model,
                "language": {"te": "te-IN", "hi": "hi-IN"}.get(language, "en-IN"), "speed": 1.0,
                "audio_config": {"sample_rate": 48000, "num_channels": 1, "sample_width": 2,
                                 "encoding": "linear_pcm", "container": "wav"}}
        async with httpx.AsyncClient(timeout=settings.provider_timeout_seconds) as client:
            response = await client.post(f"{settings.gnani_base_url}/api/v1/tts/inference",
                                         headers={**_headers(), "Content-Type": "application/json"}, json=body)
        response.raise_for_status()
        return SynthesizedAudio(response.content, "audio/wav", voice)
