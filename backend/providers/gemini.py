"""Gemini LLM and embedding adapters. google-genai imports only when selected."""
import asyncio

from .base import EmbeddingProvider, LLMProvider, STTProvider, Transcription
from ..config import settings


class GeminiLLMProvider(LLMProvider):
    name = "gemini"
    model_name = settings.gemini_llm_model

    def __init__(self) -> None:
        if not (settings.gemini_api_key and self.model_name):
            raise ValueError("GEMINI_API_KEY and GEMINI_LLM_MODEL are required for Gemini")
        from google import genai
        self.client = genai.Client(api_key=settings.gemini_api_key)

    async def generate(self, system: str, question: str) -> str:
        def call() -> str:
            response = self.client.models.generate_content(
                model=self.model_name, contents=question,
                config={"system_instruction": system, "temperature": 0.2, "max_output_tokens": 500},
            )
            return (response.text or "").strip()
        return await asyncio.to_thread(call)


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"
    model_name = settings.gemini_embed_model

    def __init__(self) -> None:
        if not (settings.gemini_api_key and self.model_name):
            raise ValueError("GEMINI_API_KEY and GEMINI_EMBED_MODEL are required for Gemini embeddings")
        from google import genai
        self.client = genai.Client(api_key=settings.gemini_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def call() -> list[list[float]]:
            response = self.client.models.embed_content(model=self.model_name, contents=texts)
            return [[float(value) for value in item.values] for item in response.embeddings]
        return await asyncio.to_thread(call)


class GeminiSTTProvider(STTProvider):
    """Gemini inline-audio transcription; model and key come only from settings."""
    name = "gemini"

    def __init__(self) -> None:
        if not (settings.gemini_api_key and settings.gemini_stt_model):
            raise ValueError("GEMINI_API_KEY and GEMINI_STT_MODEL are required for Gemini STT")
        from google import genai
        self.client = genai.Client(api_key=settings.gemini_api_key)

    async def transcribe(self, audio: bytes) -> Transcription:
        def call() -> str:
            from google.genai import types
            prompt = ("Transcribe exactly, keep the original script (Telugu, Devanagari or Latin), "
                      "output only the transcript")
            response = self.client.models.generate_content(
                model=settings.gemini_stt_model,
                contents=[types.Part.from_bytes(data=audio, mime_type="audio/wav"), prompt],
            )
            return (response.text or "").strip()
        return Transcription(await asyncio.to_thread(call))

