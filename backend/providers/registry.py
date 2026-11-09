"""Provider registry with timeout, retry/backoff, and optional fallback selection."""
from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import EmbeddingProvider, LLMProvider, STTProvider, TTSProvider
from .bge_m3 import BGEM3EmbeddingProvider
from .edge_tts import EdgeTTSProvider
from .gemini import GeminiEmbeddingProvider, GeminiLLMProvider, GeminiSTTProvider
from .gnani import GnaniSTTProvider, GnaniTTSProvider
from .mock import MockEmbeddingProvider, MockLLMProvider, MockSTTProvider, MockTTSProvider
from .openai_compat import OpenAICompatLLMProvider
from ..config import settings

T = TypeVar("T")


def _retrying(call: Callable[[], Awaitable[T]]) -> Callable[[], Awaitable[T]]:
    """Retry transient provider failures; each concrete call also gets a timeout."""
    return retry(
        retry=retry_if_exception_type(Exception), stop=stop_after_attempt(settings.provider_max_retries + 1),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2), reraise=True,
    )(call)


async def call_provider(call: Callable[[], Awaitable[T]]) -> T:
    """Apply retry/backoff and timeout without nesting event loops."""
    import asyncio
    wrapped = _retrying(call)
    return await asyncio.wait_for(wrapped(), timeout=settings.provider_timeout_seconds)


def get_embedding_provider() -> EmbeddingProvider:
    selected = settings.active_embed_provider
    if selected == "hash":
        return MockEmbeddingProvider()
    if selected == "gemini":
        return GeminiEmbeddingProvider()
    if selected == "bge-m3":
        return BGEM3EmbeddingProvider()
    raise ValueError(f"Unsupported EMBED_PROVIDER: {selected}")


def get_llm_provider() -> LLMProvider:
    selected = settings.llm_provider.lower()
    if selected == "mock":
        return MockLLMProvider()
    if selected == "gemini":
        return GeminiLLMProvider()
    if selected == "openai_compat":
        return OpenAICompatLLMProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {selected}")


def get_stt_provider(selected: str | None = None) -> STTProvider:
    selected = (selected or settings.stt_provider).lower()
    if selected == "mock":
        return MockSTTProvider()
    if selected == "gemini":
        return GeminiSTTProvider()
    if selected == "gnani":
        return GnaniSTTProvider()
    raise ValueError(f"Unsupported STT_PROVIDER: {selected}")


def get_tts_provider(selected: str | None = None) -> TTSProvider:
    selected = (selected or settings.tts_provider).lower()
    if selected == "mock":
        return MockTTSProvider()
    if selected == "edge":
        return EdgeTTSProvider()
    if selected == "gnani":
        return GnaniTTSProvider()
    if selected == "gemini":
        raise ValueError("Gemini TTS is deferred until its PCM response format is smoke-tested")
    raise ValueError(f"Unsupported TTS_PROVIDER: {selected}")

