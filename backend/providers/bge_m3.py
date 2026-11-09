"""Optional local BGE-M3 provider; its heavy dependency remains lazy."""
import asyncio

from .base import EmbeddingProvider
from ..config import settings
from ..embeddings import _bge_embed


class BGEM3EmbeddingProvider(EmbeddingProvider):
    name = "bge-m3"
    model_name = settings.embed_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(_bge_embed, texts)
