import asyncio

from backend.providers.mock import MockEmbeddingProvider, MockLLMProvider
from backend.providers.registry import call_provider, get_embedding_provider, get_llm_provider


def test_mock_providers_are_deterministic_and_offline():
    async def scenario():
        vectors = await MockEmbeddingProvider().embed(["needle breaks", "needle breaks"])
        assert vectors[0] == vectors[1] and len(vectors[0]) == 512
        assert await MockLLMProvider().generate("ignored", "ignored") == "HANDOFF"
    asyncio.run(scenario())


def test_registry_selects_mock_configuration():
    async def scenario():
        assert get_embedding_provider().name == "hash"
        assert get_llm_provider().name == "mock"
        assert await call_provider(lambda: MockLLMProvider().generate("", "")) == "HANDOFF"
    asyncio.run(scenario())
