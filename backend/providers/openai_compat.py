"""OpenAI-compatible LLM adapter configured entirely through environment settings."""
import asyncio

from openai import OpenAI

from .base import LLMProvider
from ..config import settings


class OpenAICompatLLMProvider(LLMProvider):
    name = "openai_compat"
    model_name = settings.llm_model

    def __init__(self) -> None:
        if not (settings.llm_api_key and settings.llm_model):
            raise ValueError("LLM_API_KEY and LLM_MODEL are required for openai_compat")
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                             timeout=settings.provider_timeout_seconds)

    async def generate(self, system: str, question: str) -> str:
        def call() -> str:
            response = self.client.chat.completions.create(
                model=self.model_name, temperature=0.2, max_tokens=500,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
            )
            return (response.choices[0].message.content or "").strip()
        return await asyncio.to_thread(call)
