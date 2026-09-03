"""
Unified LLM provider interface.
Supports Google Gemini and OpenAI, selected at startup via config.
Same provider is used for both embeddings and text generation.
Fully non-blocking asynchronous implementation.
"""

from __future__ import annotations

import json
import asyncio
import logging
from abc import ABC, abstractmethod

from backend.config import Settings, LLMProvider

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def generate_structured(self, prompt: str, system_prompt: str = "") -> dict:
        """Generate structured JSON output from a prompt."""
        ...

    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider (fully asynchronous with high-quota models & fallback)."""

    def __init__(self, api_key: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = "gemini-3.5-flash-lite"
        self._fallback_models = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.7-flash"]
        self._embed_model = "gemini-embedding-001"
        self._dim = 768

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        results = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            max_retries = 3
            last_err = None
            for attempt in range(max_retries):
                try:
                    response = await self._client.aio.models.embed_content(
                        model=self._embed_model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self._dim,
                        ),
                    )
                    for emb in response.embeddings:
                        results.append(emb.values)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                        wait_time = 4 * (attempt + 1)
                        logger.warning(
                            "Gemini embed rate limit (429) encountered. Retrying in %ds (attempt %d/%d)...",
                            wait_time, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise e
            if last_err:
                raise last_err
        return results

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=0.1,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_structured(self, prompt: str, system_prompt: str = "") -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            response_mime_type="application/json",
            temperature=0.1,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        text = response.text or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini structured output: %s", text[:200])
            return {"entities": [], "relationships": []}

    def embedding_dimension(self) -> int:
        return self._dim


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider (fully asynchronous)."""

    def __init__(self, api_key: str):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = "gpt-4o"
        self._embed_model = "text-embedding-3-small"
        self._dim = 1536

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        batch_size = 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(
                model=self._embed_model,
                input=batch,
            )
            for item in response.data:
                results.append(item.embedding)
        return results

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    async def generate_structured(self, prompt: str, system_prompt: str = "") -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse OpenAI structured output: %s", text[:200])
            return {"entities": [], "relationships": []}

    def embedding_dimension(self) -> int:
        return self._dim


def create_llm_provider(settings: Settings) -> BaseLLMProvider:
    """Factory function to create the correct LLM provider."""
    if settings.llm_provider == LLMProvider.GEMINI:
        return GeminiProvider(api_key=settings.gemini_api_key)
    elif settings.llm_provider == LLMProvider.OPENAI:
        return OpenAIProvider(api_key=settings.openai_api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
