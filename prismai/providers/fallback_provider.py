"""Provider cascade — MiMo → OpenRouter → DeepSeek."""

from __future__ import annotations

import time
from typing import Any

import httpx

from prismai.utils.config import get_settings
from prismai.utils.logger import get_logger

logger = get_logger(__name__)


class FallbackProvider:
    """Cascading LLM provider with automatic fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._providers = self._build_provider_chain()
        self._client: httpx.AsyncClient | None = None
        self._usage: dict[str, int] = {"mimo": 0, "openrouter": 0, "deepseek": 0}

    def _build_provider_chain(self) -> list[dict[str, Any]]:
        """Build ordered list of providers to try."""
        providers = []
        if self.settings.mimo_api_key and self.settings.mimo_api_key != "demo_key":
            providers.append({
                "name": "mimo",
                "base_url": self.settings.mimo_base_url,
                "api_key": self.settings.mimo_api_key,
                "model": self.settings.mimo_model,
            })
        if self.settings.openrouter_api_key:
            providers.append({
                "name": "openrouter",
                "base_url": self.settings.openrouter_base_url,
                "api_key": self.settings.openrouter_api_key,
                "model": "google/gemini-flash-1.5",
            })
        if self.settings.deepseek_api_key:
            providers.append({
                "name": "deepseek",
                "base_url": self.settings.deepseek_base_url,
                "api_key": self.settings.deepseek_api_key,
                "model": "deepseek-chat",
            })
        # Always have MiMo as fallback even with demo key
        if not providers:
            providers.append({
                "name": "mimo",
                "base_url": self.settings.mimo_base_url,
                "api_key": self.settings.mimo_api_key,
                "model": self.settings.mimo_model,
            })
        return providers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Try each provider in order until one succeeds."""
        last_error: Exception | None = None

        for provider in self._providers:
            try:
                result = await self._call_provider(
                    provider, messages, temperature, max_tokens, **kwargs
                )
                self._usage[provider["name"]] = self._usage.get(provider["name"], 0) + 1
                return result
            except Exception as exc:
                logger.warning(
                    "provider_failed",
                    provider=provider["name"],
                    error=str(exc),
                )
                last_error = exc
                continue

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        )

    async def _call_provider(
        self,
        provider: dict[str, Any],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call a specific provider's OpenAI-compatible endpoint."""
        client = await self._get_client()
        payload = {
            "model": provider["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        headers = {
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        resp = await client.post(
            f"{provider['base_url']}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "provider_success",
            provider=provider["name"],
            latency_ms=round(elapsed_ms, 1),
        )
        return data

    async def complete(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> str:
        """Simple completion with automatic fallback."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        data = await self.chat(messages, **kwargs)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected response: {exc}") from exc

    @property
    def usage_stats(self) -> dict[str, int]:
        return dict(self._usage)
