"""Result caching — in-memory or Redis-backed."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from prismai.utils.config import get_settings
from prismai.utils.logger import get_logger

logger = get_logger(__name__)


class InMemoryCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl: int = 3600, max_size: int = 1000) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._store:
            ts, value = self._store[key]
            if time.time() - ts < self.ttl:
                self._hits += 1
                return value
            del self._store[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        # Evict oldest if at capacity
        if len(self._store) >= self.max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        self._store.clear()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "type": "memory",
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
        }


class RedisCache:
    """Redis-backed cache."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600) -> None:
        self.ttl = ttl
        self.redis_url = redis_url
        self._client = None
        self._hits = 0
        self._misses = 0

    async def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(self.redis_url, decode_responses=True)
                await self._client.ping()
                logger.info("redis_connected", url=self.redis_url)
            except Exception as exc:
                logger.warning("redis_connection_failed", error=str(exc))
                raise
        return self._client

    async def get(self, key: str) -> Any | None:
        try:
            client = await self._ensure_client()
            data = await client.get(f"prismai:{key}")
            if data:
                self._hits += 1
                return json.loads(data)
            self._misses += 1
            return None
        except Exception:
            self._misses += 1
            return None

    async def set(self, key: str, value: Any) -> None:
        try:
            client = await self._ensure_client()
            await client.setex(
                f"prismai:{key}", self.ttl, json.dumps(value, default=str)
            )
        except Exception as exc:
            logger.warning("redis_set_failed", error=str(exc))

    async def clear(self) -> None:
        try:
            client = await self._ensure_client()
            keys = await client.keys("prismai:*")
            if keys:
                await client.delete(*keys)
        except Exception as exc:
            logger.warning("redis_clear_failed", error=str(exc))


class AnalysisCache:
    """Unified cache interface for analysis results."""

    def __init__(self) -> None:
        settings = get_settings()
        self.ttl = settings.cache_ttl

        # Try Redis first, fall back to in-memory
        try:
            self._backend: Any = RedisCache(settings.redis_url, self.ttl)
            self._use_redis = True
        except Exception:
            self._backend = InMemoryCache(self.ttl)
            self._use_redis = False
            logger.info("using_in_memory_cache")

    @staticmethod
    def make_key(content: str, modality: str, analyses: list[str]) -> str:
        """Generate a cache key from analysis parameters."""
        raw = f"{content[:500]}:{modality}:{','.join(sorted(analyses))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def get(self, key: str) -> Any | None:
        if self._use_redis and isinstance(self._backend, RedisCache):
            return await self._backend.get(key)
        return self._backend.get(key)

    async def set(self, key: str, value: Any) -> None:
        if self._use_redis and isinstance(self._backend, RedisCache):
            await self._backend.set(key, value)
        else:
            self._backend.set(key, value)
