"""
Response Cache — pre-synthesized common replies for zero-latency responses.
"""

import asyncio
import hashlib
import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Caches frequently used responses (both text and pre-synthesized audio).

    Reduces latency to ~0ms for common phrases like greetings,
    confirmations, and acknowledgements.
    """

    # Common phrases to pre-synthesize at startup
    COMMON_PHRASES = {
        "greeting": [
            "Buenos días, señor.",
            "Buenas tardes, señor.",
            "¿En qué puedo ayudarle?",
        ],
        "acknowledgement": [
            "Entendido, señor.",
            "Entendido.",
            "Comprendido.",
            "Por supuesto.",
        ],
        "thinking": [
            "Un momento, por favor.",
            "Procesando su solicitud.",
            "Permítame un momento.",
        ],
        "confirmation": [
            "Listo.",
            "Hecho.",
            "Completado.",
        ],
        "farewell": [
            "Hasta luego, señor.",
            "Que tenga un buen día.",
        ],
    }

    def __init__(self) -> None:
        self._text_cache: dict[str, str] = {}
        self._audio_cache: dict[str, bytes] = {}
        self._redis = None

    async def initialize(self, redis_url: str | None = None) -> None:
        """Initialize cache, optionally with Redis backend."""
        if redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(redis_url)
                await self._redis.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning("Redis not available, using in-memory cache: %s", e)
                self._redis = None

    async def pre_synthesize(self, tts_service) -> None:
        """Pre-synthesize common phrases at startup."""
        logger.info("Pre-synthesizing %d common phrases...",
                     sum(len(v) for v in self.COMMON_PHRASES.values()))

        for category, phrases in self.COMMON_PHRASES.items():
            for phrase in phrases:
                key = self._make_key(phrase)
                try:
                    audio = await asyncio.get_event_loop().run_in_executor(
                        None, tts_service._do_synthesize, phrase
                    )
                    self._audio_cache[key] = audio
                    self._text_cache[key] = phrase
                except Exception as e:
                    logger.warning("Failed to pre-synthesize '%s': %s", phrase, e)

        logger.info("✅ Pre-synthesized %d phrases", len(self._audio_cache))

    def get_audio(self, text: str) -> bytes | None:
        """Get cached audio for a text phrase, or None if not cached."""
        key = self._make_key(text.strip())
        return self._audio_cache.get(key)

    def has(self, text: str) -> bool:
        """Check if a phrase is cached."""
        return self._make_key(text.strip()) in self._audio_cache

    def _make_key(self, text: str) -> str:
        """Create a cache key from text."""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def shutdown(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
