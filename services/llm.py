"""
LLM Streaming Engine — Ollama-based with fallback support.

Streams tokens from the LLM and emits them as events for the chunker.
"""

import asyncio
import logging

import httpx

from config import settings
from core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class LLMService:
    """
    Ollama LLM client with streaming token generation.

    Supports:
    - Token-by-token streaming
    - Abort controller via asyncio.Event
    - Automatic fallback to secondary model
    - System prompt from SOUL.md
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._client: httpx.AsyncClient | None = None
        self._system_prompt: str = ""
        self._current_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the HTTP client and load system prompt."""
        self._client = httpx.AsyncClient(
            base_url=settings.ollama.base_url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=5.0, pool=5.0),
        )
        self._system_prompt = settings.ollama.load_system_prompt()

        # Verify Ollama is reachable
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info("✅ Ollama connected. Available models: %s", models)
        except Exception as e:
            logger.warning("⚠️ Ollama not reachable at %s: %s", settings.ollama.base_url, e)

    async def generate(
        self,
        messages: list[dict[str, str]],
        interrupt_flag: asyncio.Event | None = None,
        model: str | None = None,
    ) -> None:
        """
        Stream a response from the LLM.

        Emits LLM_TOKEN events for each token and LLM_COMPLETE when done.
        Abortable via the interrupt_flag.
        """
        if not self._client:
            logger.error("LLM client not initialized")
            return

        target_model = model or settings.ollama.model

        # Build the messages list with system prompt
        full_messages = [
            {"role": "system", "content": self._system_prompt},
            *messages,
        ]

        payload = {
            "model": target_model,
            "messages": full_messages,
            "stream": True,
            "options": {
                "temperature": settings.ollama.temperature,
                "num_predict": settings.ollama.max_tokens,
            },
        }

        try:
            await self._stream_response(payload, interrupt_flag, target_model)
        except Exception as e:
            if target_model != settings.ollama.fallback_model:
                logger.warning(
                    "Primary model '%s' failed: %s. Trying fallback '%s'...",
                    target_model, e, settings.ollama.fallback_model,
                )
                payload["model"] = settings.ollama.fallback_model
                await self._stream_response(payload, interrupt_flag, settings.ollama.fallback_model)
            else:
                logger.error("LLM generation failed: %s", e)
                await self.event_bus.emit(Event(EventType.ERROR, {"error": str(e)}))

    async def _stream_response(
        self,
        payload: dict,
        interrupt_flag: asyncio.Event | None,
        model_name: str,
    ) -> None:
        """Stream tokens from the Ollama chat endpoint."""
        logger.debug("Generating with model: %s", model_name)

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                # Check interrupt
                if interrupt_flag and interrupt_flag.is_set():
                    logger.info("LLM generation aborted by interrupt")
                    return

                if not line.strip():
                    continue

                try:
                    import json
                    data = json.loads(line)
                except Exception:
                    continue

                # Extract token from Ollama's streaming format
                message = data.get("message", {})
                token = message.get("content", "")

                if token:
                    await self.event_bus.emit(Event(EventType.LLM_TOKEN, {"token": token}))

                # Check if generation is done
                if data.get("done", False):
                    break

        # Signal completion
        await self.event_bus.emit(Event(EventType.LLM_COMPLETE))

    async def abort(self) -> None:
        """Abort current generation."""
        # The interrupt flag check in _stream_response handles this
        logger.debug("LLM abort requested")

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
