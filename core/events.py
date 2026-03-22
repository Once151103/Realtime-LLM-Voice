"""
Event system — async event bus and interrupt signaling.

Provides a lightweight pub/sub event bus using asyncio for
coordinating between pipeline services.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Events emitted across the voice pipeline."""

    # VAD events
    SPEECH_START = auto()
    SPEECH_END = auto()

    # STT events
    TRANSCRIPT_PARTIAL = auto()
    TRANSCRIPT_FINAL = auto()

    # LLM events
    LLM_TOKEN = auto()
    LLM_COMPLETE = auto()

    # Chunker events
    CHUNK_READY = auto()

    # TTS events
    AUDIO_CHUNK_READY = auto()
    TTS_COMPLETE = auto()

    # Control events
    INTERRUPT = auto()
    STATE_CHANGE = auto()

    # System events
    ERROR = auto()
    SESSION_START = auto()
    SESSION_END = auto()


@dataclass
class Event:
    """A pipeline event with type and arbitrary payload."""

    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Lightweight async event bus for pipeline coordination.

    Supports pub/sub with multiple handlers per event type.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._interrupt = asyncio.Event()
        self._lock = asyncio.Lock()

    def on(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("Registered handler for %s: %s", event_type, handler.__name__)

    def off(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler for an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    async def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        handlers = self._handlers.get(event.type, [])
        if not handlers:
            return

        # Special handling for INTERRUPT — set the shared flag immediately
        if event.type == EventType.INTERRUPT:
            self._interrupt.set()
            logger.info("🛑 INTERRUPT signal emitted")

        # Fire all handlers concurrently
        tasks = [asyncio.create_task(h(event)) for h in handlers]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Handler %s failed for %s: %s",
                        handlers[i].__name__,
                        event.type,
                        result,
                    )

    async def emit_fire_and_forget(self, event: Event) -> None:
        """Emit an event without waiting for handlers to complete."""
        handlers = self._handlers.get(event.type, [])
        if event.type == EventType.INTERRUPT:
            self._interrupt.set()
        for handler in handlers:
            asyncio.create_task(handler(event))

    @property
    def interrupt_flag(self) -> asyncio.Event:
        """Shared interrupt flag — services should check this frequently."""
        return self._interrupt

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag after handling."""
        self._interrupt.clear()
        logger.debug("Interrupt flag cleared")

    def is_interrupted(self) -> bool:
        """Check if interrupt is active."""
        return self._interrupt.is_set()
