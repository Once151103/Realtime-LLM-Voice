"""
Sentence / Phoneme Chunker — splits LLM output into synthesis-ready chunks.

Reduces perceived latency by sending text to TTS early, at natural
sentence boundaries.
"""

import logging
import re

from core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)

# Boundaries that trigger a chunk dispatch
_HARD_BOUNDARIES = re.compile(r'[.!?。！？]')
_SOFT_BOUNDARIES = re.compile(r'[,;:，；：]')
_CONJUNCTION_BOUNDARIES = re.compile(
    r'\b(y|pero|sin embargo|además|también|'
    r'and|but|however|also|therefore|so|then)\b',
    re.IGNORECASE,
)


class ChunkerService:
    """
    Adaptive text chunker for streaming TTS.

    Strategy:
    - Short chunks (4-6 words) for greetings / first response
    - Medium chunks (8-15 words) for normal explanation
    - Dispatches on punctuation, conjunctions, or max buffer length

    Emits CHUNK_READY events with text ready for synthesis.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._buffer = ""
        self._word_count = 0
        self._is_first_chunk = True
        self._min_words = 4
        self._max_words = 15

    async def feed_token(self, token: str) -> None:
        """
        Feed a token from the LLM stream.

        Checks for sentence boundaries and dispatches chunks when ready.
        """
        self._buffer += token

        # Count words in current buffer
        self._word_count = len(self._buffer.split())

        # Determine minimum words based on context
        min_words = self._min_words if self._is_first_chunk else 6

        # Check for hard boundaries (end of sentence)
        if _HARD_BOUNDARIES.search(token) and self._word_count >= min_words:
            await self._dispatch()
            return

        # Check for soft boundaries (comma, semicolon)
        if _SOFT_BOUNDARIES.search(token) and self._word_count >= min_words:
            await self._dispatch()
            return

        # Check for conjunction boundaries
        stripped = token.strip().lower()
        if _CONJUNCTION_BOUNDARIES.match(stripped) and self._word_count >= min_words:
            # Dispatch everything before the conjunction
            parts = self._buffer.rsplit(stripped, 1)
            if len(parts) == 2 and parts[0].strip():
                chunk = parts[0].strip()
                self._buffer = stripped + parts[1]
                self._word_count = len(self._buffer.split())
                await self._emit_chunk(chunk)
                return

        # Force dispatch at max words to avoid long silence
        if self._word_count >= self._max_words:
            await self._dispatch()

    async def flush(self) -> None:
        """Flush any remaining text in the buffer."""
        if self._buffer.strip():
            await self._dispatch()
        self._is_first_chunk = True

    async def clear(self) -> None:
        """Clear the buffer (on interruption)."""
        self._buffer = ""
        self._word_count = 0
        self._is_first_chunk = True

    async def _dispatch(self) -> None:
        """Dispatch the current buffer as a chunk."""
        chunk = self._buffer.strip()
        self._buffer = ""
        self._word_count = 0

        if chunk:
            await self._emit_chunk(chunk)

    async def _emit_chunk(self, text: str) -> None:
        """Emit a CHUNK_READY event."""
        logger.debug("📦 Chunk [%d words]: %s", len(text.split()), text)
        await self.event_bus.emit(Event(EventType.CHUNK_READY, {"text": text}))
        self._is_first_chunk = False
