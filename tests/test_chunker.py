"""Tests for the sentence chunker."""

import pytest

from core.events import EventBus, EventType
from services.chunker import ChunkerService


class TestChunker:
    """Test text chunking at sentence boundaries."""

    @pytest.fixture
    def chunker_with_events(self):
        """Create a chunker and capture emitted events."""
        event_bus = EventBus()
        chunks = []

        async def capture_chunk(event):
            chunks.append(event.data["text"])

        event_bus.on(EventType.CHUNK_READY, capture_chunk)
        chunker = ChunkerService(event_bus)
        return chunker, chunks

    @pytest.mark.asyncio
    async def test_period_boundary(self, chunker_with_events):
        """Should dispatch on period."""
        chunker, chunks = chunker_with_events

        tokens = ["Hola", " señor", ".", " ¿Cómo", " está", "?"]
        for token in tokens:
            await chunker.feed_token(token)
        await chunker.flush()

        assert len(chunks) >= 1
        assert "Hola señor." in chunks[0]

    @pytest.mark.asyncio
    async def test_comma_boundary(self, chunker_with_events):
        """Should dispatch on comma after enough words."""
        chunker, chunks = chunker_with_events

        tokens = ["Primero", " necesitamos", " revisar", " los", " datos", ",",
                   " luego", " continuamos", "."]
        for token in tokens:
            await chunker.feed_token(token)
        await chunker.flush()

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_max_words_force_dispatch(self, chunker_with_events):
        """Should force dispatch at max words even without boundary."""
        chunker, chunks = chunker_with_events

        # Feed many words without punctuation
        for i in range(20):
            await chunker.feed_token(f" palabra{i}")
        await chunker.flush()

        assert len(chunks) >= 2  # Should have forced at least one mid-dispatch

    @pytest.mark.asyncio
    async def test_clear_on_interrupt(self, chunker_with_events):
        """Should clear buffer on interrupt."""
        chunker, chunks = chunker_with_events

        await chunker.feed_token("Esto")
        await chunker.feed_token(" es")
        await chunker.clear()
        await chunker.flush()

        assert len(chunks) == 0  # Buffer was cleared

    @pytest.mark.asyncio
    async def test_flush_remaining(self, chunker_with_events):
        """Flush should dispatch remaining buffer."""
        chunker, chunks = chunker_with_events

        await chunker.feed_token("Texto")
        await chunker.feed_token(" restante")
        await chunker.flush()

        assert len(chunks) == 1
        assert "Texto restante" in chunks[0]
