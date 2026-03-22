"""Tests for the audio queue."""

import asyncio

import numpy as np
import pytest

from core.events import EventBus
from services.audio_queue import AudioQueueService


class TestAudioQueue:
    """Test audio queue with crossfade and interruption."""

    @pytest.fixture
    def audio_queue(self):
        event_bus = EventBus()
        return AudioQueueService(event_bus)

    def _make_pcm(self, duration_ms: int = 100, frequency: float = 440.0) -> bytes:
        """Generate a test sine wave as PCM 16-bit."""
        sample_rate = 22050
        num_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t) * 16000
        return signal.astype(np.int16).tobytes()

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, audio_queue):
        """Basic enqueue and dequeue."""
        audio_queue._is_playing = True
        test_audio = self._make_pcm()
        await audio_queue.enqueue(test_audio)
        assert not audio_queue._queue.empty()

    @pytest.mark.asyncio
    async def test_clear_with_fadeout(self, audio_queue):
        """Clear should empty the queue."""
        audio_queue._is_playing = True
        for _ in range(5):
            await audio_queue.enqueue(self._make_pcm())

        assert not audio_queue._queue.empty()
        await audio_queue.clear_with_fadeout()
        assert audio_queue._queue.empty()

    def test_process_chunk_normalization(self, audio_queue):
        """Chunks should be volume-normalized."""
        quiet_audio = (np.ones(1000, dtype=np.int16) * 100).tobytes()
        processed = audio_queue._process_chunk(quiet_audio)
        result = np.frombuffer(processed, dtype=np.int16)
        # Should be louder after normalization
        assert np.max(np.abs(result)) > 100

    def test_interrupt_fadeout(self, audio_queue):
        """Interrupt fadeout should silence the end."""
        test_audio = self._make_pcm(duration_ms=200)
        faded = audio_queue._apply_interrupt_fadeout(test_audio)
        result = np.frombuffer(faded, dtype=np.int16)
        # End should be near silence
        assert np.max(np.abs(result[-100:])) < 100
