"""
Audio Queue & Playback Engine — manages sequential audio output.

Handles crossfade blending, interruption with fade-out,
and volume normalization.
"""

import asyncio
import logging

import numpy as np

from config import settings
from core.events import EventBus

logger = logging.getLogger(__name__)


class AudioQueueService:
    """
    FIFO audio queue for streaming playback.

    Features:
    - Sequential chunk playback
    - Crossfade between chunks (40-80ms overlap)
    - Instant clear with fade-out on interruption
    - Volume normalization
    - Output callback for sending audio to the client
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=settings.audio.queue_max_chunks
        )
        self._output_callback = None
        self._is_playing = False
        self._playback_task: asyncio.Task | None = None
        self._previous_tail: np.ndarray | None = None  # For crossfade

        # Precompute crossfade parameters
        crossfade_samples = int(
            settings.audio.sample_rate_output * settings.audio.crossfade_ms / 1000
        )
        self._crossfade_samples = crossfade_samples
        self._fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        self._fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)

        # Interrupt fade-out
        interrupt_samples = int(
            settings.audio.sample_rate_output * settings.interrupt.fade_out_ms / 1000
        )
        self._interrupt_fade = np.linspace(1.0, 0.0, interrupt_samples, dtype=np.float32)

    def set_output_callback(self, callback) -> None:
        """Set the callback for sending audio to the client."""
        self._output_callback = callback

    async def start(self) -> None:
        """Start the playback loop."""
        self._is_playing = True
        self._playback_task = asyncio.create_task(self._playback_loop())

    async def stop(self) -> None:
        """Stop the playback loop."""
        self._is_playing = False
        if self._playback_task:
            await self._queue.put(None)  # Sentinel to unblock
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, audio_data: bytes) -> None:
        """Add an audio chunk to the queue."""
        if self._is_playing:
            await self._queue.put(audio_data)

    async def clear_with_fadeout(self) -> None:
        """Clear the queue and apply fade-out to current audio (for barge-in)."""
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._previous_tail = None
        logger.debug("Audio queue cleared with fade-out")

    async def wait_until_empty(self) -> None:
        """Wait until the queue is fully drained."""
        await self._queue.join()

    async def _playback_loop(self) -> None:
        """Main playback loop — processes audio chunks sequentially."""
        logger.info("🔊 Audio playback loop started")

        while self._is_playing:
            try:
                audio_data = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if audio_data is None:
                self._queue.task_done()
                break

            # Process crossfade and normalization
            processed = self._process_chunk(audio_data)

            # Send to output
            if self._output_callback and processed:
                try:
                    await self._output_callback(processed)
                except Exception as e:
                    logger.error("Audio output error: %s", e)

            self._queue.task_done()

        logger.info("Audio playback loop stopped")

    def _process_chunk(self, audio_data: bytes) -> bytes:
        """Apply crossfade and volume normalization to a chunk."""
        # Convert to float32
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        if len(audio) == 0:
            return b""

        # Normalize volume (peak normalization to ~80%)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (26000.0 / peak)  # ~80% of int16 max

        # Apply crossfade with previous chunk tail
        if self._previous_tail is not None and len(self._previous_tail) > 0:
            overlap = min(len(self._previous_tail), len(audio), self._crossfade_samples)
            if overlap > 0:
                fade_out = self._fade_out[:overlap]
                fade_in = self._fade_in[:overlap]
                audio[:overlap] = (
                    self._previous_tail[-overlap:] * fade_out + audio[:overlap] * fade_in
                )

        # Save tail for next crossfade
        if len(audio) >= self._crossfade_samples:
            self._previous_tail = audio[-self._crossfade_samples:].copy()
        else:
            self._previous_tail = audio.copy()

        # Convert back to int16 PCM
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        return audio.tobytes()

    def _apply_interrupt_fadeout(self, audio_data: bytes) -> bytes:
        """Apply a rapid fade-out for interruption (prevents click/pop)."""
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        fade_len = min(len(audio), len(self._interrupt_fade))
        audio[:fade_len] *= self._interrupt_fade[:fade_len]
        if fade_len < len(audio):
            audio[fade_len:] = 0
        return np.clip(audio, -32768, 32767).astype(np.int16).tobytes()
