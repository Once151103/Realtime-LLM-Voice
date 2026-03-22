"""
Speech-to-Text — faster-whisper streaming transcription.

Accumulates audio from the microphone and transcribes using
CTranslate2-accelerated Whisper.
"""

import asyncio
import io
import logging
import time

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from config import settings
from core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class STTService:
    """
    faster-whisper based speech-to-text service.

    Accumulates audio frames while user is speaking, then transcribes
    on SPEECH_END. Designed for low latency with beam_size=1.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._model: WhisperModel | None = None
        self._audio_buffer: list[bytes] = []
        self._is_listening = False

    async def initialize(self) -> None:
        """Load the Whisper model."""
        logger.info(
            "Loading Whisper model '%s' on %s (%s)...",
            settings.whisper.model_size,
            settings.whisper.device,
            settings.whisper.compute_type,
        )
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info("✅ Whisper STT ready")

    def _load_model(self) -> WhisperModel:
        """Load faster-whisper model (runs in thread)."""
        return WhisperModel(
            settings.whisper.model_size,
            device=settings.whisper.device,
            compute_type=settings.whisper.compute_type,
        )

    async def start_listening(self) -> None:
        """Start accumulating audio frames."""
        if not self._is_listening:
            self._audio_buffer.clear()
            self._is_listening = True

    async def feed_audio(self, audio_data: bytes) -> None:
        """Feed a raw audio frame (PCM 16-bit, 16kHz) to the buffer."""
        if self._is_listening:
            self._audio_buffer.append(audio_data)

    async def finalize(self) -> None:
        """
        Stop listening and transcribe the accumulated audio.

        Emits TRANSCRIPT_FINAL with the recognized text.
        """
        if not self._is_listening or not self._audio_buffer:
            self._is_listening = False
            return

        self._is_listening = False
        audio_data = b"".join(self._audio_buffer)
        self._audio_buffer.clear()

        if len(audio_data) < 3200:  # Less than 100ms of audio at 16kHz 16-bit
            return

        # Transcribe in thread pool
        loop = asyncio.get_event_loop()
        start = time.time()
        text = await loop.run_in_executor(None, self._transcribe, audio_data)
        elapsed = (time.time() - start) * 1000

        if text.strip():
            logger.info("📝 Transcribed in %.0f ms: %s", elapsed, text)
            await self.event_bus.emit(
                Event(EventType.TRANSCRIPT_FINAL, {"text": text})
            )

    def _transcribe(self, audio_data: bytes) -> str:
        """Run Whisper transcription (runs in thread)."""
        if self._model is None:
            return ""

        # Convert PCM 16-bit to float32 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe
        segments, info = self._model.transcribe(
            audio_np,
            beam_size=settings.whisper.beam_size,
            language=settings.whisper.language,
            vad_filter=settings.whisper.vad_filter,
            without_timestamps=True,
        )

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    async def cancel(self) -> None:
        """Cancel current listening session."""
        self._is_listening = False
        self._audio_buffer.clear()
