"""
TTS Streaming Worker — Piper TTS integration.

Synthesizes text chunks into audio using Piper's ONNX-based neural TTS.
Supports voice cloning via fine-tuned models.
"""

import asyncio
import io
import logging
import wave

import numpy as np

from config import settings
from core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class TTSService:
    """
    Piper TTS synthesis worker.

    - Loads ONNX model once and keeps it warm
    - Synthesizes text chunks asynchronously
    - Outputs PCM 16-bit audio at 22050Hz
    - Cancelable via interrupt flag
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._voice = None
        self._synthesize_fn = None
        self._pending_chunks = 0

    async def initialize(self) -> None:
        """Load the Piper TTS model and prewarm."""
        logger.info("Loading Piper TTS model: %s", settings.piper.model_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        # Prewarm — run a dummy synthesis to initialize GPU context
        logger.info("Prewarming TTS...")
        await loop.run_in_executor(None, self._prewarm)
        logger.info("✅ Piper TTS ready")

    def _load_model(self) -> None:
        """Load Piper voice model (runs in thread)."""
        from piper import PiperVoice

        self._voice = PiperVoice.load(
            settings.piper.model_path,
            config_path=settings.piper.config_path,
            use_cuda=True,
        )

    def _prewarm(self) -> None:
        """Run a dummy synthesis to warm up the model."""
        if self._voice is None:
            return
        # Synthesize a short phrase to initialize all compute paths
        dummy_audio = io.BytesIO()
        with wave.open(dummy_audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(settings.audio.sample_rate_output)
            self._voice.synthesize(
                "Inicializando.",
                wav,
                speaker_id=settings.piper.speaker_id,
                length_scale=settings.piper.length_scale,
                noise_scale=settings.piper.noise_scale,
                noise_w=settings.piper.noise_w,
            )

    async def synthesize(
        self,
        text: str,
        interrupt_flag: asyncio.Event | None = None,
    ) -> None:
        """
        Synthesize text to audio and emit AUDIO_CHUNK_READY.

        Runs synthesis in a thread pool to avoid blocking the event loop.
        """
        if not self._voice or not text.strip():
            return

        # Check interrupt before starting
        if interrupt_flag and interrupt_flag.is_set():
            return

        self._pending_chunks += 1

        loop = asyncio.get_event_loop()
        try:
            audio_data = await loop.run_in_executor(None, self._do_synthesize, text)

            # Check interrupt after synthesis
            if interrupt_flag and interrupt_flag.is_set():
                return

            if audio_data:
                await self.event_bus.emit(
                    Event(EventType.AUDIO_CHUNK_READY, {"audio": audio_data})
                )
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            await self.event_bus.emit(Event(EventType.ERROR, {"error": f"TTS: {e}"}))
        finally:
            self._pending_chunks -= 1
            if self._pending_chunks <= 0:
                self._pending_chunks = 0
                await self.event_bus.emit(Event(EventType.TTS_COMPLETE))

    def _do_synthesize(self, text: str) -> bytes:
        """Run Piper synthesis (runs in thread)."""
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(settings.audio.sample_rate_output)
            self._voice.synthesize(
                text,
                wav,
                speaker_id=settings.piper.speaker_id,
                length_scale=settings.piper.length_scale,
                noise_scale=settings.piper.noise_scale,
                noise_w=settings.piper.noise_w,
            )

        # Extract raw PCM from WAV
        audio_buffer.seek(0)
        with wave.open(audio_buffer, "rb") as wav:
            pcm_data = wav.readframes(wav.getnframes())

        return pcm_data

    async def cancel(self) -> None:
        """Cancel pending synthesis (interrupt flag handles in-progress work)."""
        self._pending_chunks = 0
        logger.debug("TTS cancelled")
