"""
Voice Activity Detection — Silero VAD integration.

Detects user speech to trigger STT and barge-in interruption.
"""

import asyncio
import logging

import numpy as np
import torch

from config import settings
from core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class VADService:
    """
    Silero VAD wrapper for real-time speech detection.

    Processes audio frames and emits SPEECH_START / SPEECH_END events.
    When the agent is SPEAKING and user speech is detected, triggers barge-in.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._model = None
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

        self._threshold = settings.vad.threshold
        self._min_speech_frames = max(
            1, settings.vad.min_speech_ms // (settings.vad.window_size_samples * 1000 // settings.vad.sample_rate)
        )
        self._min_silence_frames = max(
            1, settings.vad.min_silence_ms // (settings.vad.window_size_samples * 1000 // settings.vad.sample_rate)
        )

    async def initialize(self) -> None:
        """Load Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info("✅ Silero VAD ready")

    def _load_model(self):
        """Load Silero VAD from torch.hub (runs in thread)."""
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        return model

    async def process_frame(self, audio_data: bytes) -> None:
        """
        Process a raw audio frame (PCM 16-bit, 16kHz).

        Detects speech onset and offset, emitting events accordingly.
        """
        if self._model is None:
            return

        # Convert bytes to float32 tensor
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np)

        # Run VAD inference
        loop = asyncio.get_event_loop()
        confidence = await loop.run_in_executor(
            None, self._infer, audio_tensor
        )

        is_speech = confidence > self._threshold

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1

            if not self._is_speaking and self._speech_frames >= self._min_speech_frames:
                self._is_speaking = True
                logger.debug("🎤 Speech detected (confidence: %.2f)", confidence)
                await self.event_bus.emit(Event(EventType.SPEECH_START))
        else:
            self._speech_frames = 0
            self._silence_frames += 1

            if self._is_speaking and self._silence_frames >= self._min_silence_frames:
                self._is_speaking = False
                logger.debug("🔇 Speech ended")
                await self.event_bus.emit(Event(EventType.SPEECH_END))

    def _infer(self, audio_tensor: torch.Tensor) -> float:
        """Run VAD inference (runs in thread)."""
        with torch.no_grad():
            confidence = self._model(audio_tensor, settings.vad.sample_rate).item()
        return confidence

    async def reset(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        if self._model is not None:
            self._model.reset_states()
