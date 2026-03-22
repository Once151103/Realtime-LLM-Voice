"""
Desktop Voice Client — PyAudio-based real-time audio I/O.

Captures microphone input and plays back synthesized audio from the server.
"""

import asyncio
import logging
import struct

import numpy as np
import pyaudio

from config import settings

logger = logging.getLogger(__name__)

# PyAudio format constants
FORMAT = pyaudio.paInt16
CHANNELS = 1


class AudioIO:
    """
    Low-level audio I/O using PyAudio.

    Handles:
    - Microphone capture (PCM 16-bit, 16kHz, mono)
    - Speaker playback (PCM 16-bit, 22050Hz, mono)
    - Buffer management with underrun protection
    """

    def __init__(self) -> None:
        self._pa: pyaudio.PyAudio | None = None
        self._input_stream: pyaudio.Stream | None = None
        self._output_stream: pyaudio.Stream | None = None
        self._input_callback = None
        self._playback_buffer: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)
        self._is_running = False

        # Calculate frame sizes
        self._input_frames = int(
            settings.audio.sample_rate_input * settings.audio.chunk_size_ms / 1000
        )
        self._output_frames = int(
            settings.audio.sample_rate_output * settings.audio.chunk_size_ms / 1000
        )

    def initialize(self) -> None:
        """Initialize PyAudio and open input/output streams."""
        self._pa = pyaudio.PyAudio()

        # Log available devices
        info = self._pa.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount", 0)
        logger.info("Audio devices:")
        for i in range(num_devices):
            dev = self._pa.get_device_info_by_host_api_device_index(0, i)
            direction = ""
            if dev.get("maxInputChannels", 0) > 0:
                direction += "IN "
            if dev.get("maxOutputChannels", 0) > 0:
                direction += "OUT"
            logger.info("  [%d] %s (%s)", i, dev["name"], direction.strip())

        # Open input stream (microphone)
        self._input_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=settings.audio.sample_rate_input,
            input=True,
            frames_per_buffer=self._input_frames,
            stream_callback=self._on_input_audio,
        )

        # Open output stream (speakers)
        self._output_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=settings.audio.sample_rate_output,
            output=True,
            frames_per_buffer=self._output_frames,
            stream_callback=self._on_output_audio,
        )

        logger.info("✅ Audio I/O initialized (input: %dHz, output: %dHz)",
                     settings.audio.sample_rate_input, settings.audio.sample_rate_output)

    def set_input_callback(self, callback) -> None:
        """Set callback for incoming microphone audio."""
        self._input_callback = callback

    def _on_input_audio(self, in_data, frame_count, time_info, status):
        """PyAudio input callback — forwards mic audio."""
        if self._input_callback and in_data:
            # Schedule the callback on the event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        self._input_callback(in_data),
                    )
            except RuntimeError:
                pass

        return (None, pyaudio.paContinue)

    def _on_output_audio(self, in_data, frame_count, time_info, status):
        """PyAudio output callback — plays audio from the buffer."""
        bytes_needed = frame_count * CHANNELS * settings.audio.sample_width

        try:
            data = self._playback_buffer.get_nowait()
            # Pad or truncate to match expected frame size
            if len(data) < bytes_needed:
                data = data + b"\x00" * (bytes_needed - len(data))
            elif len(data) > bytes_needed:
                # Put the extra back
                extra = data[bytes_needed:]
                data = data[:bytes_needed]
                try:
                    self._playback_buffer.put_nowait(extra)
                except asyncio.QueueFull:
                    pass
        except (asyncio.QueueEmpty, Exception):
            # Silence on underrun
            data = b"\x00" * bytes_needed

        return (data, pyaudio.paContinue)

    async def play_audio(self, audio_data: bytes) -> None:
        """Queue audio for playback."""
        try:
            # Split into frame-sized chunks for smoother playback
            chunk_size = self._output_frames * CHANNELS * settings.audio.sample_width
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self._playback_buffer.put(chunk)
        except asyncio.QueueFull:
            logger.warning("Playback buffer full, dropping audio")

    def clear_playback(self) -> None:
        """Clear the playback buffer (for interruption)."""
        while not self._playback_buffer.empty():
            try:
                self._playback_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

    def start(self) -> None:
        """Start audio streams."""
        self._is_running = True
        if self._input_stream:
            self._input_stream.start_stream()
        if self._output_stream:
            self._output_stream.start_stream()

    def stop(self) -> None:
        """Stop and close audio streams."""
        self._is_running = False
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("Audio I/O stopped")
