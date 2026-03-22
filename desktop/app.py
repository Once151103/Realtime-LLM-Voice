"""
Desktop Voice App — main entry point for the PyAudio desktop client.

Connects directly to the voice pipeline (no WebSocket needed for local use).
Captures mic → processes through the full pipeline → plays back through speakers.
"""

import asyncio
import logging
import signal
import sys

from rich.console import Console

from config import settings
from core.events import EventBus
from core.orchestrator import VoiceOrchestrator
from desktop.audio_io import AudioIO
from desktop.ui import TerminalUI
from services.audio_queue import AudioQueueService
from services.chunker import ChunkerService
from services.llm import LLMService
from services.stt import STTService
from services.tts import TTSService
from services.vad import VADService

logger = logging.getLogger(__name__)
console = Console()


async def run_desktop() -> None:
    """
    Run the desktop voice agent.

    Initializes all services, wires the pipeline, and starts the main loop.
    Audio flows: Mic → VAD → STT → LLM → Chunker → TTS → AudioQueue → Speaker
    """
    console.print("\n[bold cyan]╔══════════════════════════════════════╗[/]")
    console.print("[bold cyan]║     JARVIS — Voice AI Agent          ║[/]")
    console.print("[bold cyan]║     Realtime Desktop Client          ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════╝[/]\n")

    # Create event bus
    event_bus = EventBus()

    # Initialize services
    console.print("[dim]Initializing services...[/]")

    vad = VADService(event_bus)
    stt = STTService(event_bus)
    llm = LLMService(event_bus)
    chunker = ChunkerService(event_bus)
    tts = TTSService(event_bus)
    audio_queue = AudioQueueService(event_bus)

    # Initialize models (this takes a few seconds)
    await vad.initialize()
    await stt.initialize()
    await llm.initialize()
    await tts.initialize()

    # Create orchestrator
    orchestrator = VoiceOrchestrator(
        event_bus=event_bus,
        vad_service=vad,
        stt_service=stt,
        llm_service=llm,
        chunker_service=chunker,
        tts_service=tts,
        audio_queue_service=audio_queue,
    )

    # Initialize audio I/O
    audio_io = AudioIO()
    audio_io.initialize()

    # Wire mic input to the orchestrator
    async def on_mic_audio(audio_data: bytes) -> None:
        await orchestrator.handle_audio_frame(audio_data)
        if stt._is_listening:
            await stt.feed_audio(audio_data)

    audio_io.set_input_callback(on_mic_audio)

    # Wire audio queue output to speakers
    audio_queue.set_output_callback(audio_io.play_audio)

    # Initialize terminal UI
    ui = TerminalUI(event_bus)

    # Start everything
    await audio_queue.start()
    audio_io.start()
    await ui.start()
    await orchestrator.start_session("desktop")

    console.print("[bold green]✅ All systems online. Speak to begin.[/]\n")

    # Keep running until interrupted
    stop_event = asyncio.Event()

    def handle_signal():
        stop_event.set()

    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, handle_signal)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: handle_signal())
    except Exception:
        signal.signal(signal.SIGINT, lambda s, f: handle_signal())

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # Cleanup
    console.print("\n[dim]Shutting down...[/]")
    await ui.stop()
    await orchestrator.stop_session()
    await audio_queue.stop()
    audio_io.stop()
    await llm.shutdown()
    console.print("[bold]Goodbye, señor. 👋[/]\n")


def main():
    """Entry point for the desktop app."""
    logging.basicConfig(
        level=getattr(logging, settings.server.log_level),
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(run_desktop())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
