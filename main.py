"""
Realtime Voice AI Agent — FastAPI + WebSocket Server.

Main entry point for the server-based deployment.
The desktop app (desktop/app.py) runs the pipeline directly without this server.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from config import settings
from core.events import EventBus
from core.orchestrator import VoiceOrchestrator
from services.audio_queue import AudioQueueService
from services.chunker import ChunkerService
from services.llm import LLMService
from services.stt import STTService
from services.tts import TTSService
from services.vad import VADService
from transport.websocket_handler import WebSocketHandler

logger = logging.getLogger(__name__)

# Global references (initialized in lifespan)
event_bus: EventBus | None = None
orchestrator: VoiceOrchestrator | None = None
ws_handler: WebSocketHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and shutdown services."""
    global event_bus, orchestrator, ws_handler

    logging.basicConfig(
        level=getattr(logging, settings.server.log_level),
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("🚀 Starting Realtime Voice AI Agent server...")

    # Create event bus and services
    event_bus = EventBus()

    vad = VADService(event_bus)
    stt = STTService(event_bus)
    llm = LLMService(event_bus)
    chunker = ChunkerService(event_bus)
    tts = TTSService(event_bus)
    audio_queue = AudioQueueService(event_bus)

    # Initialize all services (loads models)
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

    # Start audio queue playback
    await audio_queue.start()

    # Create WebSocket handler
    ws_handler = WebSocketHandler(orchestrator, event_bus)

    logger.info("✅ All services initialized. Server ready.")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await audio_queue.stop()
    await llm.shutdown()
    logger.info("Server stopped.")


app = FastAPI(
    title="Realtime Voice AI Agent",
    description="Low-latency voice assistant server (Jarvis-style)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    state = orchestrator.state.name if orchestrator else "NOT_INITIALIZED"
    return JSONResponse({
        "status": "ok",
        "agent_state": state,
        "model": settings.ollama.model,
    })


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice streaming."""
    if ws_handler:
        await ws_handler.handle_connection(websocket)
    else:
        await websocket.close(code=1013, reason="Server not ready")


def run():
    """Run the server."""
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower(),
    )


if __name__ == "__main__":
    run()
