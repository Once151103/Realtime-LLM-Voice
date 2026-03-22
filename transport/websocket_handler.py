"""
WebSocket handler — manages bidirectional audio streaming connections.
"""

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from core.events import Event, EventBus, EventType
from core.orchestrator import VoiceOrchestrator

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    Manages WebSocket connections for real-time audio streaming.

    Protocol:
    - Binary frames: raw PCM audio (client → server: 16kHz 16-bit,
      server → client: 22050Hz 16-bit)
    - Text frames: JSON control messages (state changes, transcripts)
    """

    def __init__(self, orchestrator: VoiceOrchestrator, event_bus: EventBus) -> None:
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self._active_connections: set[WebSocket] = set()

        # Register for events we need to forward to clients
        self.event_bus.on(EventType.STATE_CHANGE, self._broadcast_state)
        self.event_bus.on(EventType.TRANSCRIPT_FINAL, self._broadcast_transcript)
        self.event_bus.on(EventType.AUDIO_CHUNK_READY, self._broadcast_audio)

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection lifecycle."""
        await websocket.accept()
        self._active_connections.add(websocket)
        logger.info("🔌 Client connected (%d total)", len(self._active_connections))

        # Set audio output callback to send to this client
        if self.orchestrator.audio_queue:
            self.orchestrator.audio_queue.set_output_callback(
                lambda audio: self._send_audio(websocket, audio)
            )

        try:
            # Start session
            await self.orchestrator.start_session()

            # Receive audio loop
            while True:
                data = await websocket.receive()

                if "bytes" in data:
                    # Binary frame = audio data
                    await self.orchestrator.handle_audio_frame(data["bytes"])

                    # Also feed to STT if listening
                    if self.orchestrator.stt and self.orchestrator.stt._is_listening:
                        await self.orchestrator.stt.feed_audio(data["bytes"])

                elif "text" in data:
                    # Text frame = control message
                    await self._handle_control_message(websocket, data["text"])

        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e)
        finally:
            self._active_connections.discard(websocket)
            await self.orchestrator.stop_session()

    async def _handle_control_message(self, ws: WebSocket, message: str) -> None:
        """Process a JSON control message from the client."""
        try:
            msg = json.loads(message)
            action = msg.get("action")

            if action == "start":
                await self.orchestrator.start_session(msg.get("session_id", "default"))
            elif action == "stop":
                await self.orchestrator.stop_session()
            elif action == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

        except json.JSONDecodeError:
            logger.warning("Invalid JSON message: %s", message[:100])

    async def _send_audio(self, ws: WebSocket, audio_data: bytes) -> None:
        """Send audio data to a specific client."""
        try:
            await ws.send_bytes(audio_data)
        except Exception:
            pass  # Connection might have closed

    async def _broadcast_state(self, event: Event) -> None:
        """Broadcast state change to all connected clients."""
        msg = json.dumps({
            "type": "state_change",
            "from": str(event.data["from"]),
            "to": str(event.data["to"]),
        })
        await self._broadcast_text(msg)

    async def _broadcast_transcript(self, event: Event) -> None:
        """Broadcast transcript to all connected clients."""
        msg = json.dumps({
            "type": "transcript",
            "text": event.data.get("text", ""),
        })
        await self._broadcast_text(msg)

    async def _broadcast_audio(self, event: Event) -> None:
        """Broadcast audio chunk to all connected clients."""
        audio = event.data.get("audio", b"")
        if audio:
            for ws in list(self._active_connections):
                try:
                    await ws.send_bytes(audio)
                except Exception:
                    self._active_connections.discard(ws)

    async def _broadcast_text(self, message: str) -> None:
        """Broadcast a text message to all connected clients."""
        for ws in list(self._active_connections):
            try:
                await ws.send_text(message)
            except Exception:
                self._active_connections.discard(ws)
