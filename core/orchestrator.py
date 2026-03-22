"""
Voice Orchestrator — Central brain that coordinates all pipeline services.

Manages the state machine, routes audio/text streams, handles interruptions,
and dispatches work to VAD, STT, LLM, Chunker, TTS, and Audio Queue.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from core.events import Event, EventBus, EventType
from core.state import AgentState, StateError

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Tracks the state of an ongoing conversation."""

    session_id: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    current_transcript: str = ""
    current_response: str = ""
    turn_count: int = 0
    started_at: float = field(default_factory=time.time)

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.turn_count += 1

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def get_history(self, max_turns: int = 20) -> list[dict[str, str]]:
        """Get recent conversation history for LLM context."""
        return self.messages[-(max_turns * 2):]


class VoiceOrchestrator:
    """
    Central coordinator for the realtime voice pipeline.

    Flow:
        Mic Audio → VAD → STT → LLM → Chunker → TTS → Audio Queue → Speaker

    The orchestrator:
    - Maintains the agent state machine (IDLE → LISTENING → THINKING → SPEAKING)
    - Wires event handlers between services
    - Handles barge-in interruption
    - Measures latency metrics
    """

    def __init__(
        self,
        event_bus: EventBus,
        vad_service=None,
        stt_service=None,
        llm_service=None,
        chunker_service=None,
        tts_service=None,
        audio_queue_service=None,
    ) -> None:
        self.event_bus = event_bus
        self.vad = vad_service
        self.stt = stt_service
        self.llm = llm_service
        self.chunker = chunker_service
        self.tts = tts_service
        self.audio_queue = audio_queue_service

        self._state = AgentState.IDLE
        self._context = ConversationContext()
        self._state_lock = asyncio.Lock()

        # Latency tracking
        self._speech_end_time: float | None = None
        self._first_audio_time: float | None = None

        self._register_handlers()

    def _register_handlers(self) -> None:
        """Wire up event handlers for the pipeline."""
        self.event_bus.on(EventType.SPEECH_START, self._on_speech_start)
        self.event_bus.on(EventType.SPEECH_END, self._on_speech_end)
        self.event_bus.on(EventType.TRANSCRIPT_FINAL, self._on_transcript_final)
        self.event_bus.on(EventType.LLM_TOKEN, self._on_llm_token)
        self.event_bus.on(EventType.LLM_COMPLETE, self._on_llm_complete)
        self.event_bus.on(EventType.CHUNK_READY, self._on_chunk_ready)
        self.event_bus.on(EventType.AUDIO_CHUNK_READY, self._on_audio_chunk_ready)
        self.event_bus.on(EventType.TTS_COMPLETE, self._on_tts_complete)

    async def _set_state(self, new_state: AgentState) -> None:
        """Transition to a new state with validation."""
        async with self._state_lock:
            if not self._state.can_transition_to(new_state):
                raise StateError(self._state, new_state)
            old_state = self._state
            self._state = new_state
            logger.info("State: %s → %s", old_state, new_state)
            await self.event_bus.emit(
                Event(EventType.STATE_CHANGE, {"from": old_state, "to": new_state})
            )

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def context(self) -> ConversationContext:
        return self._context

    # ── Lifecycle ───────────────────────────────────────────────────

    async def start_session(self, session_id: str = "default") -> None:
        """Start a new conversation session."""
        self._context = ConversationContext(session_id=session_id)
        await self._set_state(AgentState.LISTENING)
        await self.event_bus.emit(Event(EventType.SESSION_START, {"session_id": session_id}))
        logger.info("🎙️ Session started: %s", session_id)

    async def stop_session(self) -> None:
        """End the current conversation session."""
        await self._handle_interrupt()
        if self._state != AgentState.IDLE:
            # Force back to IDLE regardless of current state
            async with self._state_lock:
                self._state = AgentState.IDLE
        await self.event_bus.emit(Event(EventType.SESSION_END))
        logger.info("Session ended")

    # ── Audio Input ─────────────────────────────────────────────────

    async def handle_audio_frame(self, audio_data: bytes) -> None:
        """
        Main entry point — process an incoming audio frame from the microphone.

        Routes to VAD for speech detection. If in SPEAKING state and speech
        is detected, triggers barge-in interruption.
        """
        if self.vad:
            await self.vad.process_frame(audio_data)

    # ── Event Handlers ──────────────────────────────────────────────

    async def _on_speech_start(self, event: Event) -> None:
        """User started speaking."""
        if self._state == AgentState.SPEAKING:
            # Barge-in! User is interrupting the assistant
            logger.info("🛑 Barge-in detected! Interrupting...")
            await self._handle_interrupt()
            await self._set_state(AgentState.LISTENING)
        elif self._state == AgentState.IDLE:
            await self._set_state(AgentState.LISTENING)

        # Start/continue STT processing
        if self.stt:
            await self.stt.start_listening()

    async def _on_speech_end(self, event: Event) -> None:
        """User stopped speaking — time to process."""
        self._speech_end_time = time.time()
        if self._state == AgentState.LISTENING and self.stt:
            await self.stt.finalize()

    async def _on_transcript_final(self, event: Event) -> None:
        """Got final transcript — send to LLM."""
        transcript: str = event.data.get("text", "")
        if not transcript.strip():
            return

        logger.info("📝 User: %s", transcript)
        self._context.add_user_message(transcript)
        self._context.current_response = ""

        await self._set_state(AgentState.THINKING)

        # Start LLM generation
        if self.llm:
            asyncio.create_task(
                self.llm.generate(
                    messages=self._context.get_history(),
                    interrupt_flag=self.event_bus.interrupt_flag,
                )
            )

    async def _on_llm_token(self, event: Event) -> None:
        """LLM produced a token — feed to chunker."""
        if self.event_bus.is_interrupted():
            return

        token: str = event.data.get("token", "")
        self._context.current_response += token

        if self.chunker:
            await self.chunker.feed_token(token)

    async def _on_llm_complete(self, event: Event) -> None:
        """LLM finished generating — flush remaining text."""
        if self.event_bus.is_interrupted():
            return

        self._context.add_assistant_message(self._context.current_response)
        logger.info("🤖 Assistant: %s", self._context.current_response)

        if self.chunker:
            await self.chunker.flush()

    async def _on_chunk_ready(self, event: Event) -> None:
        """Text chunk ready — send to TTS for synthesis."""
        if self.event_bus.is_interrupted():
            return

        chunk_text: str = event.data.get("text", "")
        if not chunk_text.strip():
            return

        # Transition to SPEAKING on first chunk
        if self._state == AgentState.THINKING:
            await self._set_state(AgentState.SPEAKING)

        if self.tts:
            asyncio.create_task(
                self.tts.synthesize(
                    text=chunk_text,
                    interrupt_flag=self.event_bus.interrupt_flag,
                )
            )

    async def _on_audio_chunk_ready(self, event: Event) -> None:
        """Audio synthesized — push to playback queue."""
        if self.event_bus.is_interrupted():
            return

        audio_data: bytes = event.data.get("audio", b"")
        if not audio_data:
            return

        # Track TTFA (Time To First Audio)
        if self._first_audio_time is None and self._speech_end_time is not None:
            self._first_audio_time = time.time()
            ttfa = (self._first_audio_time - self._speech_end_time) * 1000
            logger.info("⚡ TTFA: %.0f ms", ttfa)

        if self.audio_queue:
            await self.audio_queue.enqueue(audio_data)

    async def _on_tts_complete(self, event: Event) -> None:
        """TTS finished all chunks — transition back to listening after playback."""
        # Wait for audio queue to drain, then go back to LISTENING
        if self.audio_queue:
            await self.audio_queue.wait_until_empty()

        if self._state == AgentState.SPEAKING:
            self._first_audio_time = None
            self._speech_end_time = None
            await self._set_state(AgentState.LISTENING)

    # ── Interruption ────────────────────────────────────────────────

    async def _handle_interrupt(self) -> None:
        """Handle barge-in — stop everything and clear buffers."""
        interrupt_start = time.time()

        # Signal all services to stop
        await self.event_bus.emit_fire_and_forget(Event(EventType.INTERRUPT))

        # Abort LLM generation
        if self.llm:
            await self.llm.abort()

        # Clear TTS queue
        if self.tts:
            await self.tts.cancel()

        # Clear and fade-out audio
        if self.audio_queue:
            await self.audio_queue.clear_with_fadeout()

        # Clear chunker buffer
        if self.chunker:
            await self.chunker.clear()

        # Clear interrupt flag for next cycle
        self.event_bus.clear_interrupt()

        interrupt_ms = (time.time() - interrupt_start) * 1000
        logger.info("🛑 Interrupt handled in %.0f ms (target: <150 ms)", interrupt_ms)
