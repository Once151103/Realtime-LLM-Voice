"""
Desktop UI — Rich terminal interface for the voice agent.

Displays agent state, transcription, and response in real-time.
"""

import asyncio
import logging

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from core.events import Event, EventBus, EventType
from core.state import AgentState

logger = logging.getLogger(__name__)

# State display mapping
STATE_DISPLAY = {
    AgentState.IDLE: ("💤", "IDLE", "dim"),
    AgentState.LISTENING: ("🎧", "LISTENING", "green"),
    AgentState.THINKING: ("🧠", "THINKING", "yellow"),
    AgentState.SPEAKING: ("🗣️", "SPEAKING", "blue"),
}


class TerminalUI:
    """
    Rich-based terminal UI for the voice agent.

    Shows:
    - Current agent state with emoji indicator
    - User transcription in real-time
    - Agent response text
    - Latency metrics
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.console = Console()
        self._state = AgentState.IDLE
        self._user_text = ""
        self._agent_text = ""
        self._ttfa = None
        self._live: Live | None = None

        # Register event handlers
        self.event_bus.on(EventType.STATE_CHANGE, self._on_state_change)
        self.event_bus.on(EventType.TRANSCRIPT_FINAL, self._on_transcript)
        self.event_bus.on(EventType.LLM_TOKEN, self._on_llm_token)
        self.event_bus.on(EventType.LLM_COMPLETE, self._on_llm_complete)

    def _render(self) -> Layout:
        """Render the current UI state."""
        layout = Layout()

        emoji, label, style = STATE_DISPLAY.get(
            self._state, ("❓", "UNKNOWN", "red")
        )

        # Header
        header = Panel(
            Text(f"  {emoji}  {label}", style=f"bold {style}", justify="center"),
            title="[bold]JARVIS[/bold] Voice Agent",
            border_style=style,
            height=3,
        )

        # User speech
        user_panel = Panel(
            Text(self._user_text or "...", style="green"),
            title="👤 User",
            border_style="green",
            height=5,
        )

        # Agent response
        agent_style = "blue" if self._state == AgentState.SPEAKING else "dim blue"
        cursor = "▌" if self._state in (AgentState.THINKING, AgentState.SPEAKING) else ""
        agent_panel = Panel(
            Text((self._agent_text or "...") + cursor, style=agent_style),
            title="🤖 Jarvis",
            border_style="blue",
            height=8,
        )

        # Metrics
        ttfa_str = f"{self._ttfa:.0f} ms" if self._ttfa else "—"
        metrics = Text(f"  TTFA: {ttfa_str}  │  Ctrl+C to exit", style="dim")

        layout.split_column(
            Layout(header, size=3),
            Layout(user_panel, size=5),
            Layout(agent_panel, size=8),
            Layout(metrics, size=1),
        )

        return layout

    async def start(self) -> None:
        """Start the live terminal display."""
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=10,
            screen=True,
        )
        self._live.start()
        logger.info("Terminal UI started")

    async def stop(self) -> None:
        """Stop the terminal display."""
        if self._live:
            self._live.stop()

    def _update(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._render())

    async def _on_state_change(self, event: Event) -> None:
        self._state = event.data["to"]
        if self._state == AgentState.THINKING:
            self._agent_text = ""
        self._update()

    async def _on_transcript(self, event: Event) -> None:
        self._user_text = event.data.get("text", "")
        self._update()

    async def _on_llm_token(self, event: Event) -> None:
        self._agent_text += event.data.get("token", "")
        self._update()

    async def _on_llm_complete(self, event: Event) -> None:
        self._update()
