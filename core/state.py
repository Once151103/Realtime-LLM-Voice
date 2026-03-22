"""
Agent state machine — defines states and valid transitions.
"""

from enum import Enum, auto
from typing import Self


class AgentState(Enum):
    """Voice agent states."""

    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()

    def can_transition_to(self, target: Self) -> bool:
        """Check if a transition from self to target is valid."""
        return target in _VALID_TRANSITIONS.get(self, set())

    def __str__(self) -> str:
        return self.name


# Valid state transitions
_VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.LISTENING},
    AgentState.LISTENING: {AgentState.THINKING, AgentState.IDLE},
    AgentState.THINKING: {AgentState.SPEAKING, AgentState.LISTENING},  # LISTENING = interrupted
    AgentState.SPEAKING: {AgentState.LISTENING, AgentState.IDLE},  # LISTENING = barge-in
}


class StateError(Exception):
    """Raised on invalid state transition."""

    def __init__(self, current: AgentState, target: AgentState):
        self.current = current
        self.target = target
        super().__init__(f"Invalid transition: {current} → {target}")
