"""Tests for the state machine."""

import pytest

from core.state import AgentState, StateError


class TestAgentState:
    """Test state transitions."""

    def test_valid_transitions(self):
        """Verify all valid state transitions."""
        assert AgentState.IDLE.can_transition_to(AgentState.LISTENING)
        assert AgentState.LISTENING.can_transition_to(AgentState.THINKING)
        assert AgentState.LISTENING.can_transition_to(AgentState.IDLE)
        assert AgentState.THINKING.can_transition_to(AgentState.SPEAKING)
        assert AgentState.THINKING.can_transition_to(AgentState.LISTENING)  # interrupted
        assert AgentState.SPEAKING.can_transition_to(AgentState.LISTENING)  # barge-in
        assert AgentState.SPEAKING.can_transition_to(AgentState.IDLE)

    def test_invalid_transitions(self):
        """Verify invalid state transitions are rejected."""
        assert not AgentState.IDLE.can_transition_to(AgentState.THINKING)
        assert not AgentState.IDLE.can_transition_to(AgentState.SPEAKING)
        assert not AgentState.LISTENING.can_transition_to(AgentState.SPEAKING)
        assert not AgentState.SPEAKING.can_transition_to(AgentState.THINKING)

    def test_self_transitions_invalid(self):
        """States should not transition to themselves."""
        for state in AgentState:
            assert not state.can_transition_to(state)

    def test_state_error(self):
        """StateError includes useful info."""
        err = StateError(AgentState.IDLE, AgentState.SPEAKING)
        assert "IDLE" in str(err)
        assert "SPEAKING" in str(err)
