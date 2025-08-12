"""Fallback node for unrecognized intents."""

from loguru import logger

from ..state import AgentState


def handle_fallback(state: AgentState) -> AgentState:
    """Handle unknown user requests."""

    logger.warning("Unknown intent. Transferring to human operator…")
    return state
