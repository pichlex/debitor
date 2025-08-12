"""Nodes handling invoice payments."""

from loguru import logger

from ..state import AgentState


def handle_pay(state: AgentState) -> AgentState:
    """Process invoice payment (placeholder)."""

    logger.info("Pretending to pay an invoice…")
    # Real implementation would interact with payment services here.
    return state
