"""Nodes checking account balance."""

from loguru import logger

from ..state import AgentState


def handle_balance(state: AgentState) -> AgentState:
    """Return account balance (placeholder)."""

    logger.info("Pretending to fetch account balance…")
    # Real implementation would call backend systems here.
    return state
