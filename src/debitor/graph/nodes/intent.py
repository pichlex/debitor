"""Intent classification and routing nodes."""

from __future__ import annotations

from typing import Literal

from loguru import logger

from ..state import AgentState


def classify_intent(state: AgentState) -> AgentState:
    """Very naive intent classifier used for demonstration purposes."""

    text = state.get("user_input", "").lower()
    if "pay" in text:
        state["intent"] = "pay"
    elif "balance" in text:
        state["intent"] = "balance"
    else:
        state["intent"] = "fallback"
    logger.debug("Classified intent: {}", state["intent"])
    return state


def route_intent(state: AgentState) -> Literal["pay", "balance", "fallback"]:
    """Route to the next node based on detected intent."""

    intent = state.get("intent", "fallback")
    logger.debug("Routing intent: {}", intent)
    if intent == "pay":
        return "pay"
    if intent == "balance":
        return "balance"
    return "fallback"
