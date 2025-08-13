from typing import Literal

from langchain_core.messages import AIMessage

from .state import AgentState

# Условные переходы между узлами

def should_call_tools(state: AgentState) -> Literal["tools", "finalize"]:
    messages = state.get("messages", [])
    if not messages:
        return "finalize"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


def route_switch(state: AgentState) -> str:
    """Возвращает имя следующего узла на основе state['route']."""
    route = state.get("route", "unknown")
    mapping = {
        "named_date": "named_date",
        "claims_our_side": "diagnose",
        "client_issues": "client_issues",
        "no_answer": "no_answer",
        "paid_already": "paid_already",
        "needs_reconciliation": "ask_reconciliation",
        "needs_invoice": "send_invoice",
        "unknown": "agent",  # fallback в универсальный агент с tool-calling
    }
    return mapping.get(route, "agent")
