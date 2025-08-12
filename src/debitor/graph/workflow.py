"""Assemble the LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from .nodes.balance import handle_balance
from .nodes.fallback import handle_fallback
from .nodes.intent import classify_intent, route_intent
from .nodes.pay import handle_pay
from .state import AgentState

builder = StateGraph(AgentState)

# Register nodes
builder.add_node("classify", classify_intent)
builder.add_node("pay", handle_pay)
builder.add_node("balance", handle_balance)
builder.add_node("fallback", handle_fallback)

# Entry point
builder.add_edge(START, "classify")

# Non-linear routing based on detected intent
builder.add_conditional_edges(
    "classify",
    route_intent,
    {
        "pay": "pay",
        "balance": "balance",
        "fallback": "fallback",
    },
)

# Finish after handling known intents
builder.add_edge("pay", END)
builder.add_edge("balance", END)

# End the conversation after fallback
builder.add_edge("fallback", END)

# Compile the executable graph
workflow = builder.compile()
