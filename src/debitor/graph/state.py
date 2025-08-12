"""Typed state passed between graph nodes."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Represents data shared across workflow nodes."""

    user_input: str
    intent: str
