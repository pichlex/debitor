from typing import Any

from ..state import AgentState


def entry_node(state: AgentState) -> dict[str, Any]:
    """Единая точка входа. Увеличивает счетчик хода и решает, показывать ли приветствие."""
    scratch = dict(state.get("scratch", {}) or {})
    turn = int(scratch.get("turn", 0)) + 1
    scratch["turn"] = turn
    # route помечаем для условного ребра из графа
    return {"scratch": scratch, "route": "intro" if turn == 1 else "classify_lpr"}
