from __future__ import annotations
from typing import Any, Dict
from ..state import AgentState

# Узлы, в которые мы можем «продолжать» ход (resume_at)
_ALLOWED_RESUME = {
    "classify_lpr",
    "classify_reason",
    "classify_agreement",
    "create_duplicate_request",
}

def entry_node(state: AgentState) -> Dict[str, Any]:
    """
    Точка входа каждого хода:
    - инкрементируем turn;
    - если это первый ход → intro;
    - иначе, если ранее узел пометил продолжение (resume_at) и оно допустимо → начинаем с него;
    - иначе → по умолчанию с classify_lpr.
    """
    scratch = dict(state.get("scratch", {}) or {})
    turn = int(scratch.get("turn", 0)) + 1
    scratch["turn"] = turn

    resume = scratch.pop("resume_at", None)
    if resume not in _ALLOWED_RESUME:
        resume = None

    route = "intro" if turn == 1 else (resume or "classify_lpr")
    return {"scratch": scratch, "route": route}