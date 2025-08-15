# src/agent/nodes/finalize.py
from __future__ import annotations

from typing import Any, Dict

# Ничего не импортируем из LLM: финализация не должна генерировать клиентский текст.


def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Прозрачная финализация шага:
    - не добавляет клиентских сообщений (оставляет последнюю реплику узла, который шёл до finalize);
    - НЕ перезаписывает stage (если он уже установлен);
    - убирает возможные маркеры продолжения (resume_at), чтобы следующий ход стартовал через entry.
    """
    scratch = dict(state.get("scratch", {}) or {})

    return {
        # Никаких новых AIMessage: на экране останется текст из предыдущего узла
        "messages": [],
        "stage": state.get("stage") or "Финализация",
        "scratch": scratch,
        "route": "unknown",
    }