# src/agent/graph.py
from __future__ import annotations

import atexit
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

# --- доменная логика / узлы / конфиг -------------------------------------------
from .config import settings
from .nodes.actions import (
    agree_named_date_node,
    already_paid_node,
    create_duplicate_request_node,
    intro_node,
    lpr_prompt_node,
    named_date_over_week_node,
    named_date_within_week_node,
    need_invoice_node,
    need_recon_node,
    no_answer_letter_node,
    not_lpr_node,
    pretrial_notice_node,
    propose_pay_within_week_node,
    restructuring_node,
    restructuring_confirm_node,
    send_closings_node,
    ask_lpr_node,
)
from .nodes.classifier import (
    classify_agreement_node,
    classify_claims_sub_node,
    classify_lpr_node,
    classify_named_date_window_node,
    classify_reason_node,
    classify_restructuring_agreement_node,
)
from .nodes.finalize import finalize_node
from .nodes.llm import llm_node
from .nodes.router import entry_node
from .nodes.telemetry import telemetry_node
from .policies import should_call_tools
from .sharding import parse_csv, select_shard
from .state import AgentState
from .tools import TOOLS

# --- трассировка узлов (для отладки маршрутов) ---------------------------------
# src/agent/graph.py  (замените ТОЛЬКО эту функцию)

def _wrap_traced(
    node_name: str,
    fn: Callable[[dict], dict[str, Any]],
) -> Callable[[dict], dict[str, Any]]:
    """Оборачивает узел: ведёт trace и НЕ теряет состояние scratch.
    - path/current_node обновляются на каждом шаге;
    - если узел не вернул scratch, прокидываем входной.
    - если вернул — мерджим (значения узла имеют приоритет),
      но path/current_node фиксируем из обёртки.
    """
    def _inner(state: dict[str, Any]) -> dict[str, Any]:
        # Входной scratch
        scratch_in = dict(state.get("scratch", {}) or {})

        # Новый путь за ХОД: на entry начинаем с нуля
        path = [] if node_name == "entry" else list(scratch_in.get("path", []))
        path.append(node_name)
        scratch_in["path"] = path
        scratch_in["current_node"] = node_name

        # Передаём в узел уже дополненный scratch
        state2 = dict(state)
        state2["scratch"] = scratch_in

        out = fn(state2) or {}

        # МЕРДЖ: входной scratch + node_scratch
        merged = dict(scratch_in)
        node_scratch = dict(out.get("scratch", {}) or {})
        merged.update(node_scratch)

        # Поверх фиксируем трассировочные поля
        merged["path"] = path
        merged["current_node"] = node_name

        out["scratch"] = merged
        return out

    return _inner


def _add_traced_node(g: StateGraph, name: str, fn: Callable[[dict], dict[str, Any]]) -> None:
    g.add_node(name, _wrap_traced(name, fn))

# --- утилиты --------------------------------------------------------------------
def _resolve_sqlite_path(path: str) -> str:
    """Абсолютный путь к SQLite-файлу + создание директорий."""
    p = Path(path)
    if not p.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        p = (repo_root / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

_opened_contexts: list = []

def _open_sqlite_saver(path: str):
    """
    Открывает SqliteSaver как контекстный менеджер и возвращает объект чекпойнтера.
    Контекст закрывается автоматически при завершении процесса.
    """
    cm = SqliteSaver.from_conn_string(path)
    saver = cm.__enter__()
    _opened_contexts.append(cm)
    atexit.register(lambda: cm.__exit__(None, None, None))
    return saver

# --- построение графа -----------------------------------------------------------
def build_graph() -> StateGraph[AgentState]:
    g = StateGraph(AgentState)

    # Узел входа: считает ход и решает, показывать ли приветствие
    _add_traced_node(g, "entry", entry_node)

    # Узлы по схеме (все с трассировкой)
    _add_traced_node(g, "intro", intro_node)
    _add_traced_node(g, "ask_lpr", ask_lpr_node)  # ← новый шаг после приветствия
    _add_traced_node(g, "classify_lpr", classify_lpr_node)
    _add_traced_node(g, "not_lpr", not_lpr_node)
    _add_traced_node(g, "lpr_prompt", lpr_prompt_node)

    _add_traced_node(g, "classify_reason", classify_reason_node)
    _add_traced_node(g, "named_date_window", classify_named_date_window_node)

    _add_traced_node(g, "named_date_within_week", named_date_within_week_node)
    _add_traced_node(g, "named_date_over_week", named_date_over_week_node)

    _add_traced_node(g, "propose_pay_within_week", propose_pay_within_week_node)
    _add_traced_node(g, "classify_agreement", classify_agreement_node)

    _add_traced_node(g, "agree_named_date", agree_named_date_node)
    _add_traced_node(g, "restructuring", restructuring_node)
    _add_traced_node(g, "classify_restructuring_agreement", classify_restructuring_agreement_node)
    _add_traced_node(g, "restructuring_confirm", restructuring_confirm_node)
    _add_traced_node(g, "pretrial_notice", pretrial_notice_node)
    _add_traced_node(g, "no_answer_letter", no_answer_letter_node)

    _add_traced_node(g, "classify_claims_sub", classify_claims_sub_node)
    _add_traced_node(g, "send_closings", send_closings_node)
    _add_traced_node(g, "already_paid", already_paid_node)
    _add_traced_node(g, "need_recon", need_recon_node)
    _add_traced_node(g, "need_invoice", need_invoice_node)
    _add_traced_node(g, "create_duplicate_request", create_duplicate_request_node)

    # Fallback/инструменты/финализация/телеметрия
    _add_traced_node(g, "agent", llm_node)
    _add_traced_node(g, "tools", ToolNode(tools=TOOLS))
    _add_traced_node(g, "finalize", finalize_node)
    _add_traced_node(g, "telemetry", telemetry_node)

    # Рёбра
    g.add_edge(START, "entry")
    g.add_conditional_edges(
    "entry",
    lambda s: s.get("route", "classify_lpr"),
    {
        "intro": "intro",
        "classify_lpr": "classify_lpr",
        "classify_reason": "classify_reason",
        "classify_agreement": "classify_agreement",
        "classify_restructuring_agreement": "classify_restructuring_agreement",
        "create_duplicate_request": "create_duplicate_request",
    },
)


    # Ход 1: приветствие → вопрос про ЛПР → завершение хода
    g.add_edge("intro", "telemetry")
    g.add_edge("ask_lpr", "telemetry")

    # Дальше: классификация ЛПР и ветвление
    g.add_conditional_edges(
        "classify_lpr",
        lambda s: s.get("route", "unknown"),
        {"ask_lpr": "ask_lpr", "is_lpr": "lpr_prompt", "not_lpr": "not_lpr", "unknown": "agent"},
    )

    # Вопрос ЛПР о плане оплаты — ждём ответ → завершаем ход
    g.add_edge("lpr_prompt", "telemetry")

    # Классификация причин отсутствия оплаты
    g.add_conditional_edges(
        "classify_reason",
        lambda s: s.get("route", "unknown"),
        {
            "named_date": "named_date_window",
            "claims_our_side": "classify_claims_sub",
            "client_issues": "propose_pay_within_week",
            "no_answer": "no_answer_letter",
            "unknown": "agent",
        },
    )

    # «В пределах недели / позже»
    g.add_conditional_edges(
        "named_date_window",
        lambda s: s.get("route", "unknown"),
        {
            "named_date_within_week": "named_date_within_week",
            "named_date_over_week": "named_date_over_week",
            "unknown": "agent",
        },
    )

    # Предложили оплатить за неделю — ждём согласие → завершение хода
    g.add_edge("propose_pay_within_week", "telemetry")

    # Согласие/несогласие на следующем ходе
    g.add_conditional_edges(
        "classify_agreement",
        lambda s: s.get("route", "unknown"),
        {"agree": "agree_named_date", "disagree": "restructuring", "unknown": "agent"},
    )

    g.add_edge("restructuring", "telemetry")
    g.add_edge("restructuring_confirm", "finalize")

    g.add_conditional_edges(
    "classify_restructuring_agreement",
    lambda s: s.get("route", "unknown"),
    {
        "agree": "restructuring_confirm",
        "disagree": "pretrial_notice",
        "unknown": "agent",
    },
    )

    # Подветки «проблема на нашей стороне»
    g.add_conditional_edges(
        "classify_claims_sub",
        lambda s: s.get("route", "unknown"),
        {
            "duplicate_closings": "send_closings",
            "already_paid": "already_paid",
            "needs_reconciliation": "need_recon",
            "needs_invoice": "need_invoice",
            "unknown": "agent",
        },
    )

    # Эти узлы задают вопрос/информируют → завершаем ход
    g.add_edge("send_closings", "telemetry")
    g.add_edge("need_recon", "telemetry")
    g.add_edge("need_invoice", "create_duplicate_request")
    g.add_edge("create_duplicate_request", "telemetry")

    # «Нет ответа» — завершаем
    g.add_edge("no_answer_letter", "finalize")

    # Финальные состояния
    for node in ["named_date_within_week", "named_date_over_week", "agree_named_date", "already_paid"]:
        g.add_edge(node, "finalize")

    g.add_conditional_edges("agent", lambda s: should_call_tools(s), {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "telemetry")
    g.add_edge("telemetry", END)

    return g

# --- компиляция и шардинг -------------------------------------------------------
@lru_cache(maxsize=32)
def compile_graph_for_dsn(model_name: str, openai_api_key: str | None, checkpoint_dsn: str):
    # LLM через переменные окружения (используются в узлах)
    import os
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["MODEL_NAME"] = model_name

    # Checkpointer
    if checkpoint_dsn and checkpoint_dsn.startswith("sqlite:///"):
        sqlite_path = checkpoint_dsn.replace("sqlite:///", "")
        memory = _open_sqlite_saver(_resolve_sqlite_path(sqlite_path))
    elif checkpoint_dsn and checkpoint_dsn.startswith("sqlite:////"):
        # абсолютный путь
        sqlite_path = checkpoint_dsn.replace("sqlite:////", "/")
        memory = _open_sqlite_saver(sqlite_path)
    else:
        memory = MemorySaver()

    g = build_graph()
    return g.compile(checkpointer=memory)

def get_graph_for_thread(thread_id: str):
    """Выбирает шард по thread_id и возвращает (скомпилированный граф, индекс шарда)."""
    shard_count = int(getattr(settings, "shard_count", 1) or 1)
    idx = select_shard(thread_id, shard_count)

    dsn_list: list[str] = parse_csv(getattr(settings, "checkpoint_dsn_shards", ""))
    model_list: list[str] = parse_csv(getattr(settings, "model_names", ""))
    key_list: list[str] = parse_csv(getattr(settings, "openai_api_keys", ""))

    dsn = dsn_list[idx] if dsn_list and idx < len(dsn_list) else settings.checkpoint_dsn
    model = model_list[idx] if model_list and idx < len(model_list) else settings.model_name
    key = key_list[idx] if key_list and idx < len(key_list) else None

    # для узлов, которым нужна MODEL_NAME
    import os
    os.environ["MODEL_NAME"] = model

    compiled = compile_graph_for_dsn(model, key, dsn)
    return compiled, idx