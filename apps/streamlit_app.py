# apps/streamlit_app.py
import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# гарантируем импорт пакета из корня репозитория
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.graph import get_graph_for_thread
from src.agent.otel import setup_otel

# --- инициализация окружения/телеметрии ---
load_dotenv()
setup_otel()

st.set_page_config(page_title="LangGraph Agent — Demo", layout="wide")
st.title("LangGraph Agent — Streamlit Demo")

# ---------- helpers ----------
def _get_ui_history(thread_id: str):
    if "history_by_thread" not in st.session_state:
        st.session_state.history_by_thread = {}
    return st.session_state.history_by_thread.setdefault(thread_id, [])

def _set_ui_history(thread_id: str, history):
    if "history_by_thread" not in st.session_state:
        st.session_state.history_by_thread = {}
    st.session_state.history_by_thread[thread_id] = history

# ---------- sidebar ----------
with st.sidebar:
    st.subheader("Сессия")
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    if "last_thread_id" not in st.session_state:
        st.session_state.last_thread_id = st.session_state.thread_id

    thread_id = st.text_input("thread_id", value=st.session_state.thread_id)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Новый thread_id", type="secondary"):
            thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    with c2:
        if st.button("Сбросить историю", type="secondary"):
            _set_ui_history(thread_id, [])
            st.session_state.thread_id = thread_id
            st.rerun()

    st.session_state.thread_id = thread_id

    # если тред изменился и его истории ещё нет — создадим пустую
    if st.session_state.last_thread_id != thread_id and thread_id not in st.session_state.get("history_by_thread", {}):
        _set_ui_history(thread_id, [])
    st.session_state.last_thread_id = thread_id

    st.divider()
    st.subheader("Метаданные кейса")
    agent_name = st.text_input("Имя оператора", value=os.getenv("AGENT_NAME", "Иван"))
    company = st.text_input("Компания", value=os.getenv("COMPANY", "<НАЗВАНИЕ КОМПАНИИ>"))
    act_date = st.text_input("Дата акта", value=os.getenv("ACT_DATE", "01.01.2024"))
    act_amount = st.text_input("Сумма акта", value=os.getenv("ACT_AMOUNT", "100 000 ₽"))
    debt_sum = st.text_input("Сумма долга", value=os.getenv("DEBT_SUM", "100 000 ₽"))

    st.divider()
    show_debug = st.checkbox("Показывать внутренние заметки (debug)", value=True)

# ---------- рендер текущей истории ----------
ui_history = _get_ui_history(st.session_state.thread_id)
for msg in ui_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- ввод реплики ----------
prompt = st.chat_input("Введите реплику клиента…")

if prompt:
    # 1) мгновенно добавляем пользовательскую реплику в локальную историю
    ui_history.append({"role": "user", "content": prompt})
    _set_ui_history(st.session_state.thread_id, ui_history)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) готовим контекст и вызываем граф
    graph, shard_idx = get_graph_for_thread(st.session_state.thread_id)
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
    meta = {
        "agent_name": agent_name,
        "company": company,
        "act_date": act_date,
        "act_amount": act_amount,
        "debt_sum": debt_sum,
    }
    state_in = {"messages": [HumanMessage(content=prompt)], "meta": meta}
    out = graph.invoke(state_in, cfg)

    # 3) берём последнюю клиентскую реплику ассистента и добавляем в локальную историю
    last_ai = out.get("messages", [])[-1].content if out.get("messages") else ""
    ui_history.append({"role": "assistant", "content": last_ai})
    _set_ui_history(st.session_state.thread_id, ui_history)
    with st.chat_message("assistant"):
        st.markdown(last_ai)

    # 4) внутренние заметки — только в режиме debug
    if show_debug:
        with st.expander("Внутреннее (для отладки)"):
            scratch = out.get("scratch", {}) or {}
            st.markdown(f"**Текущий узел:** `{scratch.get('current_node', '-')}`")
            st.markdown("**Путь за ход:** " + " → ".join(scratch.get("path", [])))
            if scratch.get("ops_note"):
                st.markdown("**Ops-note:** " + scratch["ops_note"])
            st.json({
                "shard": shard_idx,
                "route": out.get("route"),
                "stage": out.get("stage"),
                "scratch": scratch,
            })


st.caption(
    "История хранится локально по текущему thread_id и не перезаписывается данными из чекпойнтера. "
    "Для чистого сценария создайте новый thread_id."
)
