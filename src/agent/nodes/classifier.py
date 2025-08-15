# src/agent/nodes/classifier.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()

from ..state import AgentState
from ..config import settings
from ..prompts import (
    DESC_REASON,
    DESC_NAMED_DATE_WINDOW,
    DESC_AGREEMENT,
    DESC_CLAIMS_SUB,
)

# --------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------
class OutRoute(BaseModel):
    route: str
    notes: Optional[str] = None
    target_date: Optional[str] = None  # ISO YYYY-MM-DD

_llm = ChatOpenAI(model=settings.model_name, temperature=0)

def _last_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages", []) or []):
        t = getattr(m, "type", None)
        if t in ("human", "user") or m.__class__.__name__ == "HumanMessage":
            return str(m.content)
    return ""

# --------------------------------------------------------------------
# КЛАССИФИКАЦИЯ ЛПР
# Главный принцип: решение принимается по последней реплике пользователя.
# Добавлены few-shot примеры (русские формулировки/аббревиатуры).
# --------------------------------------------------------------------
_LPR_SYSTEM = """Ты — маршрутизатор диалога по задолженности.
Определи, является ли текущий собеседник ЛПР (лицом, принимающим решения по оплате).

Решение принимай в ПЕРВУЮ ОЧЕРЕДЬ по ПОСЛЕДНЕЙ реплике пользователя.
История диалога дана только для контекста и не важнее последней реплики.

Верни route строго одним из:
- 'is_lpr'   — если собеседник явно подтверждает, что он ЛПР / директор / отвечает за оплату;
- 'not_lpr'  — если собеседник явно НЕ ЛПР (секретарь, бухгалтер без полномочий и т.п.);
- 'ask_lpr'  — если реплика неинформативна или нельзя понять статус.

Примеры:
  Пользователь: «я лпр» → route=is_lpr
  Пользователь: «я директор/руковожу/принимаю решения по оплате» → route=is_lpr
  Пользователь: «я бухгалтер, ЛПР Иван Петров» → route=not_lpr
  Пользователь: «секретарь/оператор/не я» → route=not_lpr
  Пользователь: «алло», «да», «слушаю», пусто → route=ask_lpr
  Ты: "С кем я могу переговорить по поводу задолженности <НАЗВАНИЕ КОМПАНИИ> перед Яндекс Доставкой?", Пользователь: "Со мной" → route=is_lpr

Если сомневаешься — выбери 'ask_lpr'.
В поле notes коротко (1–2 фразы) объясни причину решения.
"""

_prompt_lpr = ChatPromptTemplate.from_messages(
    [
        ("system", _LPR_SYSTEM),
        # История — опционально, не обязательна для вывода
        MessagesPlaceholder(variable_name="messages", optional=True),
        # Явно подсовываем последнюю пользовательскую реплику
        ("user", "Последняя реплика пользователя (ключевая):\n{last_user}"),
    ]
)

def classify_lpr_node(state: AgentState) -> Dict[str, Any]:
    msgs = state.get("messages", []) or []
    last_user = _last_user_text(state)
    chain = _prompt_lpr | _llm.with_structured_output(OutRoute)
    out = chain.invoke({"messages": msgs, "last_user": last_user})
    route = out.route if out.route in {"ask_lpr", "is_lpr", "not_lpr"} else "ask_lpr"
    scratch = dict(state.get("scratch", {}) or {})
    if out.notes:
        scratch["notes_lpr"] = out.notes
    return {"route": route, "scratch": scratch}

# --------------------------------------------------------------------
# КЛАССИФИКАЦИЯ ПРИЧИНЫ
# --------------------------------------------------------------------
_prompt_reason = ChatPromptTemplate.from_messages(
    [
        ("system", DESC_REASON),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def classify_reason_node(state: AgentState) -> Dict[str, Any]:
    chain = _prompt_reason | _llm.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    route = (
        out.route
        if out.route in {"named_date", "claims_our_side", "client_issues", "no_answer"}
        else "unknown"
    )
    scratch = dict(state.get("scratch", {}) or {})
    if out.target_date:
        scratch["payment_date"] = out.target_date
    if out.notes:
        scratch["notes_reason"] = out.notes
    return {"route": route, "scratch": scratch}

# --------------------------------------------------------------------
# ОКНО ДАТЫ (≤7 дней или >7)
# --------------------------------------------------------------------
_prompt_date_window = ChatPromptTemplate.from_messages(
    [
        ("system", DESC_NAMED_DATE_WINDOW),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def classify_named_date_window_node(state: AgentState) -> Dict[str, Any]:
    chain = _prompt_date_window | _llm.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    route = (
        out.route
        if out.route in {"named_date_within_week", "named_date_over_week"}
        else "unknown"
    )
    scratch = dict(state.get("scratch", {}) or {})
    if out.target_date:
        scratch["payment_date"] = out.target_date
    if out.notes:
        scratch["notes_date_window"] = out.notes
    return {"route": route, "scratch": scratch}

# --------------------------------------------------------------------
# СОГЛАСИЕ/НЕСОГЛАСИЕ НА ОПЛАТУ В ТЕЧЕНИЕ НЕДЕЛИ
# --------------------------------------------------------------------
_prompt_agree = ChatPromptTemplate.from_messages(
    [
        ("system", DESC_AGREEMENT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def classify_agreement_node(state: AgentState) -> Dict[str, Any]:
    chain = _prompt_agree | _llm.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    route = out.route if out.route in {"agree", "disagree"} else "unknown"
    scratch = dict(state.get("scratch", {}) or {})
    if out.target_date:
        scratch["payment_date"] = out.target_date
    if out.notes:
        scratch["notes_agree"] = out.notes
    return {"route": route, "scratch": scratch}

# --------------------------------------------------------------------
# ПОДТИПЫ «ПРОБЛЕМА НА НАШЕЙ СТОРОНЕ»
# --------------------------------------------------------------------
_prompt_claims = ChatPromptTemplate.from_messages(
    [
        ("system", DESC_CLAIMS_SUB),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def classify_claims_sub_node(state: AgentState) -> Dict[str, Any]:
    chain = _prompt_claims | _llm.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    route = (
        out.route
        if out.route
        in {"duplicate_closings", "already_paid", "needs_reconciliation", "needs_invoice"}
        else "unknown"
    )
    scratch = dict(state.get("scratch", {}) or {})
    if out.notes:
        scratch["notes_claims_sub"] = out.notes
    return {"route": route, "scratch": scratch}

_prompt_cls_restruct = ChatPromptTemplate.from_messages([
    ("system",
     "Определи реакцию клиента на предложение реструктуризации долга. "
     "Варианты: agree (согласен), disagree (не согласен), unknown (недостаточно данных). "
     "Ответ строго JSON: {\"route\": \"agree|disagree|unknown\"}."),
    MessagesPlaceholder("messages"),
])

def classify_restructuring_agreement_node(state: Dict[str, Any]) -> Dict[str, Any]:
    chain = _prompt_cls_restruct | _llm
    res = chain.invoke({"messages": state.get("messages", [])})
    route = "unknown"
    try:
        import json
        route = json.loads(res.content).get("route", "unknown")
    except Exception:
        pass
    return {"route": route, "stage": "Классификация согласия на реструктуризацию"}
