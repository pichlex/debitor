from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..config import settings
from ..prompts import DESC_AGREEMENT, DESC_CLAIMS_SUB, DESC_LPR, DESC_NAMED_DATE_WINDOW, DESC_REASON
from ..state import AgentState

from dotenv import load_dotenv
load_dotenv()

_model = ChatOpenAI(model=settings.model_name, temperature=0)

class OutRoute(BaseModel):
    route: str
    notes: str | None = None
    target_date: str | None = None

# B — LPR classifier
_prompt_lpr = ChatPromptTemplate.from_messages([
    ("system", DESC_LPR),
    MessagesPlaceholder(variable_name="messages"),
])

def classify_lpr_node(state: AgentState) -> dict[str, Any]:
    chain = _prompt_lpr | _model.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    return {"route": out.route or "unknown", "scratch": {"notes": out.notes or ""}}

# D — reason classifier
_prompt_reason = ChatPromptTemplate.from_messages([
    ("system", DESC_REASON),
    MessagesPlaceholder(variable_name="messages"),
])

def classify_reason_node(state: AgentState) -> dict[str, Any]:
    chain = _prompt_reason | _model.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    scratch = dict(state.get("scratch", {}) or {})
    if out.target_date:
        scratch["payment_date"] = out.target_date
    return {"route": out.route or "unknown", "scratch": scratch}

# n1 — window for named date
_prompt_date_window = ChatPromptTemplate.from_messages([
    ("system", DESC_NAMED_DATE_WINDOW),
    MessagesPlaceholder(variable_name="messages"),
])

def classify_named_date_window_node(state: AgentState) -> dict[str, Any]:
    chain = _prompt_date_window | _model.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    scratch = dict(state.get("scratch", {}) or {})
    if out.target_date:
        scratch["payment_date"] = out.target_date
    return {"route": out.route or "unknown", "scratch": scratch}

# n7 — agreement on pay within week
_prompt_agree = ChatPromptTemplate.from_messages([
    ("system", DESC_AGREEMENT),
    MessagesPlaceholder(variable_name="messages"),
])

def classify_agreement_node(state: AgentState) -> dict[str, Any]:
    chain = _prompt_agree | _model.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    scratch = dict(state.get("scratch", {}) or {})
    if out.target_date:
        scratch["payment_date"] = out.target_date
    scratch["offered_restruct"] = scratch.get("offered_restruct", False)
    return {"route": out.route or "unknown", "scratch": scratch}

# n2 — sub-claims
_prompt_claims = ChatPromptTemplate.from_messages([
    ("system", DESC_CLAIMS_SUB),
    MessagesPlaceholder(variable_name="messages"),
])

def classify_claims_sub_node(state: AgentState) -> dict[str, Any]:
    chain = _prompt_claims | _model.with_structured_output(OutRoute)
    out = chain.invoke({"messages": state.get("messages", [])})
    return {"route": out.route or "unknown"}
