from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Маршруты по уточнённой схеме
Route = Literal[
    # Этап знакомства / LPR
    "intro", "classify_lpr", "is_lpr", "not_lpr",
    # Причины неоплаты
    "named_date", "claims_our_side", "client_issues", "no_answer",
    # Подветки named_date
    "named_date_within_week", "named_date_over_week",
    # Подветки claims_our_side
    "duplicate_closings", "already_paid", "needs_reconciliation", "needs_invoice",
    # Прочие доменные этапы
    "restructuring", "pretrial_notice",
    # Служебные
    "unknown"
]

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Route
    stage: str | None
    scratch: dict[str, Any]
    meta: dict[str, Any]
    plan: str | None
