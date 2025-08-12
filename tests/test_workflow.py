# ruff: noqa: S101
from debitor.graph.state import AgentState
from debitor.graph.workflow import workflow


def test_balance_path() -> None:
    state: AgentState = {"user_input": "check balance"}
    result = workflow.invoke(state)  # type: ignore[arg-type]
    assert result["intent"] == "balance"

def test_pay_path() -> None:
    state: AgentState = {"user_input": "please pay"}
    result = workflow.invoke(state)  # type: ignore[arg-type]
    assert result["intent"] == "pay"

def test_fallback_path() -> None:
    state: AgentState = {"user_input": "unknown command"}
    result = workflow.invoke(state)  # type: ignore[arg-type]
    assert result["intent"] == "fallback"
