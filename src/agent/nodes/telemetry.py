from typing import Any

from opentelemetry import trace

from ..state import AgentState

tracer = trace.get_tracer(__name__)

# Хук телеметрии: один span на шаг графа с основными атрибутами

def telemetry_node(state: AgentState) -> dict[str, Any]:
    with tracer.start_as_current_span("graph.telemetry") as span:
        span.set_attribute("agent.route", state.get("route", ""))
        span.set_attribute("agent.stage", state.get("stage", ""))
        scratch = state.get("scratch", {}) or {}
        if "payment_date" in scratch:
            span.set_attribute("agent.payment_date", str(scratch.get("payment_date")))
        meta = state.get("meta", {}) or {}
        for k, v in meta.items():
            span.set_attribute(f"agent.meta.{k}", str(v))
    return {}
