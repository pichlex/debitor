from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel

from ..graph import get_graph_for_thread
from ..otel import setup_otel

setup_otel()  # инициализация OTEL до создания приложения

app = FastAPI(title="LangGraph Agent MVP")
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)

class ChatRequest(BaseModel):
    thread_id: str
    input: str
    metadata: dict | None = None

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    (graph, shard_idx) = get_graph_for_thread(req.thread_id)
    config = {"configurable": {"thread_id": req.thread_id}}
    state_in = {"messages": [HumanMessage(content=req.input)], "meta": req.metadata or {}}

    with tracer.start_as_current_span("api.chat") as span:
        span.set_attribute("thread.id", req.thread_id)
        span.set_attribute("shard.index", shard_idx)
        out = graph.invoke(state_in, config)

    messages = out.get("messages", [])
    last = messages[-1].content if messages else ""
    return {"thread_id": req.thread_id, "output": last, "shard": shard_idx}
