import sys
from pathlib import Path

# гарантируем импорт пакета из корня
sys.path.append(str(Path(__file__).resolve().parent.parent))

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent.graph import get_graph_for_thread
from src.agent.otel import setup_otel

app = typer.Typer()

@app.command()
def main(thread: str = typer.Option("t0", "--thread"), prompt: str = typer.Argument(...)):
    load_dotenv()
    setup_otel()
    graph, shard_idx = get_graph_for_thread(thread)
    config = {"configurable": {"thread_id": thread}}
    out = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
    msgs = out.get("messages", [])
    print(msgs[-1].content if msgs else "(no output)")
    typer.echo(f"[debug] shard={shard_idx} route={out.get('route')} stage={out.get('stage')}")

if __name__ == "__main__":
    app()
