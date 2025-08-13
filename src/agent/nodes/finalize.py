import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from ..prompts import FINALIZE_PROMPT
from ..state import AgentState


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Финализирующий узел: формирует краткий деловой ответ по контексту."""
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", FINALIZE_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    ai = (prompt | llm).invoke({"messages": state.get("messages", [])})
    return {"messages": [ai]}
