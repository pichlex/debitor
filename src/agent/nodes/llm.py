import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from ..prompts import SYSTEM_PROMPT
from ..state import AgentState
from ..tools import TOOLS

load_dotenv()

def llm_node(state: AgentState) -> dict[str, Any]:
    """Универсальный LLM-узел с bind_tools и поддержкой истории сообщений."""
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(TOOLS)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    ai = (prompt | llm).invoke({"messages": state.get("messages", [])})
    return {"messages": [ai]}
