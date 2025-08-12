"""Entry point for running the demo LangGraph workflow."""

from loguru import logger

from debitor.graph.state import AgentState
from debitor.graph.workflow import workflow


def main() -> None:
    """Run a tiny demo conversation graph.

    The workflow is intentionally minimal but showcases how the project
    can be extended with additional nodes and edges to support complex
    non-linear conversations.
    """

    state: AgentState = {"user_input": "check my balance"}
    result = workflow.invoke(state)  # type: ignore[arg-type]
    logger.info("Final state: {}", result)


if __name__ == "__main__":
    main()
