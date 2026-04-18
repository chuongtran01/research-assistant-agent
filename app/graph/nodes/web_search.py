from graph.state import AgentState
from tools.search_tool import SearchTool


def web_search_node(state: AgentState) -> AgentState:
    """
    Web search node:
    - takes query from state or current plan step
    - executes search tool
    - stores results in state
    """

    print("Web Search Node invoked")

    task = state["current_task"]

    args = task.get("args", {})
    query = args.get("query", "")

    if not query:
        return {
            **state,
            "search_results": [],
            "final_answer": state.get("final_answer", "Web search step missing query"),
        }

    search_tool = SearchTool()
    results = search_tool.invoke(query)

    return {
        "current_step_index": state["current_step_index"] + 1,
        "search_results": results,
    }
