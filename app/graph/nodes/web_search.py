from graph.state import AgentState
from tools.search_tool import SearchTool


def web_search_node(state: AgentState) -> AgentState:
    """
    Runs web search for the user's query during the research branch.
    """

    print("Web Search Node invoked")

    query = state.get("query", "").strip()

    if not query:
        return {
            "search_results": [],
        }

    search_tool = SearchTool()
    results = search_tool.invoke({"query": query})

    return {
        "search_results": results,
    }
