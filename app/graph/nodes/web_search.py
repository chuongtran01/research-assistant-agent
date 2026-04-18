from graph.state import AgentState
from tools.search_tool import SearchTool


def web_search_node(state: AgentState) -> AgentState:
    """
    Runs one queued web search and accumulates the results.
    """

    print("Web Search Node invoked")

    current_task = state.get("current_task", {"name": "web_search", "args": {}})
    query = current_task.get("args", {}).get("query", state.get("query", "")).strip()

    if not query:
        return {}

    search_tool = SearchTool()
    results = search_tool.invoke({"query": query})
    existing_results = list(state.get("search_results", []))

    combined_results = existing_results + results
    deduped_results = []
    seen_urls: set[str] = set()
    for result in combined_results:
        url = result.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        deduped_results.append(result)

    return {
        "search_results": deduped_results,
    }
