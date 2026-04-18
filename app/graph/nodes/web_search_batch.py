from concurrent.futures import ThreadPoolExecutor, as_completed

from graph.state import AgentState
from tools.search_tool import SearchTool


def _search_once(query: str) -> list[dict]:
    search_tool = SearchTool()
    return search_tool.invoke({"query": query})


def web_search_batch_node(state: AgentState) -> AgentState:
    """
    Runs batched web searches, merges results, and enqueues summarization.
    """

    print("Web Search Batch Node invoked")

    current_task = state.get(
        "current_task", {"name": "web_search_batch", "args": {}})
    raw_queries = current_task.get("args", {}).get(
        "queries", state.get("search_queries", []))
    pending_tasks = list(state.get("pending_tasks", []))

    if isinstance(raw_queries, str):
        queries = [raw_queries]
    elif isinstance(raw_queries, list):
        queries = raw_queries
    else:
        queries = []

    normalized_queries = []
    seen_queries: set[str] = set()
    for value in queries:
        if not isinstance(value, str):
            continue
        query = value.strip()
        if not query:
            continue
        lowered = query.lower()
        if lowered in seen_queries:
            continue
        seen_queries.add(lowered)
        normalized_queries.append(query)

    if not normalized_queries:
        return {
            "search_results": [],
            "pending_tasks": pending_tasks + [{"name": "summarize", "args": {}}],
        }

    max_workers = min(len(normalized_queries), 3)
    collected_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Searching for queries: {normalized_queries}")
        futures = {
            executor.submit(_search_once, query): query
            for query in normalized_queries
        }
        for future in as_completed(futures):
            try:
                collected_results.extend(future.result())
            except Exception as exc:
                query = futures[future]
                print(f"Web search failed for query '{query}': {exc}")

    deduped_results = []
    seen_urls: set[str] = set()
    for result in collected_results:
        url = result.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        deduped_results.append(result)

    return {
        "search_results": deduped_results,
        "pending_tasks": pending_tasks + [{"name": "summarize", "args": {}}],
    }
