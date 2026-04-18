from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from graph.state import AgentState
from observability import trace_node_event
from tools.search_tool import SearchTool


def _search_once(query: str) -> tuple[str, list[dict], float]:
    started_at = perf_counter()
    search_tool = SearchTool()
    results = search_tool.invoke({"query": query})
    return query, results, round((perf_counter() - started_at) * 1000, 2)


def web_search_batch_node(state: AgentState) -> AgentState:
    """
    Runs batched web searches, merges results, and enqueues summarization.
    """

    trace_node_event(state, "web_search_batch", "node_started")

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
        trace_node_event(
            state,
            "web_search_batch",
            "node_completed",
            searched_queries=[],
            total_result_count=0,
        )
        return {
            "search_results": [],
            "pending_tasks": pending_tasks + [{"name": "summarize", "args": {}}],
        }

    max_workers = min(len(normalized_queries), 3)
    collected_results: list[dict] = []

    trace_node_event(
        state,
        "web_search_batch",
        "batch_search_started",
        search_queries=normalized_queries,
        max_workers=max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_search_once, query): query
            for query in normalized_queries
        }
        for future in as_completed(futures):
            try:
                query, results, duration_ms = future.result()
                collected_results.extend(results)
                trace_node_event(
                    state,
                    "web_search_batch",
                    "search_query_completed",
                    query=query,
                    result_count=len(results),
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                query = futures[future]
                trace_node_event(
                    state,
                    "web_search_batch",
                    "search_query_failed",
                    query=query,
                    error=str(exc),
                )

    deduped_results = []
    seen_urls: set[str] = set()
    for result in collected_results:
        url = result.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        deduped_results.append(result)

    trace_node_event(
        state,
        "web_search_batch",
        "node_completed",
        searched_queries=normalized_queries,
        total_result_count=len(deduped_results),
    )

    return {
        "search_results": deduped_results,
        "pending_tasks": pending_tasks + [{"name": "summarize", "args": {}}],
    }
