from typing import Annotated

from observability import trace_node_event
from pydantic import BaseModel

from graph.state import AgentState
from tools.llm import LLM


class SearchQueryPlan(BaseModel):
    queries: Annotated[list[str], "One to three focused search queries"]


SYSTEM_PROMPT = """
You are a search query planner for a research assistant.

Given a user query and prior memories, generate 1-3 focused web-search queries.

Rules:
- Use at most 3 queries
- Prefer precise search phrases over conversational wording
- If the request is time-sensitive, include the freshest terms that help retrieval
- Avoid duplicates or near-duplicates
- Return ONLY valid JSON
- No explanation
- No markdown
- No extra text
"""

SEARCH_QUERY_PROMPT = """
User query:
{query}

Prior memories:
{memory_context}
"""


def search_query_node(state: AgentState) -> AgentState:
    """
    Generates concrete web-search queries and seeds one batched search task.
    """

    trace_node_event(state, "search_query", "node_started")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=SearchQueryPlan)

    query = state["query"]
    memories = state.get("memory_context") or []
    memory_block = "\n".join(
        f"- {memory}" for memory in memories
    ) if memories else "(none)"

    prompt = SEARCH_QUERY_PROMPT.format(
        query=query,
        memory_context=memory_block,
    )
    response = llm.structured_chat(
        prompt,
        trace={
            "run_id": state.get("run_id"),
            "node": "search_query",
            "operation": "generate_search_queries",
        },
    )

    deduped_queries: list[str] = []
    seen: set[str] = set()
    for value in response.queries:
        normalized = value.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped_queries.append(normalized)
        if len(deduped_queries) == 3:
            break

    if not deduped_queries:
        deduped_queries = [query]

    pending_tasks = list(state.get("pending_tasks", []))

    trace_node_event(
        state,
        "search_query",
        "node_completed",
        generated_queries=deduped_queries,
    )

    return {
        "search_queries": deduped_queries,
        "search_results": [],
        "summary": "",
        "pending_tasks": pending_tasks + [
            {"name": "web_search_batch", "args": {"queries": deduped_queries}}
        ],
    }
