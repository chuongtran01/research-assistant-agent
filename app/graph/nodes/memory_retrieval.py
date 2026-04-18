from graph.state import AgentState
from memory.vector_store import vector_store
from observability import preview_text, trace_node_event


def memory_retrieval_node(state: AgentState) -> dict:
    """
    Loads relevant prior memories into ``memory_context`` before planning.
    """
    trace_node_event(
        state,
        "memory_retrieval",
        "node_started",
        query_preview=preview_text(state.get("query", "")),
    )

    query = state.get("query", "") or ""

    results = vector_store.similarity_search(query, k=5)

    memory_context = [
        mem.page_content for mem in results
    ]

    trace_node_event(
        state,
        "memory_retrieval",
        "node_completed",
        retrieved_memory_count=len(memory_context),
    )

    return {"memory_context": memory_context}
