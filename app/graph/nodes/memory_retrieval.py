from graph.state import AgentState
from memory.vector_store import vector_store


def memory_retrieval_node(state: AgentState) -> dict:
    """
    Loads relevant prior memories into ``memory_context`` before planning.
    """
    print("Memory Retrieval Node invoked")

    query = state.get("query", "") or ""

    results = vector_store.similarity_search(query, k=5)

    memory_context = [
        mem.page_content for mem in results
    ]

    return {"memory_context": memory_context}
