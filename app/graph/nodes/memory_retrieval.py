from graph.state import AgentState
from memory.vector_store import retrieve_memories


def memory_retrieval_node(state: AgentState) -> dict:
    """
    Loads relevant prior memories into ``memory_context`` before planning.
    """
    print("Memory Retrieval Node invoked")

    query = state.get("query", "") or ""

    memories = retrieve_memories()

    results = [
        mem["fact"] for mem in memories
        if query.lower() in mem["fact"].lower()
    ]

    return {"memory_context": results}
