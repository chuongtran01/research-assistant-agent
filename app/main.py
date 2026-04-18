from graph.builder import build_graph
from memory.vector_store import retrieve_memories

if __name__ == "__main__":
    graph = build_graph()

    while True:
        query = input("Ask: ")

        result = graph.invoke({
            "query": query,
            "chat_history": [],
            "memory_context": [],
            "plan": [],
            "current_step_index": 0,
            "current_task": None,
            "search_results": [],
            "summary": "",
            "final_answer": "",
        })

        print("Memory", retrieve_memories())

        print("--------------------------------")

        print("Answer: ", result["final_answer"])
