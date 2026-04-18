from graph.builder import build_graph

if __name__ == "__main__":
    graph = build_graph()

    while True:
        query = input("Ask: ")

        result = graph.invoke({
            "query": query,
            "chat_history": [],
            "memory_context": [],
            "search_queries": [],
            "pending_tasks": [],
            "search_results": [],
            "summary": "",
            "stored_facts": [],
            "final_answer": "",
        })

        print("Answer: ", result["final_answer"])
