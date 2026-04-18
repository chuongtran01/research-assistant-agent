from graph.builder import build_graph

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

        print("Answer: ", result["final_answer"])
