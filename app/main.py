from graph.builder import build_graph
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    graph = build_graph()

    while True:
        query = input("Ask: ")

        result = graph.invoke({
            "query": query,
            "chat_history": [],
            "memory": [],
            "plan": [],
            "current_step_index": 0,
            "current_task": None,
            "search_results": [],
            "summary": "",
            "final_answer": "",
        })

        print("Answer: ", result)
