from observability import configure_logging, new_run_id, preview_text, trace_run_event
from graph.builder import build_graph

if __name__ == "__main__":
    configure_logging()
    graph = build_graph()

    while True:
        query = input("Ask: ")
        run_id = new_run_id()
        trace_run_event(
            run_id,
            "run_started",
            query_preview=preview_text(query),
        )

        try:
            result = graph.invoke({
                "run_id": run_id,
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
        except Exception as exc:
            trace_run_event(
                run_id,
                "run_failed",
                error=str(exc),
            )
            raise

        trace_run_event(
            run_id,
            "run_completed",
            final_answer_preview=preview_text(result["final_answer"]),
        )

        print("Answer: ", result["final_answer"])
