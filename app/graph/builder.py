from graph.state import AgentState
from langgraph.graph import StateGraph, START, END
from graph.nodes.planner import planner_node
from graph.nodes.search_query import search_query_node
from graph.nodes.web_search import web_search_node
from graph.nodes.summarize import summarize_node
from graph.nodes.final import final_node
from graph.nodes.memory_write import memory_write_node
from graph.nodes.memory_retrieval import memory_retrieval_node
from graph.nodes.router import router_node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("memory_retrieval", memory_retrieval_node)
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("search_query", search_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("memory_write", memory_write_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "memory_retrieval")
    graph.add_edge("memory_retrieval", "planner")
    graph.add_edge("planner", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["current_task"]["name"],
        {
            "search_query": "search_query",
            "web_search": "web_search",
            "summarize": "summarize",
            "memory_write": "memory_write",
            "final": "final",
        },
    )

    graph.add_edge("search_query", "router")
    graph.add_edge("web_search", "router")
    graph.add_edge("summarize", "router")
    graph.add_edge("memory_write", "router")

    graph.add_edge("final", END)

    return graph.compile()
