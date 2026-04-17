from graph.state import AgentState
from langgraph.graph import StateGraph, START, END
from graph.nodes.planner import planner_node
from graph.nodes.web_search import web_search_node
from graph.nodes.summarize import summarize_node
from graph.nodes.final import final_node
from graph.nodes.memory import memory_node
from graph.nodes.router import router_node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("memory", memory_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["current_task"]["name"],
        {
            "web_search": "web_search",
            "summarize": "summarize",
            "memory": "memory",
            "final": "final",
        },
    )

    graph.add_edge("web_search", "summarize")
    graph.add_edge("summarize", "router")
    graph.add_edge("memory", "router")

    graph.add_edge("final", END)

    return graph.compile()
