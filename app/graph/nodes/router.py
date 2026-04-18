from graph.state import AgentState


def router_node(state: AgentState) -> AgentState:
    """
    Normalizes the top-level route selected by the planner.
    """
    route = state.get("route", "direct_answer")

    return {
        "route": route if route in {"research", "direct_answer"} else "direct_answer",
    }
