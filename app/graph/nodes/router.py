from graph.state import AgentState


def router_node(state: AgentState) -> AgentState:
    """
    PURE ROUTING NODE:

    Executes one step from the plan.
    """

    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)

    if not plan or idx >= len(plan):
        return {
            **state,
            "final_answer": "No plan to execute",
        }

    task = plan[idx]

    return {
        "current_task": task,
    }
