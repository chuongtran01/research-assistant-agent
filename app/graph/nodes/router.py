from graph.state import AgentState


def router_node(state: AgentState) -> AgentState:
    """
    Pops the next task from the queue and exposes it as current_task.
    """
    pending_tasks = list(state.get("pending_tasks", []))

    return {
        "current_task": pending_tasks.pop(0) if pending_tasks else {"name": "final", "args": {}},
        "pending_tasks": pending_tasks,
    }
