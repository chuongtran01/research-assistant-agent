from graph.state import AgentState
from observability import trace_node_event


def router_node(state: AgentState) -> AgentState:
    """
    Pops the next task from the queue and exposes it as current_task.
    """
    pending_tasks = list(state.get("pending_tasks", []))
    current_task = pending_tasks.pop(0) if pending_tasks else {
        "name": "direct_answer",
        "args": {},
    }

    trace_node_event(
        state,
        "router",
        "task_dispatched",
        dispatched_task=current_task["name"],
        remaining_tasks=[task["name"] for task in pending_tasks],
    )

    return {
        "current_task": current_task,
        "pending_tasks": pending_tasks,
    }
