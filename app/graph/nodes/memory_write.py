from graph.state import AgentState
from observability import trace_node_event
from pydantic import BaseModel
from typing import Annotated, List
from tools.llm import LLM

from tools.memory import store_memory


class Memory(BaseModel):
    facts: Annotated[List[str], "A list of 1-2 durable facts"]


SYSTEM_PROMPT = """
You are a memory agent that extract 1-2 durable facts from the given information and store them in the memory database.

IMPORTANT:
- Return ONLY valid JSON
- No explanation
- No markdown
- No extra text

Return a JSON object with the following fields:
- facts: A list of 1-2 durable facts
"""


MEMORY_PROMPT = """
Information:
{information}
"""


def memory_write_node(state: AgentState) -> AgentState:
    """
    Persists durable facts from the summary and enqueues the grounded response.
    """

    trace_node_event(state, "memory_write", "node_started")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=Memory)

    query = state["query"]
    summary = state["summary"]

    information = query + "\n\n" + summary
    prompt = MEMORY_PROMPT.format(information=information)
    response = llm.structured_chat(
        prompt,
        trace={
            "run_id": state.get("run_id"),
            "node": "memory_write",
            "operation": "extract_durable_facts",
        },
    )

    facts = response.facts

    pending_tasks = list(state.get("pending_tasks", []))

    if not facts:
        trace_node_event(
            state,
            "memory_write",
            "node_completed",
            stored_fact_count=0,
            next_task="grounded_final",
        )
        return {
            "stored_facts": [],
            "pending_tasks": pending_tasks + [{"name": "grounded_final", "args": {}}],
        }

    store_memory(facts)

    trace_node_event(
        state,
        "memory_write",
        "node_completed",
        stored_fact_count=len(facts),
        next_task="grounded_final",
    )

    return {
        "stored_facts": facts,
        "pending_tasks": pending_tasks + [{"name": "grounded_final", "args": {}}],
    }
