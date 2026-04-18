from typing import Annotated, Literal

from graph.state import AgentState
from tools.llm import LLM
from pydantic import BaseModel
from langchain_core.messages import HumanMessage


class RouteDecision(BaseModel):
    route: Annotated[
        Literal["research", "direct_answer"],
        "Route to research when web search is needed, otherwise direct_answer",
    ]


SYSTEM_PROMPT = """
You are a routing agent for a research assistant.

Decide whether the assistant should:
- research: run web search before answering
- direct_answer: answer directly without web search

- Choose research when the question depends on fresh, external, missing, or uncertain information
- Choose direct_answer when the question can be answered well without web search
- If the user asks about recent, latest, current, or time-sensitive information, choose research
- Return ONLY valid JSON
- No explanation
- No markdown
"""

PLANNER_PROMPT = """
User query:
{query}

Prior memories (may be empty):
{memory_context}
"""


def planner_node(state: AgentState) -> AgentState:
    """
    Planner node:
    - examines the user query and retrieved memories
    - seeds the executor with the initial task queue
    """
    print("Planner Node invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=RouteDecision)

    query = state["query"]

    memories = state.get("memory_context") or []
    memory_block = "\n".join(
        f"- {m}" for m in memories) if memories else "(none)"

    prompt = PLANNER_PROMPT.format(query=query, memory_context=memory_block)

    decision = llm.structured_chat(prompt)

    initial_tasks = (
        [{"name": "search_query", "args": {}}]
        if decision.route == "research"
        else [{"name": "direct_answer", "args": {}}]
    )

    return {
        "chat_history": [HumanMessage(content=query)],
        "pending_tasks": initial_tasks,
    }
