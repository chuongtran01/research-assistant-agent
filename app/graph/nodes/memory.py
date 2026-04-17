from graph.state import AgentState
from pydantic import BaseModel
from typing import Annotated, List
from tools.llm import LLM
from langchain_core.messages import SystemMessage


class Memory(BaseModel):
    facts: Annotated[List[str], "A list of 1-2 durable facts"]


SYSTEM_PROMPT = """
You are a memory agent that extract 1-2 durable facts from the given information.

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


def memory_node(state: AgentState) -> AgentState:
    """
    Memory node that stores the agent's response.
    """

    print("Memory Node invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=Memory)

    query = state["query"]
    summary = state["summary"]

    information = query + "\n\n" + summary
    prompt = MEMORY_PROMPT.format(information=information)
    response = llm.structured_chat(prompt)

    facts = response.facts

    if not facts:
        return state

    # Convert to structured memory messages
    memory_messages = [
        SystemMessage(content=f"[MEMORY] {fact}")
        for fact in facts
    ]

    return {
        **state,
        "memory": memory_messages,
        "current_step_index": state["current_step_index"] + 1,
    }
