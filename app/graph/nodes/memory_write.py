from graph.state import AgentState
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
    Persists durable facts from the summary into long-term memory.
    """

    print("Memory Write Node invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=Memory)

    query = state["query"]
    summary = state["summary"]

    information = query + "\n\n" + summary
    prompt = MEMORY_PROMPT.format(information=information)
    response = llm.structured_chat(prompt)

    facts = response.facts

    if not facts:
        return {
            "stored_facts": [],
        }

    store_memory(facts)

    return {
        "stored_facts": facts,
    }
