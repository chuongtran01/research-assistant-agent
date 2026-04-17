from typing import Dict, Any, List

from graph.state import AgentState, TaskModel
from tools.llm import LLM
from pydantic import BaseModel
from typing import Annotated


class PlanModel(BaseModel):
    steps: Annotated[List[TaskModel], "A list of steps to execute"]


SYSTEM_PROMPT = """
You are a planning agent.

Your job is to break a user query into a sequence of executable steps.

Available tools:
- web_search(query: string)
- summarize(text: string)
- final(answer: string)
- memory(facts: string)

Rules:
- If the query requires external knowledge → start with web_search
- Always summarize after retrieving information
- Store useful insights in memory if relevant
- Always end with final
- Keep steps minimal but sufficient

Each step must include:
- name: tool name
- description: what the step does
- args: dictionary of arguments

IMPORTANT:
- Return ONLY valid JSON
- No explanation
- No markdown
- No extra text
"""

PLANNER_PROMPT = """
User query:
{query}
"""


def planner_node(state: AgentState) -> AgentState:
    """
    Planner node:
    - takes user query
    - generates structured execution plan
    """
    print("Planner Node invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=PlanModel)

    query = state["query"]

    prompt = PLANNER_PROMPT.format(query=query)

    plan = llm.structured_chat(prompt)

    return {
        **state,
        "chat_history": query,
        "plan": [step.model_dump() for step in plan.steps],
        "current_step_index": 0
    }
