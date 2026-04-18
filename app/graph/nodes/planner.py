from typing import Dict, Any, List

from graph.state import AgentState, TaskModel
from tools.llm import LLM
from pydantic import BaseModel
from typing import Annotated
from langchain_core.messages import HumanMessage


class PlanModel(BaseModel):
    steps: Annotated[List[TaskModel], "A list of steps to execute"]


SYSTEM_PROMPT = """
You are a planning agent that converts a user query into a sequence of executable steps.

You are NOT calling functions. You are defining a plan for a state-driven execution system.

Available actions:
- web_search: performs a web search using the provided query
- summarize: summarizes search results stored in state
- memory_write: extracts durable facts from summary and stores them
- final: produces final answer

Execution model:
- Steps are executed sequentially by a state machine
- Some steps use arguments (only when required)
- Nodes read shared state and/or step arguments

Rules:
- If external knowledge is needed, start with web_search
- If relevant memory is sufficient, DO NOT use web_search
- Always run summarize after web_search
- Always store useful insights using memory_write
- Always end with final
- Keep the plan minimal and efficient

Output format (STRICT):

{
  "steps": [
    {
      "name": "web_search | summarize | memory_write | final",
      "description": "what this step does",
      "args": {
        "query": "only required for web_search"
      }
    }
  ]
}

Important rules:
- ONLY web_search should use args.query
- summarize, memory_write, final should use empty args {}
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
    - takes user query
    - generates structured execution plan
    """
    print("Planner Node invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=PlanModel)

    query = state["query"]

    memories = state.get("memory_context") or []
    memory_block = "\n".join(
        f"- {m}" for m in memories) if memories else "(none)"

    prompt = PLANNER_PROMPT.format(query=query, memory_context=memory_block)

    plan = llm.structured_chat(prompt)

    return {
        "chat_history": [HumanMessage(content=query)],
        "plan": [step.model_dump() for step in plan.steps],
        "current_step_index": 0
    }
