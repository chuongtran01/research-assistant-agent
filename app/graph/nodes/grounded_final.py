from typing import Annotated

from langchain_core.messages import AIMessage
from observability import preview_text, trace_node_event
from pydantic import BaseModel

from graph.state import AgentState
from tools.llm import LLM


class GroundedFinalAnswer(BaseModel):
    answer: Annotated[str, "The final answer"]


SYSTEM_PROMPT = """
You are the grounded final-answer node for a research assistant.

Answer using the provided research summary and memory context.
Do not rely on unstated outside knowledge unless it is only needed to make the wording natural.
If the provided research summary and memory context are insufficient, say so plainly.

IMPORTANT:
- Return ONLY valid JSON
- No explanation
- No markdown
- No extra text

Return a JSON object with the following fields:
- answer: The final answer
"""

GROUNDED_FINAL_PROMPT = """
User query:
{query}

Relevant prior memory:
{memory_context}

Research summary:
{summary}

Instructions:
- Prefer grounded information from the research summary and memory context
- If the research summary says no relevant results were found, reflect that honestly
- If the provided context is insufficient, say so rather than inventing details
"""


def grounded_final_node(state: AgentState) -> AgentState:
    """
    Produces a grounded answer from retrieved memory and research summary.
    """

    trace_node_event(state, "grounded_final", "node_started")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=GroundedFinalAnswer)

    memory_context = state.get("memory_context") or []
    memory_block = "\n".join(
        f"- {memory}" for memory in memory_context
    ) if memory_context else "(none)"

    prompt = GROUNDED_FINAL_PROMPT.format(
        query=state["query"],
        memory_context=memory_block,
        summary=state.get("summary", "") or "(none)",
    )
    response = llm.structured_chat(
        prompt,
        trace={
            "run_id": state.get("run_id"),
            "node": "grounded_final",
            "operation": "answer_from_grounded_context",
        },
    )

    trace_node_event(
        state,
        "grounded_final",
        "node_completed",
        answer_preview=preview_text(response.answer),
    )

    return {
        "chat_history": [AIMessage(content=response.answer)],
        "final_answer": response.answer,
    }
