from typing import Annotated

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from graph.state import AgentState
from tools.llm import LLM


class DirectAnswer(BaseModel):
    answer: Annotated[str, "The final answer"]


SYSTEM_PROMPT = """
You are the direct-answer node for a research assistant.

Answer the user directly without web search.
You may use the provided memory context when it is helpful, but you may also answer from your own general knowledge.
If the question depends on current or very recent information, say that web search or current verification may be needed.

IMPORTANT:
- Return ONLY valid JSON
- No explanation
- No markdown
- No extra text

Return a JSON object with the following fields:
- answer: The final answer
"""

DIRECT_ANSWER_PROMPT = """
User query:
{query}

Relevant prior memory:
{memory_context}

Instructions:
- Use memory context when it helps
- If memory context is empty or insufficient, answer from general knowledge when you can do so confidently
- If the question requires current information, say that current verification is needed
"""


def direct_answer_node(state: AgentState) -> AgentState:
    """
    Produces a direct answer from memory and model knowledge without web search.
    """

    print("Direct Answer Node Invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=DirectAnswer)

    memory_context = state.get("memory_context") or []
    memory_block = "\n".join(
        f"- {memory}" for memory in memory_context
    ) if memory_context else "(none)"

    prompt = DIRECT_ANSWER_PROMPT.format(
        query=state["query"],
        memory_context=memory_block,
    )
    response = llm.structured_chat(prompt)

    return {
        "chat_history": [AIMessage(content=response.answer)],
        "final_answer": response.answer,
    }
