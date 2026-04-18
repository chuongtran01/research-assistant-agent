from graph.state import AgentState
from tools.llm import LLM
from pydantic import BaseModel
from typing import Annotated
from langchain_core.messages import AIMessage


class FinalAnswer(BaseModel):
    answer: Annotated[str, "The final answer"]


SYSTEM_PROMPT = """
You are the final answer node for a research assistant.

Use the provided memory context and research summary when they are helpful.
If they are empty or insufficient, you may answer from your own general knowledge.
If the question is time-sensitive and the provided research context is empty, say that you may need web search to answer reliably.

IMPORTANT:
- Return ONLY valid JSON
- No explanation
- No markdown
- No extra text

Return a JSON object with the following fields:
- answer: The final answer
"""

FINAL_ANSWER_PROMPT = """
User query:
{query}

Relevant prior memory:
{memory_context}

Research summary:
{summary}

Instructions:
- Prefer grounded information from memory context and research summary when available
- If both are empty or not useful, answer from general knowledge when you can do so confidently
- If the question depends on current or very recent information and no research summary is available, say that current verification is needed
"""


def final_node(state: AgentState) -> AgentState:
    """
    Produces the final answer from memory, research, or general knowledge.
    """

    print("Final Answer Node Invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT,
              structured_output=FinalAnswer)

    memory_context = state.get("memory_context") or []
    memory_block = "\n".join(
        f"- {memory}" for memory in memory_context
    ) if memory_context else "(none)"

    prompt = FINAL_ANSWER_PROMPT.format(
        query=state["query"],
        memory_context=memory_block,
        summary=state.get("summary", "") or "(none)",
    )

    response = llm.structured_chat(prompt)

    answer = response.answer

    return {
        "chat_history": [AIMessage(content=response.answer)],
        "final_answer": answer,
    }
