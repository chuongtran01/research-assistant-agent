from graph.state import AgentState
from tools.llm import LLM
from pydantic import BaseModel
from typing import Annotated
from langchain_core.messages import AIMessage


class FinalAnswer(BaseModel):
    answer: Annotated[str, "The final answer"]


SYSTEM_PROMPT = """
    You are a final answer that returns the final answer. You will be given a user query and a summary of the search results.
    
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
"""


def final_node(state: AgentState) -> AgentState:
    """
    Produces the final answer from retrieved memory and optional research summary.
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
