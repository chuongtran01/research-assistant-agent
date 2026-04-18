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
    """


def final_node(state: AgentState) -> AgentState:
    """
    Final answer node that returns the final answer.
    """

    print("Final Answer Node Invoked")

    llm = LLM(system_prompt=SYSTEM_PROMPT,
              structured_output=FinalAnswer)

    query = state["query"] + "\n\n" + state["summary"]
    prompt = FINAL_ANSWER_PROMPT.format(query=query)

    response = llm.structured_chat(prompt)

    answer = response.answer

    return {
        "chat_history": [AIMessage(content=response.answer)],
        "final_answer": answer,
    }
