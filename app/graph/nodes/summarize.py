from typing import Annotated
from graph.state import AgentState
from pydantic import BaseModel
from tools.llm import LLM


class Summary(BaseModel):
    summary: Annotated[str, "A concise summary of the search results"]


SYSTEM_PROMPT = """
    You are a summarizer that summarizes the agent's search results.
    
    IMPORTANT:
    - Return ONLY valid JSON
    - No explanation
    - No markdown
    - No extra text
    
    Return a JSON object with the following fields:
    - summary: A concise summary of the search results
    """

SUMMARIZE_PROMPT = """
    {search_results}
    """


def summarize_node(state: AgentState) -> AgentState:
    """
    Summarize node that summarizes the agent's response.
    """

    print("Summarize Node Invoked")

    search_results = "\n".join(state["search_results"])

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=Summary)

    prompt = SUMMARIZE_PROMPT.format(search_results=search_results)
    response = llm.structured_chat(prompt)

    return {
        **state,
        "summary": response.summary,
        "current_step_index": state["current_step_index"] + 1,
    }
