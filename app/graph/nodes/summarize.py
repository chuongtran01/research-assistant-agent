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
User query:
{query}

Search results:
{search_results}
"""


def summarize_node(state: AgentState) -> AgentState:
    """
    Summarizes the web-search results for the research branch.
    """

    print("Summarize Node Invoked")

    search_results = state.get("search_results", [])
    if not search_results:
        return {
            "summary": "No relevant web results were found.",
        }

    llm = LLM(system_prompt=SYSTEM_PROMPT, structured_output=Summary)

    prompt = SUMMARIZE_PROMPT.format(
        query=state["query"],
        search_results=search_results,
    )
    response = llm.structured_chat(prompt)

    return {
        "summary": response.summary,
    }
