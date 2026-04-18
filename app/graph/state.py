from typing import TypedDict, List, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from tools.search_tool import SearchResult


class AgentState(TypedDict, total=False):
    query: Annotated[str, "The query of the agent"]
    chat_history: Annotated[List[BaseMessage],
                            "The chat history of the agent", add_messages]
    memory_context: Annotated[List[str], "The memory context of the agent"]
    route: Annotated[Literal["research", "direct_answer"],
                     "The top-level workflow selected for the query"]
    search_results: Annotated[List[SearchResult],
                              "The search results of the agent"]
    summary: Annotated[str, "The summary of the agent"]
    stored_facts: Annotated[List[str], "The facts written to long-term memory"]
    final_answer: Annotated[str, "The final answer of the agent"]
