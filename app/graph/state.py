from typing import TypedDict, List, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from tools.search_tool import SearchResult


TaskName = Literal[
    "search_query",
    "web_search_batch",
    "summarize",
    "memory_write",
    "direct_answer",
    "grounded_final",
]


class Task(TypedDict):
    name: TaskName
    args: dict[str, object]


class AgentState(TypedDict, total=False):
    query: Annotated[str, "The query of the agent"]
    chat_history: Annotated[List[BaseMessage],
                            "The chat history of the agent", add_messages]
    memory_context: Annotated[List[str], "The memory context of the agent"]
    search_queries: Annotated[List[str], "Generated search queries for the current run"]
    pending_tasks: Annotated[List[Task], "The queue of tasks waiting to run"]
    current_task: Annotated[Task, "The task currently being executed"]
    search_results: Annotated[List[SearchResult],
                              "The search results of the agent"]
    summary: Annotated[str, "The summary of the agent"]
    stored_facts: Annotated[List[str], "The facts written to long-term memory"]
    final_answer: Annotated[str, "The final answer of the agent"]
