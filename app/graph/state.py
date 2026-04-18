from typing import TypedDict, List, Annotated, Union
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from tools.search_tool import SearchResult


class TaskModel(BaseModel):
    name: Annotated[str, "The name of the task"]
    description: Annotated[str, "The description of the task"]
    args: Annotated[dict, "The arguments of the task"]


class AgentState(TypedDict):
    query: Annotated[str, "The query of the agent"]
    chat_history: Annotated[List[BaseMessage],
                            "The chat history of the agent", add_messages]
    memory_context: Annotated[List[str], "The memory context of the agent"]
    plan: Annotated[List[TaskModel], "The plan of the agent"]
    current_step_index: Annotated[int, "The current step index of the plan"]
    current_task: Annotated[Union[TaskModel, None],
                            "The current task of the agent"]
    search_results: Annotated[List[SearchResult],
                              "The search results of the agent"]
    summary: Annotated[str, "The summary of the agent"]
    final_answer: Annotated[str, "The final answer of the agent"]
