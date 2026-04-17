from typing import TypedDict, List, Sequence, Annotated, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class TaskModel(BaseModel):
    name: Annotated[str, "The name of the task"]
    description: Annotated[str, "The description of the task"]
    args: Annotated[dict, "The arguments of the task"]


class AgentState(TypedDict):
    query: Annotated[HumanMessage, "The query of the agent"]
    chat_history: Annotated[Sequence[BaseMessage],
                            add_messages, "The chat history of the agent"]
    memory: Annotated[List[SystemMessage],
                      add_messages, "The memory of the agent"]
    plan: Annotated[List[TaskModel], "The plan of the agent"]
    current_step_index: Annotated[int, "The current step index of the plan"]
    current_task: Annotated[Union[TaskModel, None],
                            "The current task of the agent"]
    search_results: Annotated[List[str], "The search results of the agent"]
    summary: Annotated[str, "The summary of the agent"]
    final_answer: Annotated[str, "The final answer of the agent"]
