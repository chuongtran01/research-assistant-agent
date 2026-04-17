import os
import json
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, PrivateAttr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


DEFAULT_SYSTEM_PROMPT = """
You are a research assistant.

Follow instructions carefully.
Be precise and concise.
"""


class LLM(BaseModel):
    """
    Gemini-based LLM wrapper for LangGraph agents.

    Features:
    - simple chat interface
    - structured JSON output (for planner/router)
    - Gemini-compatible (no system role)
    - clean abstraction layer
    """

    _api_key: str = PrivateAttr()
    _model: ChatGoogleGenerativeAI = PrivateAttr()
    _system_prompt: str = PrivateAttr()

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str = "gemini-2.5-flash",
        structured_output: Optional[type[BaseModel]] = None,
        api_key: Optional[str] = None
    ):
        super().__init__()

        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self._model = ChatGoogleGenerativeAI(
            model=model,
            api_key=self._api_key
        )

        if structured_output:
            self._model = self._model.with_structured_output(
                structured_output, method="json_schema")

        self._system_prompt = system_prompt.strip()

    # ------------------------
    # Structured Output (CRITICAL)
    # ------------------------

    def structured_chat(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        response = self._model.invoke([
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt)
        ])

        return response
