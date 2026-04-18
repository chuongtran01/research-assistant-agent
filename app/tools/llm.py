import os
from time import perf_counter
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, PrivateAttr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from observability.tracing import trace_run_event

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
    _model_name: str = PrivateAttr()
    _structured_output_name: str = PrivateAttr()

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
        self._model_name = model
        self._structured_output_name = (
            structured_output.__name__ if structured_output else "none"
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
        trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        trace = trace or {}
        run_id = trace.get("run_id")
        event_fields = {
            "node": trace.get("node"),
            "operation": trace.get("operation"),
            "model": self._model_name,
            "output_schema": self._structured_output_name,
            "prompt_chars": len(prompt),
        }

        started_at = perf_counter()
        if run_id:
            trace_run_event(run_id, "llm_call_started", **event_fields)

        try:
            response = self._model.invoke([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=prompt)
            ])
        except Exception as exc:
            if run_id:
                trace_run_event(
                    run_id,
                    "llm_call_failed",
                    duration_ms=round((perf_counter() - started_at) * 1000, 2),
                    error=str(exc),
                    **event_fields,
                )
            raise

        if run_id:
            trace_run_event(
                run_id,
                "llm_call_completed",
                duration_ms=round((perf_counter() - started_at) * 1000, 2),
                response_type=type(response).__name__,
                **event_fields,
            )

        return response
