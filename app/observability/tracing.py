import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

_LOGGER_NAME = "research_assistant"
_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    _CONFIGURED = True


def _logger() -> logging.Logger:
    configure_logging()
    return logging.getLogger(_LOGGER_NAME)


def new_run_id() -> str:
    return uuid4().hex[:12]


def preview_text(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _task_names(tasks: list[Mapping[str, Any]]) -> list[str]:
    return [
        str(task.get("name", "unknown"))
        for task in tasks[:5]
    ]


def _state_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    pending_tasks = state.get("pending_tasks", [])
    if not isinstance(pending_tasks, list):
        pending_tasks = []

    current_task = state.get("current_task")
    current_task_name = None
    if isinstance(current_task, Mapping):
        current_task_name = current_task.get("name")

    search_queries = state.get("search_queries", [])
    if not isinstance(search_queries, list):
        search_queries = []

    search_results = state.get("search_results", [])
    if not isinstance(search_results, list):
        search_results = []

    memory_context = state.get("memory_context", [])
    if not isinstance(memory_context, list):
        memory_context = []

    stored_facts = state.get("stored_facts", [])
    if not isinstance(stored_facts, list):
        stored_facts = []

    return {
        "run_id": state.get("run_id"),
        "current_task": current_task_name,
        "pending_tasks": _task_names(pending_tasks),
        "pending_task_count": len(pending_tasks),
        "memory_context_count": len(memory_context),
        "search_query_count": len(search_queries),
        "search_result_count": len(search_results),
        "stored_fact_count": len(stored_facts),
        "has_summary": bool(state.get("summary")),
    }


def _emit(event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    _logger().info(json.dumps(payload, default=str, sort_keys=True))


def trace_run_event(run_id: str, event: str, **fields: Any) -> None:
    _emit(event, run_id=run_id, scope="run", **fields)


def trace_node_event(
    state: Mapping[str, Any],
    node: str,
    event: str,
    **fields: Any,
) -> None:
    payload = _state_snapshot(state)
    payload.update(fields)
    _emit(event, scope="node", node=node, **payload)
