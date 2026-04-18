from typing import List
import time

MEMORY_DB: List[dict] = []


def add_memory(memory: str) -> None:
    MEMORY_DB.append({
        "fact": memory,
        "timestamp": time.time()
    })


def retrieve_memories() -> List[str]:
    return MEMORY_DB
