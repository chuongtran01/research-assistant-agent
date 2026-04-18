from memory.vector_store import vector_store


def store_memory(facts: list[str]) -> None:
    """
    Store the given facts in the memory database.
    """
    vector_store.add_texts(facts)
