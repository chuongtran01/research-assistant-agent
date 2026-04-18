from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

PERSIST_DIRECTORY = Path(__file__).resolve().parents[1] / "chroma_db"

# Persistent vector store
vector_store = Chroma(
    collection_name="memory",
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIRECTORY))
