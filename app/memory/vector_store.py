from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Persistent vector store
vector_store = Chroma(
    collection_name="memory",
    embedding_function=embeddings,
    persist_directory="chroma_db")
