"""
embeddings.py

Factory function that returns an embedding model, chosen by the
EMBEDDING_PROVIDER environment variable. This keeps the rest of the
codebase (vector_store.py, retriever.py) independent of which
embedding provider is actually in use.
"""

from src import config


def get_embedding_function():
    """
    Return a LangChain-compatible embedding object based on
    config.EMBEDDING_PROVIDER.

    Supported providers:
      - "huggingface": a small local sentence-transformers model
        (no API key needed, runs locally)
    """
    provider = config.EMBEDDING_PROVIDER

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_HUGGINGFACE)

    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider}'. Use 'huggingface'.")