"""
vector_store.py

Responsible for:
  1. Splitting loaded documents into smaller chunks.
  2. Building a new ChromaDB vector store from those chunks
     (used by ingest.py).
  3. Loading an already-built ChromaDB vector store from disk
     (used by app.py / rag_pipeline.py at query time).
"""

import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config


def split_documents(documents: list[Document]) -> list[Document]:
    """Split full documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(documents: list[Document], embedding_function) -> Chroma:
    """
    Chunk the given documents, embed them, and persist them to a new
    ChromaDB collection on disk. This is the "ingestion" step and
    should be run whenever the dummy knowledge base changes.

    Any previously persisted store is removed first, since a store
    built with a different embedding model/dimensionality cannot be
    safely reused for a new embedding model.
    """
    if config.VECTORSTORE_DIR.exists():
        shutil.rmtree(config.VECTORSTORE_DIR)

    chunks = split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    return vectorstore


def load_vector_store(embedding_function) -> Chroma:
    """
    Load an existing ChromaDB collection from disk. Used at query
    time so we don't have to re-embed the knowledge base on every
    Streamlit run.
    """
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    return vectorstore