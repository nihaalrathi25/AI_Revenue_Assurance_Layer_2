"""
ingest.py

Run this script ONCE (and again any time the dummy knowledge base
documents change) to build the ChromaDB vector store:

    python ingest.py

Pipeline:
    Documents -> load -> split into chunks -> embed -> store in ChromaDB
"""

from src.document_loader import load_all_documents
from src.embeddings import get_embedding_function
from src.vector_store import build_vector_store, split_documents
from src import config


def main():
    print("Step 1/4: Loading documents from data/contracts and data/policies ...")
    documents = load_all_documents()
    print(f"  Loaded {len(documents)} documents.")

    print("Step 2/4: Splitting documents into chunks ...")
    chunks = split_documents(documents)
    print(f"  Created {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print(f"Step 3/4: Loading embedding function (provider='{config.EMBEDDING_PROVIDER}') ...")
    embedding_function = get_embedding_function()

    print(f"Step 4/4: Embedding chunks and persisting to ChromaDB at "
          f"'{config.VECTORSTORE_DIR}' ...")
    build_vector_store(documents, embedding_function)

    print("\nDone. The vector store is ready.")
    print("You can now run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
