"""
document_loader.py

Loads the dummy knowledge base (contracts + policies) off disk into
LangChain Document objects. Each Document keeps track of which file
it came from (metadata["source"]) so that later, when we retrieve
chunks, we can tell the user exactly which document a piece of
evidence came from.
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src import config


def _load_txt_files_from_dir(directory: Path, doc_type: str) -> list[Document]:
    """Load every .txt file in a directory as a separate Document."""
    documents: list[Document] = []

    if not directory.exists():
        return documents

    for file_path in sorted(directory.glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded = loader.load()  # returns a list with one Document
        for doc in loaded:
            doc.metadata["source"] = file_path.name
            doc.metadata["doc_type"] = doc_type
        documents.extend(loaded)

    return documents


def load_all_documents() -> list[Document]:
    """
    Load every document in the dummy knowledge base:
    - all contract files under data/contracts/
    - all policy files under data/policies/

    Returns a flat list of LangChain Document objects ready for
    chunking.
    """
    contract_docs = _load_txt_files_from_dir(config.CONTRACTS_DIR, "contract")
    policy_docs = _load_txt_files_from_dir(config.POLICIES_DIR, "policy")

    all_docs = contract_docs + policy_docs

    if not all_docs:
        raise FileNotFoundError(
            f"No knowledge base documents found in "
            f"{config.CONTRACTS_DIR} or {config.POLICIES_DIR}."
        )

    return all_docs
