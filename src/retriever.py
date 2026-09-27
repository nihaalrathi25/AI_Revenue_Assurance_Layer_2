"""
retriever.py

Wraps the ChromaDB vector store as a retriever, and builds the
search query used to fetch evidence. We deliberately widen the raw
user question with the Layer 1 reason codes and contract ID (when
available), because a question like "Why is this invoice risky?"
alone is too generic to reliably retrieve the right contract clause.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config


def get_retriever(vectorstore: Chroma, top_k: int = None):
    """Return a LangChain retriever backed by the given vector store."""
    k = top_k or config.RETRIEVER_TOP_K
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_retrieval_query(
    user_question: str,
    reason_codes: list[str],
    reason_code_descriptions: dict,
    contract_id: str | None = None,
) -> str:
    """
    Combine the user's raw question with Layer 1 signals so the
    retriever has enough context to find the right chunks, even if
    the user's question is vague.
    """
    parts = [user_question]

    if contract_id:
        parts.append(f"Contract {contract_id}")

    for code in reason_codes:
        parts.append(reason_code_descriptions.get(code, code))

    return " | ".join(parts)


def retrieve_evidence(
    retriever,
    user_question: str,
    reason_codes: list[str],
    reason_code_descriptions: dict,
    contract_id: str | None = None,
) -> list[Document]:
    """
    Run retrieval and return the list of relevant chunks (as
    Document objects, each carrying source metadata).
    """
    query = build_retrieval_query(
        user_question, reason_codes, reason_code_descriptions, contract_id
    )
    return retriever.invoke(query)
