"""
rag_pipeline.py

Ties together every piece of Layer 2 into one function,
`answer_question`, that the Streamlit app (or any other caller) can
use:

  Layer 1 JSON  ->  reason code lookup
                                    \
  User question ->  retriever  -> evidence  -> prompt  -> LLM -> answer

This is the only module that "knows" the full pipeline order; every
other module does one job.
"""

import json
from pathlib import Path

from src import config
from src import reason_codes as rc
from src.embeddings import get_embedding_function
from src.vector_store import load_vector_store
from src.retriever import get_retriever, retrieve_evidence
from src.prompt import build_messages
from src.llm import get_llm


def load_layer1_output(path: Path = None) -> dict:
    """
    Load the mock Layer 1 JSON from disk. In production, this
    function is the ONLY thing that needs to change: swap the file
    read for a call to the real Layer 1 API/database, keeping the
    same dict shape (invoice_id, risk_score, risk_level,
    reason_codes, ...).
    """
    layer1_path = path or config.LAYER1_OUTPUT_PATH
    with open(layer1_path, "r", encoding="utf-8") as f:
        return json.load(f)


class RevenueAssuranceCopilot:
    """
    Thin wrapper that loads the vector store and LLM once, so a
    Streamlit app can reuse the same instance across user questions
    instead of reconnecting to ChromaDB every time.
    """

    def __init__(self):
        embedding_function = get_embedding_function()
        self.vectorstore = load_vector_store(embedding_function)
        self.retriever = get_retriever(self.vectorstore)
        self.llm = get_llm()

    def answer_question(self, layer1_data: dict, user_question: str) -> dict:
        """
        Run the full Layer 2 pipeline for one question and return a
        dict with the raw LLM answer plus the retrieved evidence, so
        the UI can display them as separate, clearly labelled
        sections.
        """
        reason_code_list = layer1_data.get("reason_codes", [])
        reason_code_descriptions = rc.describe_reason_codes(reason_code_list)

        retrieved_chunks = retrieve_evidence(
            retriever=self.retriever,
            user_question=user_question,
            reason_codes=reason_code_list,
            reason_code_descriptions=reason_code_descriptions,
            contract_id=layer1_data.get("contract_id"),
        )

        messages = build_messages(
            layer1_data=layer1_data,
            reason_code_descriptions=reason_code_descriptions,
            retrieved_chunks=retrieved_chunks,
            user_question=user_question,
        )

        response = self.llm.invoke(messages)
        answer_text = getattr(response, "content", str(response))

        return {
            "answer": answer_text,
            "reason_code_descriptions": reason_code_descriptions,
            "retrieved_chunks": [
                {
                    "source": chunk.metadata.get("source", "unknown_source"),
                    "content": chunk.page_content,
                }
                for chunk in retrieved_chunks
            ],
        }
