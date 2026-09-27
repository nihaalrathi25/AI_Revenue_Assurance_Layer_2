"""
llm.py

Factory function that returns a local Ollama chat LLM.
"""

from src import config


def get_llm():
    """
    Return a LangChain-compatible Ollama chat model.

    Ollama runs locally and does not require an API key.
    """

    provider = config.LLM_PROVIDER

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.LLM_MODEL_OLLAMA,
            temperature=config.LLM_TEMPERATURE,
            base_url=config.OLLAMA_BASE_URL,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama'."
    )