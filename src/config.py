"""
config.py

Central place for all paths and environment-driven settings.
Keeping this in one file makes it easy to see (and change) every
"knob" of the system without hunting through the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CONTRACTS_DIR = DATA_DIR / "contracts"
POLICIES_DIR = DATA_DIR / "policies"
LAYER1_OUTPUT_PATH = DATA_DIR / "layer1_output.json"

VECTORSTORE_DIR = Path(
    os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "vectorstore"))
)
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "revenue_assurance_kb")

# ---------------------------------------------------------------------
# LLM provider settings
# ---------------------------------------------------------------------
# Supported values: "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

LLM_MODEL_OLLAMA = os.getenv(
    "LLM_MODEL_OLLAMA",
    "llama3.2:3b",
)

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------
# Embedding provider settings
# ---------------------------------------------------------------------
# Supported values: "huggingface"
# "huggingface" runs a small local Sentence Transformers model and
# needs no API key.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
EMBEDDING_MODEL_HUGGINGFACE = os.getenv(
    "EMBEDDING_MODEL_HUGGINGFACE", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------
# RAG / retrieval settings
# ---------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))