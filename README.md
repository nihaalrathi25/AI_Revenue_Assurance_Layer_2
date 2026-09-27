# Layer 2 — GenAI Revenue Assurance Copilot

A RAG-based GenAI layer that sits on top of an (upstream, mocked) invoice
risk-detection engine and **explains** why a flagged invoice is risky, in
plain business language, grounded in real contract and policy text.

> **Scope note:** This repo implements Layer 2 only. Layer 1 (rules/ML
> detection) is mocked with a fixed JSON file. Layer 3 (agentic extension)
> and Layer 4 (executive dashboard) are out of scope and not built here.

---

## 1. What Layer 2 does (and does not do)

- Layer 1 **detects** risky invoices and outputs a risk score, risk level,
  and short reason codes (e.g. `RATE_MISMATCH`).
- Layer 2 **explains** that finding: it retrieves the relevant contract /
  policy text from a knowledge base and asks an LLM to turn "Layer 1
  findings + retrieved evidence" into a human-readable explanation.
- Layer 2 does **not** recompute risk, does **not** re-run any rule engine
  or ML model, and the LLM is **not allowed to invent** contract clauses,
  reason-code meanings, or monetary figures that aren't in the retrieved
  evidence.

---

## 2. Project structure

```
layer2_revenue_copilot/
│
├── data/
│   ├── contracts/               # dummy contracts (C001, C002)
│   ├── policies/                # dummy billing/pricing/cancellation/RA policies
│   └── layer1_output.json       # fixed mock Layer 1 output (NOT randomly generated)
│
├── vectorstore/                 # ChromaDB persisted data (created by ingest.py)
│
├── src/
│   ├── config.py                # all paths + env-driven settings in one place
│   ├── reason_codes.py          # FIXED reason-code -> meaning lookup (not LLM-generated)
│   ├── document_loader.py       # loads .txt files into LangChain Documents
│   ├── embeddings.py            # embedding-model factory (OpenAI / HuggingFace)
│   ├── vector_store.py          # chunking + build/load ChromaDB
│   ├── retriever.py             # retriever + query construction
│   ├── prompt.py                # system + user prompt templates
│   ├── llm.py                   # LLM factory (OpenAI / Anthropic)
│   └── rag_pipeline.py          # orchestrates the whole pipeline
│
├── app.py                       # Streamlit UI
├── ingest.py                    # one-off script: build the ChromaDB vector store
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Architecture / data flow

```
Mock Layer 1 JSON  ──────────────┐
(invoice id, risk score,         │
 risk level, reason codes)       │
                                  ▼
                        ┌──────────────────┐
                        │  Layer 2 Copilot │
                        └──────────────────┘
                                  │
             ┌────────────────────┴────────────────────┐
             ▼                                          ▼
   Reason Code Interpretation                 RAG Knowledge Base
   (fixed dict in reason_codes.py)     (contracts + policies, .txt files)
             │                                          │
             │                                    document_loader.py
             │                                          │
             │                                    vector_store.py
             │                                    (chunk + embed)
             │                                          │
             │                                     ChromaDB (on disk)
             │                                          │
             │                                    retriever.py
             │                                          │
             └────────────────────┬─────────────────────┘
                                  ▼
                        prompt.py builds:
                Layer 1 findings + retrieved evidence + user question
                                  │
                                  ▼
                           llm.py → LLM call
                                  │
                                  ▼
                Layer 2 Output: Risk Explanation, Violated Clause,
                Evidence, Revenue Impact, Suggested Review Action
```

---

## 4. Setup

### 4.1 Prerequisites

- Python 3.10+
- An API key for at least one LLM provider (OpenAI or Anthropic)

### 4.2 Install dependencies

```bash
cd layer2_revenue_copilot
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.3 Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...your key...
EMBEDDING_PROVIDER=openai
```

**No OpenAI account?** Set `EMBEDDING_PROVIDER=huggingface` — this runs a
small local embedding model with no API key, so you only need an LLM key.

### 4.4 Build the vector store (run once)

```bash
python ingest.py
```

This loads every file in `data/contracts/` and `data/policies/`, splits
them into chunks, embeds them, and persists them into a ChromaDB
collection under `vectorstore/`. Re-run this any time you add or edit a
knowledge-base document.

### 4.5 Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). You'll
see the mocked invoice `INV001`, its risk score/level, and its reason
codes. Type a question (a default one is pre-filled) and click
**Ask Copilot**.

---

## 5. Example

**Question:** "Why is invoice INV001 risky?"

**Expected style of answer** (actual wording is generated by the LLM,
but the facts come only from `layer1_output.json` and the retrieved
documents):

```
### Risk Explanation
Invoice INV001 was flagged HIGH risk (score 87) for two reasons: the
billed rate differs from the negotiated contract rate, and the service
appears to already be covered under an active contract.

### Violated / Relevant Clause
Contract C001's Billing Rules clause: services covered under the active
maintenance contract must not be billed separately unless explicitly
approved.

### Evidence
[Quotes/paraphrases from contract_C001.txt and pricing_policy.txt]

### Revenue Impact
Not precisely quantifiable from the available documents beyond the
standard INR 5000/hour repair labor rate stated in the contract.

### Suggested Review Action
Hold the invoice and route it to the revenue assurance team per the
Revenue Assurance Policy before release.
```

---

## 6. How each file fits together

| File | Responsibility |
|---|---|
| `data/layer1_output.json` | Fixed mock output standing in for the real Layer 1 detection engine. |
| `data/contracts/`, `data/policies/` | Dummy but realistic business documents used as the RAG knowledge base. |
| `src/config.py` | Single source of truth for paths, model names, and tunable parameters, all overridable via `.env`. |
| `src/reason_codes.py` | Hard-coded mapping from reason code → business meaning. Kept separate from the LLM so the meaning of a code is never "guessed." |
| `src/document_loader.py` | Reads the `.txt` files off disk into LangChain `Document` objects, tagging each with its source filename. |
| `src/embeddings.py` | Returns an embedding model (OpenAI or local HuggingFace) based on `EMBEDDING_PROVIDER`. |
| `src/vector_store.py` | Splits documents into chunks and builds/loads the ChromaDB collection. |
| `src/retriever.py` | Wraps ChromaDB as a retriever and builds a richer search query using the reason codes, not just the raw question. |
| `src/prompt.py` | Builds the system prompt (rules for the LLM) and the user prompt (Layer 1 findings + evidence + question). |
| `src/llm.py` | Returns a chat model (OpenAI or Anthropic) based on `LLM_PROVIDER`. |
| `src/rag_pipeline.py` | Orchestrates the full flow: load Layer 1 JSON → retrieve → build prompt → call LLM → return structured result. |
| `ingest.py` | One-off script that runs the document → chunk → embed → ChromaDB pipeline. |
| `app.py` | Streamlit UI: shows the invoice, takes a question, displays the AI explanation separately from the raw retrieved evidence. |

---

## 7. How retrieval works

1. `ingest.py` loads all `.txt` files, splits them into ~500-character
   overlapping chunks, embeds each chunk, and stores the vectors (plus
   original text and source filename) in a ChromaDB collection on disk.
2. At query time, `retriever.py` builds a combined search string: the
   user's raw question **plus** the plain-English meaning of each Layer 1
   reason code **plus** the contract ID (if known). This keeps retrieval
   accurate even when the user's question is generic (e.g. "Why is this
   risky?").
3. ChromaDB returns the top-k most similar chunks, each still tagged with
   its source file, which the UI later displays as "Retrieved Evidence."

## 8. How Layer 1 findings and retrieved documents are combined

`prompt.py` assembles one prompt with three clearly separated sections —
`Layer 1 Findings`, `Retrieved Evidence`, and `User Question` — exactly as
specified in the project brief. The system prompt instructs the LLM to
answer only from those two inputs and never introduce outside facts.

## 9. How the LLM generates the final explanation

The LLM receives the assembled prompt and is instructed (via the system
prompt in `prompt.py`) to structure its answer into five fixed sections:
Risk Explanation, Violated/Relevant Clause, Evidence, Revenue Impact, and
Suggested Review Action. It is explicitly told not to invent clauses,
numbers, or reason-code meanings — those all come from
`reason_codes.py` and the retrieved chunks, not from the model's own
knowledge.

## 10. What the final output contains

The Streamlit UI renders, as separate sections:

- **AI Explanation** — the LLM's structured answer (five sections above).
- **Retrieved Evidence** — the raw chunks that were fed to the LLM, each
  labelled with its source document, so a human can verify the
  explanation against the actual source text.
- **Relevant Policy / Contract Sources** — a deduplicated list of the
  source filenames used.

## 11. Replacing the mock Layer 1 JSON with the real Layer 1 output later

Everything downstream depends only on the dict shape returned by
`load_layer1_output()` in `src/rag_pipeline.py`:

```python
{
  "invoice_id": "...",
  "customer": "...",
  "contract_id": "...",
  "risk_score": <int>,
  "risk_level": "...",
  "reason_codes": ["...", "..."]
}
```

To integrate the real Layer 1 engine, replace the body of
`load_layer1_output()` with a call to the real Layer 1 API, message
queue, or database — returning a dict in the same shape — and nothing
else in Layer 2 needs to change. `app.py` and `rag_pipeline.py` already
treat this dict as an external input, not something Layer 2 computes.

---

## 12. Switching LLM / embedding providers

Both are controlled purely by environment variables — no code changes:

```env
LLM_PROVIDER=openai        # or: anthropic
EMBEDDING_PROVIDER=openai  # or: huggingface
```

`src/llm.py` and `src/embeddings.py` are the only two files that know
about specific provider SDKs; everything else uses the standard LangChain
chat-model / embeddings interface.

---

## 13. Notes / limitations (by design, for this prototype)

- All data is dummy/fictional, created for development and demo purposes.
- The risk score/level/reason codes are fixed in `data/layer1_output.json`
  and are never generated or altered by Layer 2.
- No agentic behavior, no Power BI, no production architecture — this is
  intentionally a focused, single-layer prototype.
