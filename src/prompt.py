"""
prompt.py

Builds the final prompt sent to the LLM, combining:
  - Layer 1 findings (invoice id, risk score, risk level, reason codes
    with their FIXED business-rule descriptions)
  - Retrieved evidence (chunks pulled from ChromaDB, each labelled
    with its source file)
  - The user's question

The system prompt instructs the LLM to answer ONLY from the supplied
evidence and to never invent contract clauses or numbers, and to
structure its answer into the five required output sections.
"""

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are the GenAI Revenue Assurance Copilot (Layer 2) for an \
industrial services company.

Your job is to EXPLAIN invoices that an upstream detection system \
(Layer 1) has already flagged as risky. You do NOT decide risk \
yourself, and you must never re-score or re-classify the invoice.

Rules you must follow:
1. Base your explanation only on the "Layer 1 Findings" and \
"Retrieved Evidence" given to you below. Do not use outside \
knowledge about contracts, pricing, or policies.
2. Never invent a contract clause, policy rule, or numeric value \
(rate, amount, percentage) that is not present in the retrieved \
evidence.
3. If the evidence does not contain enough information to determine \
a monetary revenue impact, say so plainly instead of guessing a \
number.
4. Use the reason code descriptions exactly as given; do not \
reinterpret what a reason code means.
5. Write in clear, plain business English, as if explaining to a \
revenue assurance analyst who has not yet read the invoice details.

Structure your answer using exactly these five headings, in this \
order:

### Risk Explanation
### Violated / Relevant Clause
### Evidence
### Revenue Impact
### Suggested Review Action

Under "Evidence", briefly quote or paraphrase the specific retrieved \
passage(s) you relied on, and mention which source document(s) they \
came from."""


def format_reason_codes_block(reason_codes: list[str], descriptions: dict) -> str:
    lines = []
    for code in reason_codes:
        lines.append(f"- {code}: {descriptions.get(code, 'Unknown reason code')}")
    return "\n".join(lines)


def format_evidence_block(chunks: list[Document]) -> str:
    if not chunks:
        return "(No relevant documents were retrieved from the knowledge base.)"

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown_source")
        lines.append(f"[Evidence {i} | Source: {source}]\n{chunk.page_content.strip()}")
    return "\n\n".join(lines)


def build_user_prompt(
    layer1_data: dict,
    reason_code_descriptions: dict,
    retrieved_chunks: list[Document],
    user_question: str,
) -> str:
    """Assemble the full user-turn prompt text shown in the spec."""
    reason_codes_block = format_reason_codes_block(
        layer1_data.get("reason_codes", []), reason_code_descriptions
    )
    evidence_block = format_evidence_block(retrieved_chunks)

    return f"""Layer 1 Findings:
Invoice ID: {layer1_data.get('invoice_id', 'UNKNOWN')}
Customer: {layer1_data.get('customer', 'UNKNOWN')}
Contract ID: {layer1_data.get('contract_id', 'UNKNOWN')}
Risk Score: {layer1_data.get('risk_score', 'UNKNOWN')}
Risk Level: {layer1_data.get('risk_level', 'UNKNOWN')}
Reason Codes:
{reason_codes_block}

Retrieved Evidence:
{evidence_block}

User Question:
{user_question}
"""


def build_messages(
    layer1_data: dict,
    reason_code_descriptions: dict,
    retrieved_chunks: list[Document],
    user_question: str,
) -> list[dict]:
    """Return a chat-style message list ready for a LangChain chat model."""
    user_prompt = build_user_prompt(
        layer1_data, reason_code_descriptions, retrieved_chunks, user_question
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
