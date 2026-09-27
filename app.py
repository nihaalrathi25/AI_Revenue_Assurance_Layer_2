"""
app.py

Streamlit UI for Layer 2 - GenAI Revenue Assurance Copilot.

Shows:
  - The mock Layer 1 finding (invoice id, risk score, risk level,
    reason codes with their fixed business meaning)
  - A question box for the user
  - The LLM-generated explanation, clearly separated from the raw
    retrieved evidence that backs it up
"""

import streamlit as st

from src.rag_pipeline import RevenueAssuranceCopilot, load_layer1_output
from src import reason_codes as rc

st.set_page_config(page_title="AI Revenue Assurance Copilot", layout="centered")


@st.cache_resource(show_spinner="Connecting to knowledge base and LLM...")
def get_copilot():
    """Load the vector store + LLM once per Streamlit session/process."""
    return RevenueAssuranceCopilot()


st.title("AI Revenue Assurance Copilot")
st.caption("Layer 2 — GenAI explanation layer on top of the Layer 1 detection engine")

# ---------------------------------------------------------------------
# Load and display the mock Layer 1 finding
# ---------------------------------------------------------------------
layer1_data = load_layer1_output()

st.subheader(f"Invoice: {layer1_data.get('invoice_id', 'UNKNOWN')}")

col1, col2 = st.columns(2)
with col1:
    st.metric("Risk Score", layer1_data.get("risk_score", "N/A"))
with col2:
    st.metric("Risk Level", layer1_data.get("risk_level", "N/A"))

st.write(f"**Customer:** {layer1_data.get('customer', 'N/A')}")
st.write(f"**Contract ID:** {layer1_data.get('contract_id', 'N/A')}")

st.write("**Reason Codes:**")
for code in layer1_data.get("reason_codes", []):
    st.write(f"- `{code}` — {rc.describe_reason_code(code)}")

st.divider()

# ---------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------
st.subheader("Ask a question")

default_question = "Why is this invoice risky?"
user_question = st.text_input("Your question", value=default_question)

ask_clicked = st.button("Ask Copilot", type="primary")

st.divider()

# ---------------------------------------------------------------------
# Run the pipeline and display results
# ---------------------------------------------------------------------
if ask_clicked:
    if not user_question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            copilot = get_copilot()
        except Exception as e:
            st.error(
                "Could not connect to the vector store or LLM. "
                "Have you run `python ingest.py` and set your API key "
                "in a `.env` file?\n\n"
                f"Details: {e}"
            )
            st.stop()

        with st.spinner("Retrieving evidence and generating explanation..."):
            try:
                result = copilot.answer_question(layer1_data, user_question)
            except Exception as e:
                st.error(f"The Copilot failed to generate a response: {e}")
                st.stop()

        st.subheader("AI Explanation")
        st.markdown(result["answer"])

        st.divider()

        st.subheader("Retrieved Evidence")
        chunks = result["retrieved_chunks"]
        if not chunks:
            st.info("No relevant chunks were retrieved from the knowledge base.")
        else:
            for i, chunk in enumerate(chunks, start=1):
                with st.expander(f"Evidence {i} — Source: {chunk['source']}"):
                    st.text(chunk["content"])

        st.divider()

        st.subheader("Relevant Policy / Contract Sources")
        sources = sorted({chunk["source"] for chunk in chunks})
        if sources:
            for s in sources:
                st.write(f"- {s}")
        else:
            st.write("(none retrieved)")
