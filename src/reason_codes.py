"""
reason_codes.py

Layer 1 emits short machine-readable "reason codes" (e.g. RATE_MISMATCH).
The meaning of each code is a BUSINESS FACT, not something the LLM
should be allowed to guess or paraphrase incorrectly. So we keep a
fixed, human-authored mapping here and hand the *already-decided*
meaning to the LLM as part of its context.

If a reason code is not found in this table, we say so explicitly
rather than letting the LLM invent an explanation for it.
"""

REASON_CODE_DESCRIPTIONS = {
    "RATE_MISMATCH": (
        "The invoice rate differs from the negotiated contract rate "
        "on file for this customer/contract."
    ),
    "CONTRACT_COVERAGE_MISMATCH": (
        "The service on this invoice appears to be covered by an "
        "active contract but has been billed separately."
    ),
    "JOB_CANCELLED_INVOICE_GENERATED": (
        "An invoice was generated for a work order that was marked "
        "as cancelled."
    ),
}

UNKNOWN_REASON_CODE_TEXT = (
    "No predefined business explanation is available for this reason "
    "code. It should be treated as unclassified and reviewed manually."
)


def describe_reason_code(code: str) -> str:
    """Return the fixed business meaning of a single reason code."""
    return REASON_CODE_DESCRIPTIONS.get(code, UNKNOWN_REASON_CODE_TEXT)


def describe_reason_codes(codes: list[str]) -> dict[str, str]:
    """Return {code: description} for a list of reason codes."""
    return {code: describe_reason_code(code) for code in codes}
