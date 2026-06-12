"""Tools available to the Meridian National customer service concierge agent.

A few tools have deliberate rough edges so LangSmith Engine has something
to cluster after the load generator runs:

- search_banking_docs has a vague description so the model occasionally
  re-queries multiple times rephrasing
- account_lookup raises on malformed customer IDs and on IDs prefixed with
  "X" (simulated downstream outage)
- recent_transactions raises if the model passes a runaway limit
- find_branch raises on non-zip inputs
"""

from __future__ import annotations

from langchain_core.tools import tool

from concierge.mock_data import (
    BRANCHES,
    CUSTOMERS,
    TRANSACTIONS,
    find_branch_by_zip,
)
from concierge.retrieval import retrieve


@tool
def search_banking_docs(query: str, k: int = 4) -> str:
    """Search Meridian National banking documentation.

    Args:
        query: A natural-language search query.
        k: Number of relevant chunks to return. Defaults to 4.
    """
    chunks = retrieve(query, k=k)
    if not chunks:
        return "No relevant documentation found."
    blocks = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


@tool
def account_lookup(customer_id: str) -> dict:
    """Look up account information.

    Returns the customer's name and a list of their account IDs, account
    types, and balances. Use this when the user wants details about an
    account.
    """
    if customer_id.startswith("X"):
        raise RuntimeError(
            "Customer record service is temporarily unavailable. Try again later."
        )
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise ValueError(
            f"No customer found with ID {customer_id!r}. "
            "Customer IDs are in the format CUST-####."
        )
    return dict(customer)


_VERIFY_FIELDS = {"ssn", "phone", "email", "card", "card_number", "address", "dob", "date_of_birth"}


def _normalize_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


@tool
def verify_account_field(customer_id: str, field: str, value: str) -> dict:
    """Return {"match": bool} for whether the provided value matches the on-file field; never returns the on-file value."""
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise ValueError(
            f"No customer found with ID {customer_id!r}. "
            "Customer IDs are in the format CUST-####."
        )
    key = field.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _VERIFY_FIELDS:
        raise ValueError(
            f"Unsupported field {field!r}. Supported: ssn, phone, email, card, address, date_of_birth."
        )
    provided = (value or "").strip()
    match = False
    if key == "ssn":
        match = _normalize_digits(provided) == _normalize_digits(customer.get("ssn", ""))
    elif key == "phone":
        match = _normalize_digits(provided) == _normalize_digits(customer.get("phone", ""))
    elif key == "email":
        match = provided.lower() == customer.get("email", "").lower()
    elif key in {"card", "card_number"}:
        provided_digits = _normalize_digits(provided)
        match = any(
            provided_digits == _normalize_digits(card.get("number", ""))
            for card in customer.get("credit_cards", [])
        )
    else:
        match = False
    return {"match": bool(match)}


@tool
def recent_transactions(customer_id: str, limit: int = 5) -> list[dict]:
    """Retrieve a customer's most recent transactions.

    Args:
        customer_id: The customer ID (e.g. CUST-0001).
        limit: Optional number of transactions to return.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > 50:
        raise ValueError(
            f"limit {limit} exceeds the maximum of 50. Pick a smaller number."
        )
    if customer_id not in CUSTOMERS:
        raise ValueError(
            f"No customer found with ID {customer_id!r}. "
            "Customer IDs are in the format CUST-####."
        )
    txs = TRANSACTIONS.get(customer_id, [])
    return [dict(t) for t in txs[:limit]]


@tool
def find_branch(zip_code: str) -> dict:
    """Find a Meridian National branch.

    Args:
        zip_code: A 5-digit U.S. ZIP code.
    """
    if not (isinstance(zip_code, str) and len(zip_code) == 5 and zip_code.isdigit()):
        raise ValueError(
            f"zip_code must be a 5-digit U.S. ZIP code. Got {zip_code!r}."
        )
    branch = find_branch_by_zip(zip_code)
    if branch is None:
        return {
            "match": False,
            "message": "No Meridian National branch found in our directory for that ZIP code.",
            "nearest_known": BRANCHES[0],
        }
    return {"match": True, **branch}


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    """Initiate a transfer between two Meridian National accounts owned by the same customer.

    Args:
        from_account: The source account ID.
        to_account: The destination account ID.
        amount: The dollar amount to transfer.
    """
    if amount <= 0:
        raise ValueError("amount must be positive")
    confirmation = f"MNB-XFER-{abs(hash((from_account, to_account, amount))) % 10_000_000:07d}"
    return {
        "status": "submitted",
        "from_account": from_account,
        "to_account": to_account,
        "amount": round(amount, 2),
        "confirmation": confirmation,
        "estimated_post": "immediately",
    }


TOOLS = [
    search_banking_docs,
    account_lookup,
    verify_account_field,
    recent_transactions,
    find_branch,
    transfer_funds,
]
