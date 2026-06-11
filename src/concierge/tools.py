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
    """Look up account information; returns masked identifiers — use verify_account_field to confirm caller-stated values."""
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
    return _redact_customer(customer)


def _redact_customer(customer: dict) -> dict:
    ssn = customer.get("ssn", "")
    masked_ssn = f"***-**-{ssn[-4:]}" if len(ssn) >= 4 else "***-**-****"
    cards = []
    for c in customer.get("credit_cards", []):
        num = (c.get("number") or "").replace(" ", "")
        last4 = num[-4:] if len(num) >= 4 else "****"
        cards.append({"brand": c.get("brand"), "last4": last4, "exp": c.get("exp")})
    return {
        "customer_id": customer["customer_id"],
        "name": customer["name"],
        "ssn_last4": masked_ssn,
        "phone_last4": (customer.get("phone") or "")[-4:],
        "email_masked": _mask_email(customer.get("email", "")),
        "credit_cards": cards,
        "accounts": [dict(a) for a in customer.get("accounts", [])],
    }


def _mask_email(email: str) -> str:
    if "@" not in email:
        return "***"
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


@tool
def verify_account_field(customer_id: str, field: str, value: str) -> dict:
    """Verify a caller-provided value against the stored record; returns only {'match': bool}."""
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise ValueError(f"No customer found with ID {customer_id!r}.")
    if field == "ssn":
        stored = customer.get("ssn", "")
        return {"match": stored == value}
    if field == "phone":
        stored = "".join(ch for ch in customer.get("phone", "") if ch.isdigit())
        normalized = "".join(ch for ch in value if ch.isdigit())
        return {"match": stored == normalized}
    if field == "email":
        stored = (customer.get("email") or "").lower()
        return {"match": stored == value.lower()}
    if field == "card_last4":
        last4s = [
            (c.get("number") or "").replace(" ", "")[-4:]
            for c in customer.get("credit_cards", [])
        ]
        return {"match": value in last4s}
    raise ValueError(f"Unsupported verification field {field!r}.")


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
