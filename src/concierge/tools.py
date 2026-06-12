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
    """Look up a customer record by Meridian National customer ID.

    `customer_id` MUST be in the `CUST-####` format (for example
    `CUST-0001`). This tool does NOT search by SSN, phone, email, or card
    number — for those identifiers use `find_customer` instead. NEVER
    iterate or guess sequential customer IDs; if `find_customer` returns
    no exact match, tell the rep and stop.
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


@tool
def find_customer(identifier: str, identifier_type: str) -> dict:
    """Look up a customer by SSN, phone, email, or credit-card last 4.

    Use this whenever the rep gives you anything OTHER than a `CUST-####`
    ID. Performs an exact-match search and returns the matching customer
    record, or raises if no record matches exactly. Never returns a
    "closest" or "partial" match.

    Args:
        identifier: The value to search for (e.g. `552-19-4488`,
            `(415) 555-0142`, `alex.rivera@example.com`, or the last 4
            digits of a card like `4242`).
        identifier_type: One of `ssn`, `phone`, `email`, or `card_last4`.
    """
    field_map = {
        "ssn": lambda c: c["ssn"],
        "phone": lambda c: c["phone"],
        "email": lambda c: c["email"],
        "card_last4": lambda c: [card["number"][-4:] for card in c["credit_cards"]],
    }
    if identifier_type not in field_map:
        raise ValueError(
            f"identifier_type must be one of {sorted(field_map)}, got {identifier_type!r}."
        )
    extract = field_map[identifier_type]
    for customer in CUSTOMERS.values():
        value = extract(customer)
        if isinstance(value, list):
            if identifier in value:
                return dict(customer)
        elif value == identifier:
            return dict(customer)
    raise ValueError(
        f"No customer found with {identifier_type}={identifier!r}. "
        "Confirm the value with the rep — do not guess or substitute another customer."
    )


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
    find_customer,
    recent_transactions,
    find_branch,
    transfer_funds,
]
