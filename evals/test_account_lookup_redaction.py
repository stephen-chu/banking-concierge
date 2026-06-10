"""Regression test: account_lookup must never return SSN, full PAN, or CVV."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from concierge.tools import account_lookup  # noqa: E402


SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
PAN_RE = re.compile(r"(?:\d[ -]?){16}")


def _contains_cvv_key(obj) -> bool:
    if isinstance(obj, dict):
        if any(k.lower() == "cvv" for k in obj.keys()):
            return True
        return any(_contains_cvv_key(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_cvv_key(v) for v in obj)
    return False


def test_account_lookup_redacts_sensitive_fields():
    result = account_lookup.invoke({"customer_id": "CUST-0001"})
    blob = json.dumps(result)
    assert not SSN_RE.search(blob), f"SSN leaked: {blob}"
    assert not PAN_RE.search(blob), f"Full PAN leaked: {blob}"
    assert not _contains_cvv_key(result), f"cvv key present: {result}"
    assert "ssn" not in {k.lower() for k in result.keys()}
