import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from concierge.tools import account_lookup

SSN_RE = re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b")
PAN_RE = re.compile(r"(?:\d[ -]?){16}")


def _contains_cvv_key(obj):
    if isinstance(obj, dict):
        if any(k.lower() == "cvv" for k in obj):
            return True
        return any(_contains_cvv_key(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_cvv_key(v) for v in obj)
    return False


def test_account_lookup_redacts_sensitive_fields():
    result = account_lookup.invoke({"customer_id": "CUST-0001"})
    blob = json.dumps(result)
    assert not SSN_RE.search(blob)
    assert not PAN_RE.search(blob)
    assert not _contains_cvv_key(result)
