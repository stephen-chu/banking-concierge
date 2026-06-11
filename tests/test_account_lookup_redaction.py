"""Tests that account_lookup never surfaces unmasked PII and that
verify_account_field returns only a boolean match."""

from __future__ import annotations

import json
import re

import pytest

from concierge.tools import account_lookup, verify_account_field


SSN_PATTERN = re.compile(r"\d{3}-\d{2}-\d{4}")
LONG_DIGIT_RUN = re.compile(r"\d{13,19}")


def test_account_lookup_redacts_pii():
    result = account_lookup.invoke({"customer_id": "CUST-0002"})
    serialized = json.dumps(result)

    assert SSN_PATTERN.search(serialized) is None
    assert LONG_DIGIT_RUN.search(serialized) is None
    assert "cvv" not in serialized.lower()
    assert result["ssn_last4"] == "***-**-8810"
    assert all("number" not in card and "cvv" not in card for card in result["credit_cards"])
    assert {c["last4"] for c in result["credit_cards"]} == {"4444", "1881"}


def test_account_lookup_masks_contact_fields():
    result = account_lookup.invoke({"customer_id": "CUST-0001"})
    assert result["phone_last4"] == "0142"
    assert result["email_masked"] == "a***@example.com"
    assert result["name"] == "Alex Rivera"
    assert len(result["accounts"]) == 2


def test_verify_account_field_ssn_match_and_mismatch():
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0001", "field": "ssn", "value": "552-19-4488"}
    ) == {"match": True}
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0001", "field": "ssn", "value": "000-00-0000"}
    ) == {"match": False}


def test_verify_account_field_phone_normalizes_digits():
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0001", "field": "phone", "value": "4155550142"}
    ) == {"match": True}
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0001", "field": "phone", "value": "415-555-0142"}
    ) == {"match": True}
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0001", "field": "phone", "value": "4155550000"}
    ) == {"match": False}


def test_verify_account_field_email_case_insensitive():
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0001", "field": "email", "value": "ALEX.RIVERA@example.com"}
    ) == {"match": True}
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0001", "field": "email", "value": "wrong@example.com"}
    ) == {"match": False}


def test_verify_account_field_card_last4():
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0002", "field": "card_last4", "value": "4444"}
    ) == {"match": True}
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0002", "field": "card_last4", "value": "1881"}
    ) == {"match": True}
    assert verify_account_field.invoke(
        {"customer_id": "CUST-0002", "field": "card_last4", "value": "9999"}
    ) == {"match": False}


def test_verify_account_field_unsupported_field_raises():
    with pytest.raises(ValueError):
        verify_account_field.invoke(
            {"customer_id": "CUST-0001", "field": "address", "value": "anything"}
        )
