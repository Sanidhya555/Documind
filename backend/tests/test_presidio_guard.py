"""
Unit tests for the Presidio Layer 1 regex guardrail.

Tests cover:
- Each target entity type is correctly detected and raises HTTP 400
- Clean text passes through unchanged
- The error body contains the correct violated_entities list
- Multiple entities in one string are all reported
- Strings that look similar to PII but aren't are not blocked (false-positive check)
"""
import pytest
from fastapi import HTTPException


# ── The stdlib fallback is tested even when Presidio is not installed ──────────
from utils.presidio_guard import enterprise_regex_guard


# ── Helper ────────────────────────────────────────────────────────────────────

def _raises_400_with_entity(text: str, entity: str):
    """Assert that text triggers a 400 containing the given entity type."""
    with pytest.raises(HTTPException) as exc_info:
        enterprise_regex_guard(text)
    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert "violated_entities" in detail, f"Missing violated_entities in detail: {detail}"
    assert entity in detail["violated_entities"], (
        f"Expected {entity!r} in {detail['violated_entities']} for input: {text!r}"
    )


def _passes(text: str):
    """Assert that text passes through the guard unchanged."""
    result = enterprise_regex_guard(text)
    assert result == text, f"Guard unexpectedly modified or blocked: {text!r}"


# ── PII detection: one entity per test ───────────────────────────────────────

class TestPIIDetection:
    def test_credit_card_detected(self):
        _raises_400_with_entity("My card number is 4532015112830366", "CREDIT_CARD")

    def test_credit_card_visa(self):
        _raises_400_with_entity("Pay with 4111111111111111 please", "CREDIT_CARD")

    def test_email_detected(self):
        _raises_400_with_entity("Send the report to john.doe@company.com", "EMAIL_ADDRESS")

    def test_email_in_question(self):
        _raises_400_with_entity(
            "What documents were sent to admin@enterprise.org?", "EMAIL_ADDRESS"
        )

    def test_ip_address_detected(self):
        _raises_400_with_entity("Server is running on 192.168.1.100", "IP_ADDRESS")

    def test_ip_address_external(self):
        _raises_400_with_entity("Connect to 203.0.113.45 for access", "IP_ADDRESS")

    def test_crypto_bitcoin_detected(self):
        _raises_400_with_entity(
            "Send funds to 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "CRYPTO"
        )

    def test_ssn_detected(self):
        _raises_400_with_entity("My SSN is 123-45-6789", "US_SSN")

    def test_ssn_with_spaces(self):
        _raises_400_with_entity("Social security 078 05 1120", "US_SSN")

    def test_phone_number_detected(self):
        _raises_400_with_entity("Call me at (555) 867-5309 anytime", "PHONE_NUMBER")


# ── Multi-entity detection ────────────────────────────────────────────────────

class TestMultiEntityDetection:
    def test_email_and_credit_card_together(self):
        text = "Email: user@acme.com  Card: 4532015112830366"
        with pytest.raises(HTTPException) as exc_info:
            enterprise_regex_guard(text)
        detail = exc_info.value.detail
        entities = detail["violated_entities"]
        assert "EMAIL_ADDRESS" in entities
        assert "CREDIT_CARD" in entities

    def test_ssn_and_ip_together(self):
        text = "SSN 123-45-6789 from IP 10.0.0.1"
        with pytest.raises(HTTPException) as exc_info:
            enterprise_regex_guard(text)
        detail = exc_info.value.detail
        entities = detail["violated_entities"]
        assert "US_SSN" in entities


# ── Error body structure ──────────────────────────────────────────────────────

class TestErrorStructure:
    def test_error_body_has_required_keys(self):
        with pytest.raises(HTTPException) as exc_info:
            enterprise_regex_guard("user@example.com")
        detail = exc_info.value.detail
        assert "error" in detail
        assert "violated_entities" in detail
        assert isinstance(detail["violated_entities"], list)
        assert len(detail["violated_entities"]) > 0

    def test_error_message_is_descriptive(self):
        with pytest.raises(HTTPException) as exc_info:
            enterprise_regex_guard("user@example.com")
        assert "sensitive data" in exc_info.value.detail["error"].lower()

    def test_status_code_is_400(self):
        with pytest.raises(HTTPException) as exc_info:
            enterprise_regex_guard("4532015112830366")
        assert exc_info.value.status_code == 400

    def test_violated_entities_sorted(self):
        text = "user@acme.com  Card: 4532015112830366"
        with pytest.raises(HTTPException) as exc_info:
            enterprise_regex_guard(text)
        entities = exc_info.value.detail["violated_entities"]
        assert entities == sorted(entities)


# ── Clean text passes through ─────────────────────────────────────────────────

class TestCleanPassthrough:
    def test_normal_business_question(self):
        _passes("What did Morgan Stanley use evals for in the enterprise AI document?")

    def test_technical_question(self):
        _passes("How does the hybrid FAISS and BM25 retrieval work?")

    def test_document_summary_request(self):
        _passes("Summarize the key findings from the indexed PDF documents.")

    def test_empty_string(self):
        _passes("")

    def test_whitespace_only(self):
        _passes("   ")

    def test_numbers_that_are_not_pii(self):
        # Year, percentage, general statistics — should not be flagged
        _passes("In 2023, 98% of advisors adopted AI tools.")

    def test_short_number(self):
        _passes("Top-5 results from the index.")

    def test_company_name_with_numbers(self):
        _passes("BBVA created 2900 custom GPTs in five months.")


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_returns_original_text_unmodified(self):
        text = "What are the key AI adoption lessons?"
        assert enterprise_regex_guard(text) is text or enterprise_regex_guard(text) == text

    def test_unicode_text_does_not_crash(self):
        # Should not raise — the guard should handle unicode gracefully
        result = enterprise_regex_guard("Klarna résumé AI — über enterprise 通知")
        assert isinstance(result, str)

    def test_very_long_clean_text(self):
        text = "What is the AI adoption strategy? " * 50
        result = enterprise_regex_guard(text)
        assert result == text
