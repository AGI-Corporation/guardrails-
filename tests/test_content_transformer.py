"""
Tests for content_transformer.py
Covers: ContentTransformer (individual and combined transformers),
        TransformationResult structure.
"""

import pytest
from content_transformer import ContentTransformer, TransformationResult


@pytest.fixture
def ct():
    return ContentTransformer()


class TestSSNRedaction:

    def test_ssn_redacted(self, ct):
        result = ct.apply("My SSN is 123-45-6789 please.", ["redact_ssn"])
        assert "[SSN REDACTED]" in result.transformed
        assert "123-45-6789" not in result.transformed
        assert result.changes_made == 1

    def test_no_ssn_unchanged(self, ct):
        result = ct.apply("No sensitive data here.", ["redact_ssn"])
        assert result.transformed == "No sensitive data here."
        assert result.changes_made == 0

    def test_multiple_ssns(self, ct):
        result = ct.apply("SSN1: 111-22-3333 and SSN2: 444-55-6666", ["redact_ssn"])
        assert result.changes_made == 2
        assert "111-22-3333" not in result.transformed
        assert "444-55-6666" not in result.transformed


class TestCreditCardRedaction:

    def test_credit_card_redacted(self, ct):
        result = ct.apply("Card: 4111 1111 1111 1111", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert "4111" not in result.transformed
        assert result.changes_made == 1

    def test_dashed_format_redacted(self, ct):
        result = ct.apply("Card: 4111-1111-1111-1111", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed


class TestEmailRedaction:

    def test_email_redacted(self, ct):
        result = ct.apply("Contact me at alice@example.com please.", ["redact_email"])
        assert "[EMAIL REDACTED]" in result.transformed
        assert "alice@example.com" not in result.transformed
        assert result.changes_made == 1

    def test_no_email_unchanged(self, ct):
        result = ct.apply("No email here.", ["redact_email"])
        assert result.changes_made == 0


class TestPhoneRedaction:

    def test_phone_redacted(self, ct):
        result = ct.apply("Call me at 555-123-4567 anytime.", ["redact_phone"])
        assert "[PHONE REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_no_phone_unchanged(self, ct):
        result = ct.apply("No phone number here.", ["redact_phone"])
        assert result.changes_made == 0


class TestApplyAllPii:

    def test_combined_pii_all_redacted(self, ct):
        text = "SSN: 123-45-6789, Card: 4111 1111 1111 1111, Email: a@b.com"
        result = ct.apply_all_pii(text)
        assert "123-45-6789" not in result.transformed
        assert "4111 1111 1111 1111" not in result.transformed
        assert "a@b.com" not in result.transformed
        assert result.changes_made >= 3

    def test_clean_text_unchanged(self, ct):
        text = "This text has no PII whatsoever."
        result = ct.apply_all_pii(text)
        assert result.transformed == text
        assert result.changes_made == 0


class TestTransformationResult:

    def test_original_preserved(self, ct):
        original = "My SSN is 123-45-6789."
        result = ct.apply(original, ["redact_ssn"])
        assert result.original == original

    def test_transformations_applied_list(self, ct):
        result = ct.apply("Email: a@b.com, SSN: 123-45-6789", ["redact_ssn", "redact_email"])
        assert "redact_ssn" in result.transformations_applied
        assert "redact_email" in result.transformations_applied

    def test_unknown_transformer_ignored(self, ct):
        result = ct.apply("hello", ["nonexistent_transformer"])
        assert result.transformed == "hello"
        assert result.changes_made == 0

    def test_custom_transformer_registration(self, ct):
        def upper_all(text: str):
            return text.upper(), 1

        ct.register("to_upper", upper_all)
        result = ct.apply("hello world", ["to_upper"])
        assert result.transformed == "HELLO WORLD"
        assert result.changes_made == 1
        assert "to_upper" in result.transformations_applied
