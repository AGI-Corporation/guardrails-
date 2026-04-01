"""
Tests for content_transformer.py
"""
import pytest
from content_transformer import ContentTransformer, TransformationResult


@pytest.fixture
def transformer() -> ContentTransformer:
    return ContentTransformer()


class TestContentTransformer:
    def test_redact_ssn(self, transformer):
        result = transformer.apply(
            "SSN: 123-45-6789", ["redact_ssn"]
        )
        assert "[SSN REDACTED]" in result.transformed
        assert "123-45-6789" not in result.transformed
        assert result.changes_made == 1
        assert "redact_ssn" in result.transformations_applied

    def test_redact_credit_card(self, transformer):
        result = transformer.apply(
            "Card: 4111-1111-1111-1111", ["redact_credit_card"]
        )
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made >= 1

    def test_redact_email(self, transformer):
        result = transformer.apply(
            "Contact john.doe@example.com", ["redact_email"]
        )
        assert "[EMAIL REDACTED]" in result.transformed
        assert "john.doe@example.com" not in result.transformed

    def test_redact_phone(self, transformer):
        result = transformer.apply(
            "Call 555-867-5309 today", ["redact_phone"]
        )
        assert "[PHONE REDACTED]" in result.transformed

    def test_apply_all_pii(self, transformer):
        text = "SSN 123-45-6789 email test@example.com card 4111-1111-1111-1111"
        result = transformer.apply_all_pii(text)
        assert "[SSN REDACTED]" in result.transformed
        assert "[EMAIL REDACTED]" in result.transformed
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made >= 3

    def test_clean_text_unchanged(self, transformer):
        text = "This is clean text with no PII."
        result = transformer.apply_all_pii(text)
        assert result.transformed == text
        assert result.changes_made == 0
        assert result.transformations_applied == []

    def test_original_preserved(self, transformer):
        text = "SSN: 123-45-6789"
        result = transformer.apply_all_pii(text)
        assert result.original == text

    def test_register_custom_transformer(self, transformer):
        def redact_custom(text: str):
            import re
            result, n = re.subn(r"\bSECRET\b", "[REDACTED]", text)
            return result, n

        transformer.register("custom", redact_custom)
        result = transformer.apply("Contains SECRET info", ["custom"])
        assert "[REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_unknown_transformer_skipped(self, transformer):
        result = transformer.apply("text", ["nonexistent_transformer"])
        assert result.transformed == "text"
        assert result.changes_made == 0

    def test_multiple_ssn_redacted(self, transformer):
        text = "First: 111-22-3333 and second: 444-55-6666"
        result = transformer.apply(text, ["redact_ssn"])
        assert result.changes_made == 2
        assert "111-22-3333" not in result.transformed
        assert "444-55-6666" not in result.transformed
