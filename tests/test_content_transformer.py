"""
Tests for content_transformer.py — PII redaction, masking, custom transformers.
"""
import pytest

from content_transformer import ContentTransformer, TransformationResult


@pytest.fixture()
def transformer() -> ContentTransformer:
    return ContentTransformer()


# ── SSN redaction ─────────────────────────────────────────────────────────────

class TestRedactSSN:
    def test_ssn_redacted(self, transformer: ContentTransformer):
        result = transformer.apply("My SSN is 123-45-6789", ["redact_ssn"])
        assert "[SSN REDACTED]" in result.transformed
        assert "123-45-6789" not in result.transformed

    def test_ssn_redacted_changes_count(self, transformer: ContentTransformer):
        result = transformer.apply("SSNs: 123-45-6789 and 987-65-4321", ["redact_ssn"])
        assert result.changes_made == 2

    def test_no_ssn_unchanged(self, transformer: ContentTransformer):
        result = transformer.apply("No PII here", ["redact_ssn"])
        assert result.changes_made == 0
        assert result.transformed == "No PII here"


# ── Credit card redaction ─────────────────────────────────────────────────────

class TestRedactCreditCard:
    def test_credit_card_redacted(self, transformer: ContentTransformer):
        result = transformer.apply("Card: 4111-1111-1111-1111", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert "4111" not in result.transformed

    def test_no_credit_card_unchanged(self, transformer: ContentTransformer):
        result = transformer.apply("No card here", ["redact_credit_card"])
        assert result.changes_made == 0


# ── Email redaction ───────────────────────────────────────────────────────────

class TestRedactEmail:
    def test_email_redacted(self, transformer: ContentTransformer):
        result = transformer.apply("Email me at alice@example.com", ["redact_email"])
        assert "[EMAIL REDACTED]" in result.transformed
        assert "alice@example.com" not in result.transformed

    def test_no_email_unchanged(self, transformer: ContentTransformer):
        result = transformer.apply("No email here", ["redact_email"])
        assert result.changes_made == 0


# ── Phone redaction ───────────────────────────────────────────────────────────

class TestRedactPhone:
    def test_phone_redacted(self, transformer: ContentTransformer):
        result = transformer.apply("Call me at 555-867-5309", ["redact_phone"])
        assert "[PHONE REDACTED]" in result.transformed

    def test_no_phone_unchanged(self, transformer: ContentTransformer):
        result = transformer.apply("No phone here", ["redact_phone"])
        assert result.changes_made == 0


# ── apply_all_pii ─────────────────────────────────────────────────────────────

class TestApplyAllPii:
    def test_ssn_redacted_in_all_pii(self, transformer: ContentTransformer):
        result = transformer.apply_all_pii("SSN: 123-45-6789")
        assert "[SSN REDACTED]" in result.transformed

    def test_email_redacted_in_all_pii(self, transformer: ContentTransformer):
        result = transformer.apply_all_pii("Email: bob@test.org")
        assert "[EMAIL REDACTED]" in result.transformed

    def test_clean_text_unchanged(self, transformer: ContentTransformer):
        result = transformer.apply_all_pii("Hello, how are you?")
        assert result.changes_made == 0
        assert result.transformed == "Hello, how are you?"

    def test_multiple_pii_types(self, transformer: ContentTransformer):
        text = "SSN 123-45-6789 email me at x@y.com"
        result = transformer.apply_all_pii(text)
        assert result.changes_made >= 2
        assert len(result.transformations_applied) >= 2


# ── TransformationResult ──────────────────────────────────────────────────────

class TestTransformationResult:
    def test_original_preserved(self, transformer: ContentTransformer):
        text = "My SSN is 123-45-6789"
        result = transformer.apply_all_pii(text)
        assert result.original == text

    def test_transformations_applied_list(self, transformer: ContentTransformer):
        result = transformer.apply_all_pii("SSN: 123-45-6789")
        assert isinstance(result.transformations_applied, list)
        assert "redact_ssn" in result.transformations_applied


# ── Custom transformer registration ──────────────────────────────────────────

class TestCustomTransformer:
    def test_register_and_apply(self, transformer: ContentTransformer):
        def redact_foo(text: str):
            import re
            result, n = re.subn(r"\bfoo\b", "[FOO]", text)
            return result, n

        transformer.register("redact_foo", redact_foo)
        result = transformer.apply("foo bar foo", ["redact_foo"])
        assert result.changes_made == 2
        assert "[FOO]" in result.transformed

    def test_unknown_transformer_skipped(self, transformer: ContentTransformer):
        result = transformer.apply("hello", ["does_not_exist"])
        assert result.transformed == "hello"
        assert result.changes_made == 0


# ── apply() helper ────────────────────────────────────────────────────────────

class TestApplyHelper:
    def test_apply_returns_transformation_result(self, transformer: ContentTransformer):
        result = transformer.apply("text", ["redact_ssn"])
        assert isinstance(result, TransformationResult)

    def test_chained_transforms(self, transformer: ContentTransformer):
        text = "SSN 123-45-6789 email x@y.com"
        result = transformer.apply(text, ["redact_ssn", "redact_email"])
        assert "[SSN REDACTED]" in result.transformed
        assert "[EMAIL REDACTED]" in result.transformed
