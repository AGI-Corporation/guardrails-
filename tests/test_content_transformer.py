"""
Tests for content_transformer.py
Covers: ContentTransformer redact/mask built-ins, custom transformer
        registration, apply(), apply_all_pii(), TransformationResult.
"""

import pytest

from content_transformer import ContentTransformer, TransformationResult


@pytest.fixture
def ct():
    return ContentTransformer()


# ---------------------------------------------------------------------------
# TransformationResult dataclass
# ---------------------------------------------------------------------------

class TestTransformationResult:
    def test_fields(self):
        r = TransformationResult(
            original="foo",
            transformed="bar",
            changes_made=1,
            transformations_applied=["redact_ssn"],
        )
        assert r.original == "foo"
        assert r.transformed == "bar"
        assert r.changes_made == 1
        assert r.transformations_applied == ["redact_ssn"]


# ---------------------------------------------------------------------------
# SSN redaction
# ---------------------------------------------------------------------------

class TestRedactSSN:
    def test_ssn_redacted(self, ct):
        result = ct.apply("My SSN is 123-45-6789.", ["redact_ssn"])
        assert "[SSN REDACTED]" in result.transformed
        assert "123-45-6789" not in result.transformed
        assert result.changes_made == 1
        assert "redact_ssn" in result.transformations_applied

    def test_multiple_ssns_redacted(self, ct):
        text = "SSN 123-45-6789 and SSN 987-65-4321."
        result = ct.apply(text, ["redact_ssn"])
        assert result.changes_made == 2
        assert "123-45-6789" not in result.transformed
        assert "987-65-4321" not in result.transformed

    def test_no_ssn_no_change(self, ct):
        text = "No SSN here, just regular text."
        result = ct.apply(text, ["redact_ssn"])
        assert result.changes_made == 0
        assert result.transformed == text
        assert "redact_ssn" not in result.transformations_applied

    def test_original_preserved(self, ct):
        original = "SSN 123-45-6789"
        result = ct.apply(original, ["redact_ssn"])
        assert result.original == original


# ---------------------------------------------------------------------------
# Credit card redaction
# ---------------------------------------------------------------------------

class TestRedactCreditCard:
    def test_cc_redacted_spaces(self, ct):
        result = ct.apply("Card: 4111 1111 1111 1111", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_cc_redacted_dashes(self, ct):
        result = ct.apply("Card: 4111-1111-1111-1111", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_cc_no_match(self, ct):
        result = ct.apply("No card here", ["redact_credit_card"])
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# Email redaction
# ---------------------------------------------------------------------------

class TestRedactEmail:
    def test_email_redacted(self, ct):
        result = ct.apply("Contact me at user@example.com please.", ["redact_email"])
        assert "[EMAIL REDACTED]" in result.transformed
        assert "user@example.com" not in result.transformed
        assert result.changes_made == 1

    def test_multiple_emails(self, ct):
        text = "Emails: a@b.com and c@d.org"
        result = ct.apply(text, ["redact_email"])
        assert result.changes_made == 2

    def test_no_email_no_change(self, ct):
        result = ct.apply("No email in this text", ["redact_email"])
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# Phone redaction
# ---------------------------------------------------------------------------

class TestRedactPhone:
    def test_phone_redacted(self, ct):
        result = ct.apply("Call me at 555-867-5309.", ["redact_phone"])
        assert "[PHONE REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_no_phone_no_change(self, ct):
        result = ct.apply("No phone number here", ["redact_phone"])
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# Profanity masking
# ---------------------------------------------------------------------------

class TestMaskProfanity:
    def test_known_profanity_masked(self, ct):
        result = ct.apply("This has badword1 in it.", ["mask_profanity"])
        assert "badword1" not in result.transformed
        assert result.changes_made == 1

    def test_case_insensitive_profanity(self, ct):
        result = ct.apply("BADWORD1 is uppercase", ["mask_profanity"])
        assert "BADWORD1" not in result.transformed

    def test_clean_text_no_change(self, ct):
        result = ct.apply("This is perfectly fine.", ["mask_profanity"])
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# apply() — multiple transformers
# ---------------------------------------------------------------------------

class TestApplyMultiple:
    def test_multiple_transformers_applied(self, ct):
        text = "SSN 123-45-6789 and email user@test.com"
        result = ct.apply(text, ["redact_ssn", "redact_email"])
        assert "[SSN REDACTED]" in result.transformed
        assert "[EMAIL REDACTED]" in result.transformed
        assert result.changes_made == 2

    def test_unknown_transformer_skipped(self, ct):
        result = ct.apply("hello", ["nonexistent_transformer"])
        assert result.transformed == "hello"
        assert result.changes_made == 0
        assert result.transformations_applied == []

    def test_empty_transformer_list(self, ct):
        text = "SSN 123-45-6789"
        result = ct.apply(text, [])
        assert result.transformed == text
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# apply_all_pii()
# ---------------------------------------------------------------------------

class TestApplyAllPII:
    def test_apply_all_pii_redacts_ssn(self, ct):
        result = ct.apply_all_pii("SSN 123-45-6789")
        assert "[SSN REDACTED]" in result.transformed

    def test_apply_all_pii_redacts_cc(self, ct):
        result = ct.apply_all_pii("Card 4111 1111 1111 1111")
        assert "[CC REDACTED]" in result.transformed

    def test_apply_all_pii_redacts_email(self, ct):
        result = ct.apply_all_pii("Email me: me@example.com")
        assert "[EMAIL REDACTED]" in result.transformed

    def test_apply_all_pii_redacts_phone(self, ct):
        result = ct.apply_all_pii("Phone: 555-867-5309")
        assert "[PHONE REDACTED]" in result.transformed

    def test_apply_all_pii_no_pii(self, ct):
        result = ct.apply_all_pii("No PII here at all.")
        assert result.changes_made == 0
        assert result.transformed == "No PII here at all."


# ---------------------------------------------------------------------------
# Custom transformer registration
# ---------------------------------------------------------------------------

class TestCustomTransformerRegistration:
    def test_register_custom_transformer(self, ct):
        def replace_hello(text):
            import re
            result, n = re.subn(r"\bhello\b", "[GREETING]", text, flags=re.IGNORECASE)
            return result, n

        ct.register("replace_hello", replace_hello)
        result = ct.apply("Say hello to everyone", ["replace_hello"])
        assert "[GREETING]" in result.transformed
        assert result.changes_made == 1

    def test_register_overwrites_existing(self, ct):
        def noop(text):
            return text, 0

        ct.register("redact_ssn", noop)
        result = ct.apply("SSN 123-45-6789", ["redact_ssn"])
        # noop transformer should not redact anything
        assert "123-45-6789" in result.transformed
