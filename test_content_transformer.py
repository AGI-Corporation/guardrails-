"""
Tests for content_transformer.py
Covers ContentTransformer and all built-in transformation functions.
"""

import pytest
from content_transformer import ContentTransformer, TransformationResult


class TestTransformationResult:
    def test_dataclass_fields(self):
        result = TransformationResult(
            original="hello",
            transformed="world",
            changes_made=1,
            transformations_applied=["redact_ssn"],
        )
        assert result.original == "hello"
        assert result.transformed == "world"
        assert result.changes_made == 1
        assert result.transformations_applied == ["redact_ssn"]


class TestContentTransformerInit:
    def test_default_transformers_registered(self):
        ct = ContentTransformer()
        expected = [
            "redact_ssn",
            "redact_credit_card",
            "redact_email",
            "redact_phone",
            "mask_profanity",
        ]
        for name in expected:
            assert name in ct.transformers

    def test_register_custom_transformer(self):
        ct = ContentTransformer()

        def my_transformer(text):
            result = text.replace("foo", "bar")
            count = text.count("foo")
            return result, count

        ct.register("replace_foo", my_transformer)
        assert "replace_foo" in ct.transformers


class TestRedactSSN:
    def test_redacts_basic_ssn(self):
        ct = ContentTransformer()
        result = ct.apply("My SSN is 123-45-6789.", ["redact_ssn"])
        assert "[SSN REDACTED]" in result.transformed
        assert "123-45-6789" not in result.transformed
        assert result.changes_made == 1
        assert "redact_ssn" in result.transformations_applied

    def test_redacts_multiple_ssns(self):
        ct = ContentTransformer()
        result = ct.apply("SSNs: 123-45-6789 and 987-65-4321", ["redact_ssn"])
        assert result.changes_made == 2
        assert "123-45-6789" not in result.transformed
        assert "987-65-4321" not in result.transformed

    def test_no_ssn_no_change(self):
        ct = ContentTransformer()
        result = ct.apply("No sensitive data here.", ["redact_ssn"])
        assert result.changes_made == 0
        assert result.transformed == "No sensitive data here."
        assert result.transformations_applied == []

    def test_original_preserved(self):
        ct = ContentTransformer()
        text = "SSN 123-45-6789"
        result = ct.apply(text, ["redact_ssn"])
        assert result.original == text


class TestRedactCreditCard:
    def test_redacts_standard_credit_card(self):
        ct = ContentTransformer()
        result = ct.apply("Card: 1234 5678 9012 3456", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_redacts_hyphenated_credit_card(self):
        ct = ContentTransformer()
        result = ct.apply("Card: 1234-5678-9012-3456", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_redacts_no_separator_credit_card(self):
        ct = ContentTransformer()
        result = ct.apply("Card: 1234567890123456", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_no_credit_card_no_change(self):
        ct = ContentTransformer()
        result = ct.apply("No card data here.", ["redact_credit_card"])
        assert result.changes_made == 0
        assert result.transformations_applied == []


class TestRedactEmail:
    def test_redacts_basic_email(self):
        ct = ContentTransformer()
        result = ct.apply("Contact me at user@example.com please.", ["redact_email"])
        assert "[EMAIL REDACTED]" in result.transformed
        assert "user@example.com" not in result.transformed
        assert result.changes_made == 1

    def test_redacts_multiple_emails(self):
        ct = ContentTransformer()
        result = ct.apply("Emails: a@b.com and c@d.org", ["redact_email"])
        assert result.changes_made == 2

    def test_no_email_no_change(self):
        ct = ContentTransformer()
        result = ct.apply("No email here.", ["redact_email"])
        assert result.changes_made == 0

    def test_complex_email_redacted(self):
        ct = ContentTransformer()
        result = ct.apply("Email: first.last+tag@sub.domain.co.uk", ["redact_email"])
        assert "[EMAIL REDACTED]" in result.transformed


class TestRedactPhone:
    def test_redacts_us_phone_dashes(self):
        ct = ContentTransformer()
        result = ct.apply("Call me at 555-867-5309.", ["redact_phone"])
        assert "[PHONE REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_redacts_us_phone_dots(self):
        ct = ContentTransformer()
        result = ct.apply("Call 555.867.5309 anytime.", ["redact_phone"])
        assert "[PHONE REDACTED]" in result.transformed

    def test_no_phone_no_change(self):
        ct = ContentTransformer()
        result = ct.apply("No phone number here.", ["redact_phone"])
        assert result.changes_made == 0
        assert result.transformations_applied == []


class TestMaskProfanity:
    def test_masks_known_profanity(self):
        ct = ContentTransformer()
        result = ct.apply("This has badword1 in it.", ["mask_profanity"])
        assert "badword1" not in result.transformed
        assert "*" * len("badword1") in result.transformed
        assert result.changes_made == 1

    def test_masks_second_known_profanity(self):
        ct = ContentTransformer()
        result = ct.apply("Both badword1 and badword2 present.", ["mask_profanity"])
        assert "badword1" not in result.transformed
        assert "badword2" not in result.transformed

    def test_no_profanity_no_change(self):
        ct = ContentTransformer()
        result = ct.apply("Clean text here.", ["mask_profanity"])
        assert result.changes_made == 0
        assert result.transformations_applied == []


class TestApplyMethod:
    def test_apply_unknown_transformer_ignored(self):
        ct = ContentTransformer()
        result = ct.apply("Some text", ["nonexistent_transformer"])
        assert result.transformed == "Some text"
        assert result.changes_made == 0
        assert result.transformations_applied == []

    def test_apply_multiple_transformers(self):
        ct = ContentTransformer()
        text = "SSN 123-45-6789 and email user@example.com"
        result = ct.apply(text, ["redact_ssn", "redact_email"])
        assert "[SSN REDACTED]" in result.transformed
        assert "[EMAIL REDACTED]" in result.transformed
        assert result.changes_made == 2
        assert "redact_ssn" in result.transformations_applied
        assert "redact_email" in result.transformations_applied

    def test_apply_empty_list_no_change(self):
        ct = ContentTransformer()
        text = "SSN 123-45-6789"
        result = ct.apply(text, [])
        assert result.transformed == text
        assert result.changes_made == 0


class TestApplyAllPII:
    def test_apply_all_pii_redacts_ssn(self):
        ct = ContentTransformer()
        result = ct.apply_all_pii("SSN: 123-45-6789")
        assert "[SSN REDACTED]" in result.transformed

    def test_apply_all_pii_redacts_email(self):
        ct = ContentTransformer()
        result = ct.apply_all_pii("Email: test@example.com")
        assert "[EMAIL REDACTED]" in result.transformed

    def test_apply_all_pii_redacts_phone(self):
        ct = ContentTransformer()
        result = ct.apply_all_pii("Phone: 555-867-5309")
        assert "[PHONE REDACTED]" in result.transformed

    def test_apply_all_pii_no_pii(self):
        ct = ContentTransformer()
        result = ct.apply_all_pii("This text has no PII.")
        assert result.changes_made == 0

    def test_apply_all_pii_multiple_types(self):
        ct = ContentTransformer()
        text = "SSN 123-45-6789, email a@b.com, card 1234-5678-9012-3456"
        result = ct.apply_all_pii(text)
        assert result.changes_made >= 3
