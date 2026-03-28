"""
Tests for content_transformer.py
"""

import pytest

from content_transformer import ContentTransformer, TransformationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def transformer():
    return ContentTransformer()


# ---------------------------------------------------------------------------
# TransformationResult dataclass
# ---------------------------------------------------------------------------

class TestTransformationResult:
    def test_fields(self):
        result = TransformationResult(
            original="original text",
            transformed="new text",
            changes_made=1,
            transformations_applied=["redact_ssn"],
        )
        assert result.original == "original text"
        assert result.transformed == "new text"
        assert result.changes_made == 1
        assert result.transformations_applied == ["redact_ssn"]


# ---------------------------------------------------------------------------
# ContentTransformer — default registration
# ---------------------------------------------------------------------------

class TestContentTransformerDefaults:
    def test_default_transformers_registered(self, transformer):
        for name in ["redact_ssn", "redact_credit_card", "redact_email", "redact_phone", "mask_profanity"]:
            assert name in transformer.transformers

    def test_register_custom_transformer(self, transformer):
        def my_transform(text):
            return text.replace("hello", "hi"), text.count("hello")

        transformer.register("my_transform", my_transform)
        assert "my_transform" in transformer.transformers


# ---------------------------------------------------------------------------
# SSN redaction
# ---------------------------------------------------------------------------

class TestRedactSSN:
    def test_ssn_is_redacted(self, transformer):
        result = transformer.apply("My SSN is 123-45-6789.", ["redact_ssn"])
        assert "[SSN REDACTED]" in result.transformed
        assert "123-45-6789" not in result.transformed
        assert result.changes_made == 1
        assert "redact_ssn" in result.transformations_applied

    def test_multiple_ssns_redacted(self, transformer):
        result = transformer.apply(
            "SSN1: 123-45-6789 and SSN2: 987-65-4321", ["redact_ssn"]
        )
        assert result.changes_made == 2

    def test_no_ssn_no_change(self, transformer):
        result = transformer.apply("No sensitive data here.", ["redact_ssn"])
        assert result.changes_made == 0
        assert result.transformations_applied == []

    def test_original_unchanged(self, transformer):
        original = "My SSN is 123-45-6789."
        result = transformer.apply(original, ["redact_ssn"])
        assert result.original == original


# ---------------------------------------------------------------------------
# Credit card redaction
# ---------------------------------------------------------------------------

class TestRedactCreditCard:
    def test_credit_card_with_spaces_redacted(self, transformer):
        result = transformer.apply("Card: 4111 1111 1111 1111", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_credit_card_with_dashes_redacted(self, transformer):
        result = transformer.apply("Card: 4111-1111-1111-1111", ["redact_credit_card"])
        assert "[CC REDACTED]" in result.transformed

    def test_no_credit_card_no_change(self, transformer):
        result = transformer.apply("No card here.", ["redact_credit_card"])
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# Email redaction
# ---------------------------------------------------------------------------

class TestRedactEmail:
    def test_email_redacted(self, transformer):
        result = transformer.apply("Contact: user@example.com", ["redact_email"])
        assert "[EMAIL REDACTED]" in result.transformed
        assert "user@example.com" not in result.transformed
        assert result.changes_made == 1

    def test_multiple_emails_redacted(self, transformer):
        result = transformer.apply(
            "From: a@b.com, To: c@d.org", ["redact_email"]
        )
        assert result.changes_made == 2

    def test_no_email_no_change(self, transformer):
        result = transformer.apply("No email here.", ["redact_email"])
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# Phone redaction
# ---------------------------------------------------------------------------

class TestRedactPhone:
    def test_phone_us_format_redacted(self, transformer):
        result = transformer.apply("Call me at 555-867-5309", ["redact_phone"])
        assert "[PHONE REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_phone_with_area_code_parens(self, transformer):
        result = transformer.apply("Phone: (555) 867-5309", ["redact_phone"])
        assert "[PHONE REDACTED]" in result.transformed

    def test_no_phone_no_change(self, transformer):
        result = transformer.apply("No phone here.", ["redact_phone"])
        assert result.changes_made == 0


# ---------------------------------------------------------------------------
# apply_all_pii
# ---------------------------------------------------------------------------

class TestApplyAllPII:
    def test_apply_all_pii_redacts_multiple_types(self, transformer):
        text = "SSN: 123-45-6789, Card: 4111 1111 1111 1111, Email: a@b.com"
        result = transformer.apply_all_pii(text)
        assert "[SSN REDACTED]" in result.transformed
        assert "[CC REDACTED]" in result.transformed
        assert "[EMAIL REDACTED]" in result.transformed

    def test_apply_all_pii_clean_text(self, transformer):
        result = transformer.apply_all_pii("Hello, how are you?")
        assert result.changes_made == 0
        assert result.transformations_applied == []


# ---------------------------------------------------------------------------
# apply — unknown transformer
# ---------------------------------------------------------------------------

class TestApplyUnknownTransformer:
    def test_unknown_transformer_skipped(self, transformer):
        result = transformer.apply("test text", ["nonexistent_transformer"])
        assert result.transformed == "test text"
        assert result.changes_made == 0

    def test_mix_of_known_and_unknown(self, transformer):
        result = transformer.apply(
            "My SSN is 123-45-6789.", ["redact_ssn", "unknown"]
        )
        assert "[SSN REDACTED]" in result.transformed
        assert result.changes_made == 1


# ---------------------------------------------------------------------------
# Custom transformer
# ---------------------------------------------------------------------------

class TestCustomTransformer:
    def test_custom_transformer_applied(self, transformer):
        def redact_secret(text):
            import re
            result, n = re.subn(r"\bSECRET\b", "[REDACTED]", text)
            return result, n

        transformer.register("redact_secret", redact_secret)
        result = transformer.apply("The SECRET code is here", ["redact_secret"])
        assert "[REDACTED]" in result.transformed
        assert result.changes_made == 1

    def test_chaining_transformers(self, transformer):
        text = "SSN: 123-45-6789, Email: user@test.com"
        result = transformer.apply(text, ["redact_ssn", "redact_email"])
        assert "[SSN REDACTED]" in result.transformed
        assert "[EMAIL REDACTED]" in result.transformed
        assert len(result.transformations_applied) == 2


# ---------------------------------------------------------------------------
# Profanity masking (covers _mask_profanity inner loop)
# ---------------------------------------------------------------------------

class TestMaskProfanity:
    def test_profanity_word_masked(self, transformer):
        result = transformer.apply("This has badword1 in it", ["mask_profanity"])
        assert "badword1" not in result.transformed
        assert "*" * len("badword1") in result.transformed
        assert result.changes_made == 1
        assert "mask_profanity" in result.transformations_applied

    def test_second_profanity_word_masked(self, transformer):
        result = transformer.apply("badword2 is not allowed", ["mask_profanity"])
        assert "badword2" not in result.transformed
        assert result.changes_made == 1

    def test_both_profanity_words_masked(self, transformer):
        result = transformer.apply("badword1 and badword2 here", ["mask_profanity"])
        assert result.changes_made == 2

    def test_profanity_case_insensitive(self, transformer):
        result = transformer.apply("BADWORD1 in text", ["mask_profanity"])
        assert result.changes_made == 1

    def test_no_profanity_no_change(self, transformer):
        result = transformer.apply("Clean text with no bad words", ["mask_profanity"])
        assert result.changes_made == 0
        assert result.transformations_applied == []
