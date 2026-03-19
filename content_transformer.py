"""
Content Transformer
Handles content transformations for guardrail actions (redaction, masking).
"""

import re
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TransformationResult:
    original: str
    transformed: str
    changes_made: int
    transformations_applied: List[str]


class ContentTransformer:
    """Handle content transformations for guardrail actions"""

    def __init__(self):
        self.transformers: Dict[str, Callable] = {}
        self._register_default_transformers()

    def _register_default_transformers(self):
        self.register("redact_ssn", self._redact_ssn)
        self.register("redact_credit_card", self._redact_credit_card)
        self.register("redact_email", self._redact_email)
        self.register("redact_phone", self._redact_phone)
        self.register("mask_profanity", self._mask_profanity)

    def register(self, name: str, transformer: Callable):
        self.transformers[name] = transformer

    def apply(self, text: str, transformer_names: List[str]) -> TransformationResult:
        transformed = text
        changes = 0
        applied = []

        for name in transformer_names:
            if name in self.transformers:
                result, n = self.transformers[name](transformed)
                if n > 0:
                    transformed = result
                    changes += n
                    applied.append(name)

        return TransformationResult(
            original=text,
            transformed=transformed,
            changes_made=changes,
            transformations_applied=applied,
        )

    def apply_all_pii(self, text: str) -> TransformationResult:
        return self.apply(text, ["redact_ssn", "redact_credit_card", "redact_email", "redact_phone"])

    # ── Built-in transformers ───────────────────────────────────────────────

    @staticmethod
    def _redact_ssn(text: str):
        pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        result, n = re.subn(pattern, "[SSN REDACTED]", text)
        return result, n

    @staticmethod
    def _redact_credit_card(text: str):
        pattern = r"\b(?:\d{4}[\s-]?){3}\d{4}\b"
        result, n = re.subn(pattern, "[CC REDACTED]", text)
        return result, n

    @staticmethod
    def _redact_email(text: str):
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        result, n = re.subn(pattern, "[EMAIL REDACTED]", text)
        return result, n

    @staticmethod
    def _redact_phone(text: str):
        pattern = r"\b(?:\+1[\s-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"
        result, n = re.subn(pattern, "[PHONE REDACTED]", text)
        return result, n

    @staticmethod
    def _mask_profanity(text: str):
        # Basic placeholder - extend with actual word list in production
        profanity_list = ["badword1", "badword2"]
        result = text
        n = 0
        for word in profanity_list:
            new_result = re.sub(
                r"\b" + re.escape(word) + r"\b",
                "*" * len(word),
                result,
                flags=re.IGNORECASE,
            )
            if new_result != result:
                n += 1
                result = new_result
        return result, n
