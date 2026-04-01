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
        self.register("redact_ip_address", self._redact_ip_address)
        self.register("redact_api_key", self._redact_api_key)
        self.register("redact_jwt", self._redact_jwt)
        self.register("redact_passport", self._redact_passport)

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
        return self.apply(text, [
            "redact_ssn", "redact_credit_card", "redact_email", "redact_phone",
            "redact_ip_address", "redact_api_key", "redact_jwt", "redact_passport",
        ])

    def apply_all(self, text: str) -> TransformationResult:
        """Apply every registered transformer."""
        return self.apply(text, list(self.transformers.keys()))

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

    @staticmethod
    def _redact_ip_address(text: str):
        pattern = r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        result, n = re.subn(pattern, "[IP REDACTED]", text)
        return result, n

    @staticmethod
    def _redact_api_key(text: str):
        pattern = (
            r"\b(sk-[A-Za-z0-9]{20,}"
            r"|ghp_[A-Za-z0-9]{36}"
            r"|AKIA[A-Z0-9]{16})\b"
        )
        result, n = re.subn(pattern, "[API KEY REDACTED]", text)
        return result, n

    @staticmethod
    def _redact_jwt(text: str):
        pattern = r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
        result, n = re.subn(pattern, "[JWT REDACTED]", text)
        return result, n

    @staticmethod
    def _redact_passport(text: str):
        pattern = r"(?i)(passport\s+(number|no\.?|#)\s*[:\-]?\s*)[A-Z]{1,2}\d{6,9}"
        result, n = re.subn(pattern, r"\1[PASSPORT REDACTED]", text)
        return result, n
