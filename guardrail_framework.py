"""
🛡️ Guardrails Framework — Core Engine
"""
import json
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GuardrailEngine")


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


# Maps severity level to a numeric order for comparison
_SEVERITY_ORDER: Dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


@dataclass
class EvaluationResult:
    """Result returned by GuardrailEngine.evaluate()."""
    action: Action
    matches: List[str]
    severity: Severity = Severity.LOW
    risk_score: float = 0.0
    text: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def matched_rules(self) -> List[str]:
        """Alias for ``matches`` used by downstream consumers."""
        return self.matches


@dataclass
class GuardrailRule:
    id: str
    name: str
    severity: Severity
    action: Action
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        for p in self.patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        for k in self.keywords:
            if k.lower() in text.lower():
                return True
        return False


class GuardrailEngine:
    def __init__(self):
        self.rules: Dict[str, GuardrailRule] = {}

    def add_rule(self, rule: GuardrailRule):
        self.rules[rule.id] = rule

    def evaluate(self, text: str) -> EvaluationResult:
        matched_rules = [r for r in self.rules.values() if r.matches(text)]
        matched_ids = [r.id for r in matched_rules]

        if matched_rules:
            action = Action.BLOCK
            severity = max(
                (r.severity for r in matched_rules),
                key=lambda s: _SEVERITY_ORDER[s],
            )
            risk_score = round(len(matched_ids) / max(len(self.rules), 1), 3)
        else:
            action = Action.ALLOW
            severity = Severity.LOW
            risk_score = 0.0

        return EvaluationResult(
            action=action,
            matches=matched_ids,
            severity=severity,
            risk_score=risk_score,
            text=text,
        )


def create_default_guardrails() -> List[GuardrailRule]:
    """Return a set of sensible default guardrail rules."""
    return [
        GuardrailRule(
            "ssn", "SSN Detection", Severity.CRITICAL, Action.BLOCK,
            patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
        ),
        GuardrailRule(
            "credit_card", "Credit Card Detection", Severity.CRITICAL, Action.BLOCK,
            patterns=[r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"],
        ),
        GuardrailRule(
            "prompt_injection", "Prompt Injection", Severity.HIGH, Action.BLOCK,
            patterns=[r"ignore\s+(all\s+)?previous\s+instructions"],
            keywords=["ignore all instructions", "disregard your instructions",
                      "override safety", "jailbreak"],
        ),
        GuardrailRule(
            "violence", "Violence / Weapons", Severity.HIGH, Action.BLOCK,
            keywords=["bomb", "explosive", "weapon", "how to kill", "murder"],
        ),
        GuardrailRule(
            "pii_email", "Email Address", Severity.MEDIUM, Action.WARN,
            patterns=[r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"],
        ),
    ]


if __name__ == "__main__":
    e = GuardrailEngine()
    for r in create_default_guardrails():
        e.add_rule(r)
    result = e.evaluate("My SSN is 123-45-6789")
    print(f"action={result.action.value}, matches={result.matches}, severity={result.severity.value}")

