"""
🛡️ Guardrails Framework — Core Engine
"""
import json
import yaml
import re
import logging
from dataclasses import dataclass, field, asdict
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
            if re.search(p, text, re.I): return True
        for k in self.keywords:
            if k.lower() in text.lower(): return True
        return False

class GuardrailEngine:
    def __init__(self):
        self.rules: Dict[str, GuardrailRule] = {}

    def add_rule(self, rule: GuardrailRule):
        self.rules[rule.id] = rule

    def evaluate(self, text: str):
        matched = [r.id for r in self.rules.values() if r.matches(text)]
        return {"action": Action.BLOCK if matched else Action.ALLOW, "matches": matched}

def create_default_guardrails():
    return [GuardrailRule("ssn", "SSN", Severity.CRITICAL, Action.BLOCK, [r"\d{3}-\d{2}-\d{4}"])]

if __name__ == "__main__":
    e = GuardrailEngine()
    for r in create_default_guardrails(): e.add_rule(r)
    print(e.evaluate("My SSN is 123-45-6789"))

