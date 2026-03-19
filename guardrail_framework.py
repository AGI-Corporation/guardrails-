"""
Guardrail Testing & Definition Framework
A professional tool for defining and testing AI safety guardrails.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable
from enum import Enum
from datetime import datetime
import re


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailCategory(Enum):
    HARMFUL_CONTENT = "harmful_content"
    PII_PROTECTION = "pii_protection"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    VIOLENCE = "violence"
    ILLEGAL_ACTIVITIES = "illegal_activities"
    MISINFORMATION = "misinformation"
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"
    CUSTOM = "custom"


class Action(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    TRANSFORM = "transform"
    LOG = "log"


@dataclass
class GuardrailRule:
    """Defines a single guardrail rule."""
    id: str
    name: str
    category: GuardrailCategory
    severity: Severity
    action: Action
    patterns: List[str] = field(default_factory=list)  # regex patterns
    keywords: List[str] = field(default_factory=list)
    description: str = ""
    enabled: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating text against guardrails."""
    text: str
    action: str
    matched_rules: List[str]
    severity: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict = field(default_factory=dict)


@dataclass
class TestCase:
    """A test case for validating guardrail behavior."""
    id: str
    input_text: str
    expected_action: str  # "allow" or "block"
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of running a test case."""
    test_case_id: str
    passed: bool
    expected: str
    got: str
    matched_rules: List[str]
    execution_time_ms: float = 0.0


# =============================================================================
# GUARDRAIL ENGINE
# =============================================================================

class GuardrailEngine:
    """Core engine for evaluating text against guardrail rules."""

    def __init__(self):
        self.rules: Dict[str, GuardrailRule] = {}
        self.test_cases: Dict[str, TestCase] = {}

    def add_rule(self, rule: GuardrailRule):
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str):
        self.rules.pop(rule_id, None)

    def add_test_case(self, test_case: TestCase):
        self.test_cases[test_case.id] = test_case

    def evaluate(self, text: str) -> EvaluationResult:
        """Evaluate text against all active rules."""
        import time
        matched_rules = []
        highest_severity = None
        final_action = Action.ALLOW

        severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            matched = False
            for pattern in rule.patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matched = True
                    break

            if not matched:
                for keyword in rule.keywords:
                    if keyword.lower() in text.lower():
                        matched = True
                        break

            if matched:
                matched_rules.append(rule.id)
                if highest_severity is None or severity_order.index(rule.severity) > severity_order.index(highest_severity):
                    highest_severity = rule.severity
                    final_action = rule.action

        return EvaluationResult(
            text=text,
            action=final_action.value if matched_rules else Action.ALLOW.value,
            matched_rules=matched_rules,
            severity=highest_severity.value if highest_severity else "none",
        )

    def run_tests(self) -> List[TestResult]:
        """Run all test cases and return results."""
        results = []
        for tc in self.test_cases.values():
            import time
            start = time.time()
            result = self.evaluate(tc.input_text)
            elapsed = (time.time() - start) * 1000

            passed = result.action == tc.expected_action
            results.append(TestResult(
                test_case_id=tc.id,
                passed=passed,
                expected=tc.expected_action,
                got=result.action,
                matched_rules=result.matched_rules,
                execution_time_ms=elapsed,
            ))
        return results

    def export_rules(self, filepath: str, fmt: str = "yaml"):
        """Export rules to YAML or JSON."""
        data = {rid: asdict(rule) for rid, rule in self.rules.items()}
        # Convert enums to strings
        for r in data.values():
            r["category"] = r["category"] if isinstance(r["category"], str) else r["category"]

        with open(filepath, "w") as f:
            if fmt == "yaml":
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2, default=str)


# =============================================================================
# DEFAULT RULES & TEST CASES
# =============================================================================

def create_default_guardrails() -> List[GuardrailRule]:
    """Create a standard set of guardrail rules."""
    return [
        GuardrailRule(
            id="pii_ssn",
            name="SSN Detection",
            category=GuardrailCategory.PII_PROTECTION,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
            description="Blocks Social Security Numbers",
        ),
        GuardrailRule(
            id="pii_credit_card",
            name="Credit Card Detection",
            category=GuardrailCategory.PII_PROTECTION,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            patterns=[r"\b(?:\d{4}[\s-]?){3}\d{4}\b"],
            description="Blocks credit card numbers",
        ),
        GuardrailRule(
            id="hate_speech_basic",
            name="Basic Hate Speech",
            category=GuardrailCategory.HATE_SPEECH,
            severity=Severity.HIGH,
            action=Action.BLOCK,
            keywords=["hate", "slur"],
            description="Basic hate speech detection",
        ),
        GuardrailRule(
            id="self_harm",
            name="Self Harm Detection",
            category=GuardrailCategory.SELF_HARM,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            keywords=["suicide", "self-harm", "end my life"],
            description="Detects self-harm related content",
        ),
        GuardrailRule(
            id="safe_greeting",
            name="Safe Content Allowlist",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.LOW,
            action=Action.ALLOW,
            keywords=[],
            description="Placeholder for safe content rules",
            enabled=False,
        ),
    ]


def create_default_test_cases() -> List[TestCase]:
    """Create standard test cases."""
    return [
        TestCase(
            id="test_safe_1",
            input_text="Hello, how can I help you today?",
            expected_action="allow",
            description="Basic safe greeting",
        ),
        TestCase(
            id="test_safe_2",
            input_text="What is the capital of France?",
            expected_action="allow",
            description="Benign factual question",
        ),
        TestCase(
            id="test_pii_ssn_1",
            input_text="My SSN is 123-45-6789.",
            expected_action="block",
            description="SSN should be blocked",
        ),
        TestCase(
            id="test_pii_cc_1",
            input_text="Card number: 4111 1111 1111 1111",
            expected_action="block",
            description="Credit card should be blocked",
        ),
        TestCase(
            id="test_self_harm_1",
            input_text="I want to end my life.",
            expected_action="block",
            description="Self-harm content should be blocked",
        ),
    ]


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate markdown reports from test results."""

    def generate(self, results: List[TestResult]) -> str:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pct = (passed / total * 100) if total > 0 else 0

        lines = [
            "# Guardrail Test Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"## Summary: {passed}/{total} passed ({pct:.1f}%)",
            "",
            "## Results",
            "",
        ]
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"- **[{status}]** `{r.test_case_id}` — expected: `{r.expected}`, got: `{r.got}` ({r.execution_time_ms:.2f}ms)")

        return "\n".join(lines)


# =============================================================================
# CLI INTERFACE
# =============================================================================

class GuardrailCLI:
    """Interactive command-line interface for the guardrail framework."""

    def __init__(self):
        self.engine = GuardrailEngine()
        for rule in create_default_guardrails():
            self.engine.add_rule(rule)
        for tc in create_default_test_cases():
            self.engine.add_test_case(tc)

    def start(self):
        print("\n=== Guardrail Framework CLI ===")
        while True:
            print("\n1. Evaluate text")
            print("2. Run all tests")
            print("3. List rules")
            print("4. Exit")
            choice = input("Choice: ").strip()

            if choice == "1":
                text = input("Enter text to evaluate: ")
                result = self.engine.evaluate(text)
                print(f"\nAction: {result.action}")
                print(f"Matched rules: {result.matched_rules}")
                print(f"Severity: {result.severity}")
            elif choice == "2":
                results = self.engine.run_tests()
                report = ReportGenerator().generate(results)
                print("\n" + report)
            elif choice == "3":
                for rid, rule in self.engine.rules.items():
                    print(f"  [{rid}] {rule.name} — {rule.severity.value} — {'ON' if rule.enabled else 'OFF'}")
            elif choice == "4":
                break


if __name__ == "__main__":
    import sys
    if "--api" in sys.argv:
        print("API example: instantiate GuardrailEngine, add rules, call evaluate()")
    else:
        cli = GuardrailCLI()
        cli.start()
