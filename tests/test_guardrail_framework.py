"""
Tests for guardrail_framework.py
Covers: GuardrailEngine, GuardrailRule, EvaluationResult, TestCase/TestResult,
        ReportGenerator, default rules, default test cases, export.
"""

import os
import json
import tempfile
import pytest

from guardrail_framework import (
    GuardrailEngine,
    GuardrailRule,
    GuardrailCategory,
    Severity,
    Action,
    EvaluationResult,
    TestCase,
    TestResult,
    ReportGenerator,
    create_default_guardrails,
    create_default_test_cases,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine_with_defaults():
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    for tc in create_default_test_cases():
        engine.add_test_case(tc)
    return engine


@pytest.fixture
def minimal_engine():
    """Engine with a single simple keyword rule."""
    engine = GuardrailEngine()
    engine.add_rule(GuardrailRule(
        id="kw_test",
        name="Test keyword rule",
        category=GuardrailCategory.CUSTOM,
        severity=Severity.MEDIUM,
        action=Action.BLOCK,
        keywords=["forbidden"],
    ))
    return engine


# ── GuardrailEngine — basic operations ───────────────────────────────────

class TestGuardrailEngineBasic:

    def test_add_and_remove_rule(self):
        engine = GuardrailEngine()
        rule = GuardrailRule(
            id="r1",
            name="Rule one",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.LOW,
            action=Action.WARN,
            keywords=["test"],
        )
        engine.add_rule(rule)
        assert "r1" in engine.rules

        engine.remove_rule("r1")
        assert "r1" not in engine.rules

    def test_remove_nonexistent_rule_is_noop(self):
        engine = GuardrailEngine()
        engine.remove_rule("does_not_exist")  # should not raise

    def test_add_test_case(self):
        engine = GuardrailEngine()
        tc = TestCase(id="tc1", input_text="hello", expected_action="allow")
        engine.add_test_case(tc)
        assert "tc1" in engine.test_cases


# ── GuardrailEngine — evaluate ────────────────────────────────────────────

class TestGuardrailEngineEvaluate:

    def test_safe_text_is_allowed(self, minimal_engine):
        result = minimal_engine.evaluate("Hello, world!")
        assert result.action == "allow"
        assert result.matched_rules == []
        assert result.severity == "none"

    def test_keyword_triggers_block(self, minimal_engine):
        result = minimal_engine.evaluate("This contains forbidden content.")
        assert result.action == "block"
        assert "kw_test" in result.matched_rules
        assert result.severity == "medium"

    def test_keyword_case_insensitive(self, minimal_engine):
        result = minimal_engine.evaluate("FORBIDDEN word here")
        assert result.action == "block"

    def test_disabled_rule_does_not_fire(self):
        engine = GuardrailEngine()
        engine.add_rule(GuardrailRule(
            id="disabled",
            name="Disabled rule",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.HIGH,
            action=Action.BLOCK,
            keywords=["secret"],
            enabled=False,
        ))
        result = engine.evaluate("This contains secret info")
        assert result.action == "allow"
        assert result.matched_rules == []

    def test_regex_pattern_triggers_block(self):
        engine = GuardrailEngine()
        engine.add_rule(GuardrailRule(
            id="regex_rule",
            name="Regex rule",
            category=GuardrailCategory.PII_PROTECTION,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],
        ))
        result = engine.evaluate("My SSN is 123-45-6789.")
        assert result.action == "block"
        assert result.severity == "critical"

    def test_highest_severity_wins(self):
        engine = GuardrailEngine()
        engine.add_rule(GuardrailRule(
            id="low",
            name="Low rule",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.LOW,
            action=Action.WARN,
            keywords=["warning"],
        ))
        engine.add_rule(GuardrailRule(
            id="crit",
            name="Critical rule",
            category=GuardrailCategory.HARMFUL_CONTENT,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            keywords=["danger"],
        ))
        result = engine.evaluate("danger warning both present")
        assert result.severity == "critical"
        assert result.action == "block"

    def test_evaluation_result_has_timestamp(self, minimal_engine):
        result = minimal_engine.evaluate("test")
        assert result.timestamp  # not empty


# ── Default guardrails ────────────────────────────────────────────────────

class TestDefaultGuardrails:

    def test_ssn_blocked(self, engine_with_defaults):
        result = engine_with_defaults.evaluate("My SSN is 123-45-6789.")
        assert result.action == "block"

    def test_credit_card_blocked(self, engine_with_defaults):
        result = engine_with_defaults.evaluate("Card: 4111 1111 1111 1111")
        assert result.action == "block"

    def test_self_harm_blocked(self, engine_with_defaults):
        result = engine_with_defaults.evaluate("I want to end my life.")
        assert result.action == "block"

    def test_safe_greeting_allowed(self, engine_with_defaults):
        result = engine_with_defaults.evaluate("Hello, how can I help you today?")
        assert result.action == "allow"

    def test_factual_question_allowed(self, engine_with_defaults):
        result = engine_with_defaults.evaluate("What is the capital of France?")
        assert result.action == "allow"


# ── run_tests ─────────────────────────────────────────────────────────────

class TestRunTests:

    def test_all_default_tests_pass(self, engine_with_defaults):
        results = engine_with_defaults.run_tests()
        failed = [r for r in results if not r.passed]
        assert not failed, f"Failing tests: {[r.test_case_id for r in failed]}"

    def test_test_result_has_timing(self, engine_with_defaults):
        results = engine_with_defaults.run_tests()
        for r in results:
            assert r.execution_time_ms >= 0

    def test_failing_test_detected(self):
        engine = GuardrailEngine()
        # No rules, so everything is allowed; but test expects block
        tc = TestCase(id="bad_test", input_text="innocent text", expected_action="block")
        engine.add_test_case(tc)
        results = engine.run_tests()
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].got == "allow"
        assert results[0].expected == "block"


# ── ReportGenerator ───────────────────────────────────────────────────────

class TestReportGenerator:

    def test_report_contains_summary(self, engine_with_defaults):
        results = engine_with_defaults.run_tests()
        report = ReportGenerator().generate(results)
        assert "Summary" in report
        assert "passed" in report.lower() or "%" in report

    def test_report_shows_pass_fail(self, engine_with_defaults):
        results = engine_with_defaults.run_tests()
        report = ReportGenerator().generate(results)
        assert "PASS" in report

    def test_empty_report(self):
        report = ReportGenerator().generate([])
        assert "0/0" in report or "0%" in report


# ── Export ────────────────────────────────────────────────────────────────

class TestExport:

    def test_export_yaml(self, engine_with_defaults):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            engine_with_defaults.export_rules(path, fmt="yaml")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_json(self, engine_with_defaults):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine_with_defaults.export_rules(path, fmt="json")
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert len(data) == len(engine_with_defaults.rules)
        finally:
            os.unlink(path)
