"""
Tests for guardrail_framework.py
"""

import json
import os
import tempfile
import pytest
import yaml

from guardrail_framework import (
    Action,
    EvaluationResult,
    GuardrailCategory,
    GuardrailEngine,
    GuardrailRule,
    ReportGenerator,
    Severity,
    TestCase,
    TestResult,
    create_default_guardrails,
    create_default_test_cases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_with_defaults() -> GuardrailEngine:
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    return engine


def _block_rule(rule_id: str = "test_block", keywords=None, patterns=None) -> GuardrailRule:
    return GuardrailRule(
        id=rule_id,
        name="Test Block Rule",
        category=GuardrailCategory.HARMFUL_CONTENT,
        severity=Severity.HIGH,
        action=Action.BLOCK,
        keywords=keywords or ["forbidden"],
        patterns=patterns or [],
    )


# ---------------------------------------------------------------------------
# Enum smoke-tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_severity_values(self):
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"

    def test_action_values(self):
        assert Action.ALLOW.value == "allow"
        assert Action.BLOCK.value == "block"
        assert Action.WARN.value == "warn"
        assert Action.TRANSFORM.value == "transform"
        assert Action.LOG.value == "log"

    def test_category_values(self):
        assert GuardrailCategory.PII_PROTECTION.value == "pii_protection"
        assert GuardrailCategory.HATE_SPEECH.value == "hate_speech"
        assert GuardrailCategory.SELF_HARM.value == "self_harm"
        assert GuardrailCategory.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# GuardrailRule dataclass
# ---------------------------------------------------------------------------

class TestGuardrailRule:
    def test_defaults(self):
        rule = GuardrailRule(
            id="r1",
            name="Rule 1",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.LOW,
            action=Action.ALLOW,
        )
        assert rule.patterns == []
        assert rule.keywords == []
        assert rule.description == ""
        assert rule.enabled is True
        assert rule.metadata == {}

    def test_custom_fields(self):
        rule = GuardrailRule(
            id="r2",
            name="Rule 2",
            category=GuardrailCategory.VIOLENCE,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            patterns=[r"\bkill\b"],
            keywords=["violence"],
            description="Violence check",
            enabled=False,
            metadata={"owner": "team-a"},
        )
        assert rule.patterns == [r"\bkill\b"]
        assert rule.keywords == ["violence"]
        assert rule.enabled is False
        assert rule.metadata["owner"] == "team-a"


# ---------------------------------------------------------------------------
# GuardrailEngine — rule management
# ---------------------------------------------------------------------------

class TestGuardrailEngineRuleManagement:
    def test_add_and_retrieve_rule(self):
        engine = GuardrailEngine()
        rule = _block_rule()
        engine.add_rule(rule)
        assert "test_block" in engine.rules

    def test_remove_rule(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule())
        engine.remove_rule("test_block")
        assert "test_block" not in engine.rules

    def test_remove_nonexistent_rule_does_not_raise(self):
        engine = GuardrailEngine()
        engine.remove_rule("does_not_exist")  # should be silent

    def test_overwrite_rule(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule(rule_id="r1", keywords=["bad"]))
        engine.add_rule(_block_rule(rule_id="r1", keywords=["evil"]))
        assert engine.rules["r1"].keywords == ["evil"]

    def test_add_test_case(self):
        engine = GuardrailEngine()
        tc = TestCase(id="tc1", input_text="hello", expected_action="allow")
        engine.add_test_case(tc)
        assert "tc1" in engine.test_cases


# ---------------------------------------------------------------------------
# GuardrailEngine — evaluate
# ---------------------------------------------------------------------------

class TestGuardrailEngineEvaluate:
    def test_empty_engine_allows_everything(self):
        engine = GuardrailEngine()
        result = engine.evaluate("anything goes")
        assert result.action == "allow"
        assert result.matched_rules == []
        assert result.severity == "none"

    def test_keyword_match_blocks(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule(keywords=["forbidden"]))
        result = engine.evaluate("this is forbidden content")
        assert result.action == "block"
        assert "test_block" in result.matched_rules

    def test_keyword_match_case_insensitive(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule(keywords=["forbidden"]))
        result = engine.evaluate("This Is FORBIDDEN Content")
        assert result.action == "block"

    def test_pattern_match_blocks(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule(keywords=[], patterns=[r"\d{3}-\d{2}-\d{4}"]))
        result = engine.evaluate("My SSN is 123-45-6789")
        assert result.action == "block"

    def test_no_match_allows(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule(keywords=["forbidden"]))
        result = engine.evaluate("Hello, how are you?")
        assert result.action == "allow"
        assert result.matched_rules == []

    def test_disabled_rule_is_skipped(self):
        engine = GuardrailEngine()
        rule = _block_rule(keywords=["forbidden"])
        rule.enabled = False
        engine.add_rule(rule)
        result = engine.evaluate("forbidden content")
        assert result.action == "allow"

    def test_highest_severity_wins(self):
        engine = GuardrailEngine()
        low_rule = GuardrailRule(
            id="low_rule",
            name="Low",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.LOW,
            action=Action.WARN,
            keywords=["warn_word"],
        )
        high_rule = GuardrailRule(
            id="high_rule",
            name="High",
            category=GuardrailCategory.HARMFUL_CONTENT,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            keywords=["block_word"],
        )
        engine.add_rule(low_rule)
        engine.add_rule(high_rule)
        result = engine.evaluate("this has warn_word and block_word")
        assert result.action == "block"
        assert result.severity == "critical"

    def test_multiple_rules_matched(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule(rule_id="r1", keywords=["bad"]))
        engine.add_rule(_block_rule(rule_id="r2", keywords=["evil"]))
        result = engine.evaluate("this is bad and evil")
        assert "r1" in result.matched_rules
        assert "r2" in result.matched_rules

    def test_result_has_timestamp(self):
        engine = GuardrailEngine()
        result = engine.evaluate("hello")
        assert result.timestamp != ""

    def test_ssn_blocked_by_default_rules(self):
        engine = _make_engine_with_defaults()
        result = engine.evaluate("My SSN is 123-45-6789")
        assert result.action == "block"
        assert "pii_ssn" in result.matched_rules

    def test_credit_card_blocked(self):
        engine = _make_engine_with_defaults()
        result = engine.evaluate("Card: 4111 1111 1111 1111")
        assert result.action == "block"
        assert "pii_credit_card" in result.matched_rules

    def test_self_harm_blocked(self):
        engine = _make_engine_with_defaults()
        result = engine.evaluate("I want to end my life")
        assert result.action == "block"
        assert "self_harm" in result.matched_rules

    def test_safe_text_allowed(self):
        engine = _make_engine_with_defaults()
        result = engine.evaluate("Hello, what is the capital of France?")
        assert result.action == "allow"

    def test_hate_speech_keyword_blocked(self):
        engine = _make_engine_with_defaults()
        result = engine.evaluate("I hate everyone with slur language")
        assert result.action == "block"


# ---------------------------------------------------------------------------
# GuardrailEngine — run_tests
# ---------------------------------------------------------------------------

class TestGuardrailEngineRunTests:
    def test_run_tests_empty(self):
        engine = GuardrailEngine()
        results = engine.run_tests()
        assert results == []

    def test_run_tests_all_pass(self):
        engine = GuardrailEngine()
        for rule in create_default_guardrails():
            engine.add_rule(rule)
        for tc in create_default_test_cases():
            engine.add_test_case(tc)
        results = engine.run_tests()
        assert all(r.passed for r in results), [r for r in results if not r.passed]

    def test_test_result_fields(self):
        engine = GuardrailEngine()
        engine.add_test_case(TestCase(id="t1", input_text="hello", expected_action="allow"))
        results = engine.run_tests()
        assert len(results) == 1
        r = results[0]
        assert r.test_case_id == "t1"
        assert r.passed is True
        assert r.expected == "allow"
        assert r.got == "allow"
        assert r.execution_time_ms >= 0

    def test_test_result_failed(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule(keywords=["forbidden"]))
        engine.add_test_case(
            TestCase(id="fail_tc", input_text="forbidden text", expected_action="allow")
        )
        results = engine.run_tests()
        assert results[0].passed is False
        assert results[0].got == "block"


# ---------------------------------------------------------------------------
# GuardrailEngine — export_rules
# ---------------------------------------------------------------------------

class TestGuardrailEngineExport:
    def test_export_yaml(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule())
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            engine.export_rules(path, fmt="yaml")
            with open(path) as fh:
                data = yaml.safe_load(fh)
            assert "test_block" in data
        finally:
            os.unlink(path)

    def test_export_json(self):
        engine = GuardrailEngine()
        engine.add_rule(_block_rule())
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            engine.export_rules(path, fmt="json")
            with open(path) as fh:
                data = json.load(fh)
            assert "test_block" in data
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# create_default_guardrails / create_default_test_cases
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_guardrails_count(self):
        rules = create_default_guardrails()
        assert len(rules) >= 4

    def test_default_guardrails_have_required_fields(self):
        for rule in create_default_guardrails():
            assert rule.id
            assert rule.name
            assert isinstance(rule.category, GuardrailCategory)
            assert isinstance(rule.severity, Severity)
            assert isinstance(rule.action, Action)

    def test_default_test_cases_count(self):
        cases = create_default_test_cases()
        assert len(cases) >= 3

    def test_default_test_cases_expected_actions(self):
        cases = create_default_test_cases()
        actions = {tc.expected_action for tc in cases}
        assert "allow" in actions
        assert "block" in actions


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def _make_results(self, passed_flags):
        return [
            TestResult(
                test_case_id=f"tc{i}",
                passed=flag,
                expected="allow" if flag else "block",
                got="allow" if flag else "allow",
                matched_rules=[],
                execution_time_ms=1.0,
            )
            for i, flag in enumerate(passed_flags)
        ]

    def test_all_pass(self):
        rg = ReportGenerator()
        report = rg.generate(self._make_results([True, True]))
        assert "2/2" in report
        assert "100.0%" in report

    def test_some_fail(self):
        rg = ReportGenerator()
        report = rg.generate(self._make_results([True, False, True]))
        assert "2/3" in report
        assert "PASS" in report
        assert "FAIL" in report

    def test_empty_results(self):
        rg = ReportGenerator()
        report = rg.generate([])
        assert "0/0" in report

    def test_report_contains_test_ids(self):
        rg = ReportGenerator()
        results = self._make_results([True])
        results[0].test_case_id = "my_special_test"
        report = rg.generate(results)
        assert "my_special_test" in report


# ---------------------------------------------------------------------------
# GuardrailCLI initialization
# ---------------------------------------------------------------------------

class TestGuardrailCLI:
    def test_cli_initializes_with_default_rules(self):
        from guardrail_framework import GuardrailCLI
        cli = GuardrailCLI()
        assert len(cli.engine.rules) > 0

    def test_cli_initializes_with_default_test_cases(self):
        from guardrail_framework import GuardrailCLI
        cli = GuardrailCLI()
        assert len(cli.engine.test_cases) > 0

    def test_cli_engine_evaluates_correctly(self):
        from guardrail_framework import GuardrailCLI
        cli = GuardrailCLI()
        result = cli.engine.evaluate("My SSN is 123-45-6789")
        assert result.action == "block"
