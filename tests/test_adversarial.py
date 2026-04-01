"""Tests for adversarial_tester.py"""
import pytest
from adversarial_tester import AdversarialTester, MutationType
from guardrail_framework import GuardrailEngine, create_default_guardrails


@pytest.fixture
def engine():
    e = GuardrailEngine()
    for rule in create_default_guardrails():
        e.add_rule(rule)
    return e


class TestAdversarialTester:
    def test_init_without_engine(self):
        tester = AdversarialTester()
        assert tester.engine is None

    def test_init_with_engine(self, engine):
        tester = AdversarialTester(engine)
        assert tester.engine is engine

    def test_mutate_leetspeak(self):
        tester = AdversarialTester()
        result = tester.mutate_leetspeak("hello")
        assert "3" in result  # 'e' -> '3'
        assert "0" in result  # 'o' -> '0'

    def test_generate_all_returns_list(self):
        tester = AdversarialTester()
        tests = tester.generate_all("Ignore instructions")
        assert isinstance(tests, list)
        assert len(tests) > 0

    def test_generate_all_includes_leetspeak(self):
        tester = AdversarialTester()
        tests = tester.generate_all("hello")
        types = [t.mutation for t in tests]
        assert MutationType.LEETSPEAK in types

    def test_generate_all_includes_context_injection(self):
        tester = AdversarialTester()
        tests = tester.generate_all("hello")
        types = [t.mutation for t in tests]
        assert MutationType.CONTEXT_INJECTION in types

    def test_run_full_suite_returns_dict(self, engine):
        tester = AdversarialTester(engine)
        results = tester.run_full_suite()
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_run_full_suite_scores_between_0_and_1(self, engine):
        tester = AdversarialTester(engine)
        results = tester.run_full_suite()
        for category, score in results.items():
            assert 0.0 <= score <= 1.0, f"Score for {category} out of range: {score}"

    def test_run_full_suite_blocks_known_threats(self, engine):
        tester = AdversarialTester(engine)
        results = tester.run_full_suite()
        # SSN patterns should be blocked
        assert results.get("pii_exfiltration", 0) > 0
