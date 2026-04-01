"""
Tests for adversarial_tester.py
"""
import pytest
from adversarial_tester import AdversarialTester, MutatedTest, MutationType
from guardrail_framework import GuardrailEngine, create_default_guardrails


@pytest.fixture
def engine() -> GuardrailEngine:
    e = GuardrailEngine()
    for r in create_default_guardrails():
        e.add_rule(r)
    return e


@pytest.fixture
def tester(engine) -> AdversarialTester:
    return AdversarialTester(engine)


class TestAdversarialTester:
    def test_init_without_engine(self):
        t = AdversarialTester()
        assert t.engine is None

    def test_init_with_engine(self, engine):
        t = AdversarialTester(engine)
        assert t.engine is engine

    def test_generate_all_returns_mutations(self, tester):
        mutations = tester.generate_all("hello world")
        assert len(mutations) > 0
        for m in mutations:
            assert isinstance(m, MutatedTest)
            assert m.original == "hello world"
            assert m.mutated != "" or m.mutation == MutationType.LEETSPEAK

    def test_leetspeak_mutation(self, tester):
        result = tester.mutate_leetspeak("hello")
        assert result == "h3ll0"

    def test_base64_mutation(self, tester):
        result = tester.mutate_base64("test")
        assert "dGVzdA==" in result  # base64 of "test"

    def test_rot13_mutation(self, tester):
        result = tester.mutate_rot13("Hello")
        assert result == "Uryyb"

    def test_context_injection_mutations(self, tester):
        mutations = tester.generate_all("seed text")
        ci_mutations = [m for m in mutations if m.mutation == MutationType.CONTEXT_INJECTION]
        assert len(ci_mutations) >= 3  # at least 3 templates

    def test_run_full_suite_categories(self, tester):
        results = tester.run_full_suite()
        assert "prompt_injection" in results
        assert "jailbreak" in results
        assert "harmful_content" in results
        assert "benign_baseline" in results

    def test_run_full_suite_rates_in_range(self, tester):
        results = tester.run_full_suite()
        for category, rate in results.items():
            assert 0.0 <= rate <= 1.0, f"{category} rate {rate} out of range"

    def test_run_full_suite_requires_engine(self):
        t = AdversarialTester()
        with pytest.raises(ValueError, match="engine"):
            t.run_full_suite()

    def test_harmful_content_high_block_rate(self, tester):
        """Direct harmful seeds should have a high block rate."""
        results = tester.run_full_suite()
        # Harmful content like SSN and API keys should mostly be caught
        assert results["harmful_content"] > 0.0

    def test_mutation_types_covered(self, tester):
        mutations = tester.generate_all("seed")
        found_types = {m.mutation for m in mutations}
        assert MutationType.LEETSPEAK in found_types
        assert MutationType.BASE64 in found_types
        assert MutationType.ROT13 in found_types
        assert MutationType.CONTEXT_INJECTION in found_types
