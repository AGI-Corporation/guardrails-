"""
Tests for adversarial_tester.py
Covers: AdversarialTestGenerator (mutation methods, run_against_engine).
"""

import pytest

from adversarial_tester import AdversarialTestGenerator, MutatedTestCase, MutationType
from guardrail_framework import (
    GuardrailEngine,
    GuardrailRule,
    GuardrailCategory,
    Severity,
    Action,
    create_default_guardrails,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    e = GuardrailEngine()
    for rule in create_default_guardrails():
        e.add_rule(rule)
    return e


@pytest.fixture
def gen():
    return AdversarialTestGenerator()


SEED = "end my life"


# ── generate_typos ────────────────────────────────────────────────────────

class TestGenerateTypos:

    def test_returns_list(self, gen):
        cases = gen.generate_typos(SEED)
        assert isinstance(cases, list)

    def test_mutation_type_is_typo(self, gen):
        cases = gen.generate_typos(SEED)
        for c in cases:
            assert c.mutation_type == MutationType.TYPO

    def test_mutated_text_differs_from_original(self, gen):
        cases = gen.generate_typos(SEED)
        assert any(c.mutated != SEED for c in cases)


# ── generate_leetspeak ────────────────────────────────────────────────────

class TestGenerateLeetspeak:

    def test_returns_list(self, gen):
        cases = gen.generate_leetspeak(SEED)
        assert isinstance(cases, list)

    def test_mutation_type_is_leetspeak(self, gen):
        for c in gen.generate_leetspeak(SEED):
            assert c.mutation_type == MutationType.LEETSPEAK


# ── generate_spacing ──────────────────────────────────────────────────────

class TestGenerateSpacing:

    def test_returns_list(self, gen):
        cases = gen.generate_spacing(SEED)
        assert isinstance(cases, list)

    def test_mutation_type_is_spacing(self, gen):
        for c in gen.generate_spacing(SEED):
            assert c.mutation_type == MutationType.SPACING


# ── generate_case_variations ──────────────────────────────────────────────

class TestGenerateCaseVariations:

    def test_returns_list(self, gen):
        cases = gen.generate_case_variations(SEED)
        assert isinstance(cases, list)

    def test_mutation_type_correct(self, gen):
        for c in gen.generate_case_variations(SEED):
            assert c.mutation_type == MutationType.CASE

    def test_all_caps_included(self, gen):
        cases = gen.generate_case_variations(SEED)
        texts = [c.mutated for c in cases]
        assert any(t == t.upper() for t in texts)

    def test_lower_included(self, gen):
        cases = gen.generate_case_variations(SEED)
        texts = [c.mutated for c in cases]
        assert any(t == t.lower() for t in texts)


# ── generate_all ──────────────────────────────────────────────────────────

class TestGenerateAll:

    def test_returns_non_empty_list(self, gen):
        cases = gen.generate_all(SEED, expected_block=True)
        assert len(cases) > 0

    def test_all_are_mutated_test_cases(self, gen):
        cases = gen.generate_all(SEED, expected_block=True)
        for c in cases:
            assert isinstance(c, MutatedTestCase)

    def test_original_text_tracked(self, gen):
        cases = gen.generate_all(SEED, expected_block=True)
        for c in cases:
            assert c.original == SEED

    def test_expected_block_propagated(self, gen):
        cases = gen.generate_all(SEED, expected_block=True)
        for c in cases:
            assert c.expected_behavior == "should_block"


# ── run_against_engine ────────────────────────────────────────────────────

class TestRunAgainstEngine:

    def test_stats_keys_present(self, gen, engine):
        cases = gen.generate_all(SEED, expected_block=True)
        stats = gen.run_against_engine(cases, engine)
        assert "total" in stats
        assert "blocked" in stats
        assert "evaded" in stats
        assert "evasions" in stats

    def test_total_equals_cases(self, gen, engine):
        cases = gen.generate_all(SEED, expected_block=True)
        stats = gen.run_against_engine(cases, engine)
        assert stats["total"] == len(cases)

    def test_blocked_plus_evaded_equals_total(self, gen, engine):
        cases = gen.generate_all(SEED, expected_block=True)
        stats = gen.run_against_engine(cases, engine)
        assert stats["blocked"] + stats["evaded"] == stats["total"]

    def test_safe_seed_not_blocked(self, gen, engine):
        """Cases expected to be allowed have no 'evasions' counted."""
        cases = gen.generate_all("hello world", expected_block=False)
        stats = gen.run_against_engine(cases, engine)
        # With expected_block=False, expected_behavior="should_allow"; evasion
        # branch only triggers for "should_block" cases, so evaded should be 0.
        assert stats["evaded"] == 0

    def test_evasion_list_is_list_of_dicts(self, gen, engine):
        cases = gen.generate_all(SEED, expected_block=True)
        stats = gen.run_against_engine(cases, engine)
        for e in stats["evasions"]:
            assert "mutated" in e or "mutation_type" in e
