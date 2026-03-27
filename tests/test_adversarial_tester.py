"""
Tests for adversarial_tester.py
Covers: AdversarialTestGenerator all mutation types,
        generate_all(), run_against_engine().
"""

import pytest

from adversarial_tester import (
    AdversarialTestGenerator,
    MutatedTestCase,
    MutationType,
)
from guardrail_framework import (
    Action,
    GuardrailCategory,
    GuardrailEngine,
    GuardrailRule,
    Severity,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    return AdversarialTestGenerator()


def _make_engine_block_keyword(keyword):
    engine = GuardrailEngine()
    rule = GuardrailRule(
        id="block_kw",
        name="block keyword",
        category=GuardrailCategory.CUSTOM,
        severity=Severity.HIGH,
        action=Action.BLOCK,
        keywords=[keyword],
    )
    engine.add_rule(rule)
    return engine


# ---------------------------------------------------------------------------
# MutatedTestCase dataclass
# ---------------------------------------------------------------------------

class TestMutatedTestCase:
    def test_fields(self, generator):
        case = MutatedTestCase(
            original="hello",
            mutated="h3ll0",
            mutation_type=MutationType.LEETSPEAK,
            mutation_description="Full leetspeak",
            expected_behavior="should_block",
        )
        assert case.original == "hello"
        assert case.mutated == "h3ll0"
        assert case.mutation_type == MutationType.LEETSPEAK


# ---------------------------------------------------------------------------
# generate_leetspeak()
# ---------------------------------------------------------------------------

class TestGenerateLeetspeak:
    def test_returns_one_case(self, generator):
        cases = generator.generate_leetspeak("hello world")
        assert len(cases) == 1

    def test_mutation_type(self, generator):
        cases = generator.generate_leetspeak("hello")
        assert cases[0].mutation_type == MutationType.LEETSPEAK

    def test_leet_substitution_applied(self, generator):
        cases = generator.generate_leetspeak("eat")
        # e -> 3, a -> @, t -> 7
        assert cases[0].mutated == "3@7"

    def test_original_preserved(self, generator):
        cases = generator.generate_leetspeak("hello")
        assert cases[0].original == "hello"

    def test_expected_behavior_propagated(self, generator):
        cases = generator.generate_leetspeak("hello", expected="should_block")
        assert cases[0].expected_behavior == "should_block"


# ---------------------------------------------------------------------------
# generate_spacing()
# ---------------------------------------------------------------------------

class TestGenerateSpacing:
    def test_returns_one_case(self, generator):
        cases = generator.generate_spacing("hello world")
        assert len(cases) == 1

    def test_mutation_type(self, generator):
        cases = generator.generate_spacing("hello world")
        assert cases[0].mutation_type == MutationType.SPACING

    def test_first_word_spaced(self, generator):
        cases = generator.generate_spacing("abc def")
        # "abc" -> "a b c", then " def"
        assert "a b c" in cases[0].mutated

    def test_empty_string_returns_empty(self, generator):
        cases = generator.generate_spacing("")
        assert cases == []


# ---------------------------------------------------------------------------
# generate_case_variations()
# ---------------------------------------------------------------------------

class TestGenerateCaseVariations:
    def test_returns_two_cases(self, generator):
        cases = generator.generate_case_variations("Hello World")
        assert len(cases) == 2

    def test_uppercase_case(self, generator):
        cases = generator.generate_case_variations("Hello World")
        upper = [c for c in cases if c.mutation_description == "All uppercase"]
        assert len(upper) == 1
        assert upper[0].mutated == "HELLO WORLD"

    def test_lowercase_case(self, generator):
        cases = generator.generate_case_variations("Hello World")
        lower = [c for c in cases if c.mutation_description == "All lowercase"]
        assert len(lower) == 1
        assert lower[0].mutated == "hello world"

    def test_mutation_type(self, generator):
        cases = generator.generate_case_variations("test")
        for c in cases:
            assert c.mutation_type == MutationType.CASE


# ---------------------------------------------------------------------------
# generate_homoglyphs()
# ---------------------------------------------------------------------------

class TestGenerateHomoglyphs:
    def test_returns_one_case(self, generator):
        cases = generator.generate_homoglyphs("apple")
        assert len(cases) == 1

    def test_mutation_type(self, generator):
        cases = generator.generate_homoglyphs("apple")
        assert cases[0].mutation_type == MutationType.HOMOGLYPH

    def test_homoglyph_applied(self, generator):
        # 'a' -> Cyrillic 'а'
        cases = generator.generate_homoglyphs("a")
        assert cases[0].mutated != "a"

    def test_text_without_replaceable_chars(self, generator):
        cases = generator.generate_homoglyphs("111")
        # No homoglyphs for digits, mutated == original
        assert cases[0].mutated == "111"


# ---------------------------------------------------------------------------
# generate_context_injections()
# ---------------------------------------------------------------------------

class TestGenerateContextInjections:
    def test_returns_multiple_cases(self, generator):
        cases = generator.generate_context_injections("bad content")
        assert len(cases) >= 1

    def test_mutation_type(self, generator):
        cases = generator.generate_context_injections("bad content")
        for c in cases:
            assert c.mutation_type == MutationType.CONTEXT_INJECTION

    def test_original_embedded_in_mutated(self, generator):
        cases = generator.generate_context_injections("bad content")
        for c in cases:
            assert "bad content" in c.mutated

    def test_context_wraps_text(self, generator):
        cases = generator.generate_context_injections("test")
        # At least one should contain a wrapper phrase
        all_mutated = " ".join(c.mutated for c in cases)
        assert "test" in all_mutated


# ---------------------------------------------------------------------------
# generate_typos()
# ---------------------------------------------------------------------------

class TestGenerateTypos:
    def test_returns_up_to_3_cases(self, generator):
        cases = generator.generate_typos("hello world foo bar")
        assert len(cases) <= 3

    def test_mutation_type(self, generator):
        cases = generator.generate_typos("hello world foo bar")
        for c in cases:
            assert c.mutation_type == MutationType.TYPO

    def test_short_words_skipped(self, generator):
        # Words of 3 chars or fewer are skipped
        cases = generator.generate_typos("a bb ccc")
        assert len(cases) == 0

    def test_mutated_length_preserved(self, generator):
        cases = generator.generate_typos("hello world")
        for c in cases:
            # swap doesn't change length
            assert len(c.mutated) == len(c.original)


# ---------------------------------------------------------------------------
# generate_all()
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_returns_list_of_mutated_cases(self, generator):
        cases = generator.generate_all("bad content")
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_all_mutation_types_present(self, generator):
        cases = generator.generate_all("bad content here")
        types = {c.mutation_type for c in cases}
        expected = {
            MutationType.LEETSPEAK,
            MutationType.SPACING,
            MutationType.CASE,
            MutationType.HOMOGLYPH,
            MutationType.CONTEXT_INJECTION,
        }
        assert expected.issubset(types)

    def test_expected_behavior_block(self, generator):
        cases = generator.generate_all("dangerous text", expected_block=True)
        for c in cases:
            assert c.expected_behavior == "should_block"

    def test_expected_behavior_allow(self, generator):
        cases = generator.generate_all("safe text", expected_block=False)
        for c in cases:
            assert c.expected_behavior == "should_allow"


# ---------------------------------------------------------------------------
# run_against_engine()
# ---------------------------------------------------------------------------

class TestRunAgainstEngine:
    def test_returns_dict_with_expected_keys(self, generator):
        engine = GuardrailEngine()
        cases = generator.generate_all("safe text", expected_block=False)
        results = generator.run_against_engine(cases, engine)
        for key in ("total", "evaded", "blocked", "evasions"):
            assert key in results

    def test_total_matches_cases(self, generator):
        engine = GuardrailEngine()
        cases = generator.generate_all("text", expected_block=False)
        results = generator.run_against_engine(cases, engine)
        assert results["total"] == len(cases)

    def test_engine_blocks_all_should_block(self, generator):
        # Engine that blocks "hate" keyword
        engine = _make_engine_block_keyword("hate")
        # Generate mutations of a text with the keyword
        seed = "hate speech"
        # Use case variations which will keep the keyword visible
        cases = generator.generate_case_variations(seed)
        # Mark all as should_block
        for c in cases:
            c.expected_behavior = "should_block"

        results = generator.run_against_engine(cases, engine)
        # Case-insensitive engine should block both upper and lower
        assert results["blocked"] > 0

    def test_evasions_list_populated_when_evaded(self, generator):
        # Engine with no rules — everything allowed
        engine = GuardrailEngine()
        cases = generator.generate_all("dangerous text", expected_block=True)
        results = generator.run_against_engine(cases, engine)
        # All should be evaded since engine allows everything
        assert results["evaded"] == len(cases)
        assert len(results["evasions"]) == results["evaded"]

    def test_evasion_record_has_required_keys(self, generator):
        engine = GuardrailEngine()
        cases = [generator.generate_leetspeak("bad", "should_block")[0]]
        results = generator.run_against_engine(cases, engine)
        if results["evasions"]:
            evasion = results["evasions"][0]
            assert "original" in evasion
            assert "mutated" in evasion
            assert "mutation_type" in evasion

    def test_no_evasions_when_all_allowed(self, generator):
        engine = GuardrailEngine()
        # expected_block=False -> expected "should_allow", engine allows -> not evaded
        cases = generator.generate_all("safe text", expected_block=False)
        results = generator.run_against_engine(cases, engine)
        assert results["evaded"] == 0
        assert results["evasions"] == []
