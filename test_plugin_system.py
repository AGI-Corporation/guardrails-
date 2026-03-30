"""
Tests for plugin_system.py
Covers PluginEngine, EntropyPlugin, RepetitionPlugin, LengthPlugin,
and PluginResult.
"""

import math
import pytest
from plugin_system import (
    PluginResult,
    GuardrailPlugin,
    PluginEngine,
    EntropyPlugin,
    RepetitionPlugin,
    LengthPlugin,
    create_default_plugin_engine,
)


# ── PluginResult ─────────────────────────────────────────────────────────────

class TestPluginResult:
    def test_defaults(self):
        result = PluginResult(
            plugin_name="test",
            passed=True,
            score=0.0,
            action="allow",
        )
        assert result.details == {}
        assert result.execution_time_ms == 0.0
        assert result.error is None

    def test_custom_values(self):
        result = PluginResult(
            plugin_name="x",
            passed=False,
            score=0.8,
            action="block",
            details={"key": "val"},
            error="oops",
        )
        assert result.passed is False
        assert result.score == 0.8
        assert result.action == "block"
        assert result.details == {"key": "val"}
        assert result.error == "oops"


# ── EntropyPlugin ─────────────────────────────────────────────────────────────

class TestEntropyPlugin:
    def test_name_and_description(self):
        plugin = EntropyPlugin()
        assert plugin.name == "entropy_detector"
        assert "entropy" in plugin.description.lower()

    def test_short_text_allows(self):
        plugin = EntropyPlugin()
        result = plugin.evaluate("hello world")
        assert result.action == "allow"
        assert result.passed is True

    def test_low_entropy_long_word_allows(self):
        plugin = EntropyPlugin()
        # Low-entropy string of length >= 20
        result = plugin.evaluate("aaaaaaaaaaaaaaaaaaaaaa")
        assert result.action == "allow"

    def test_high_entropy_long_word_warns(self):
        plugin = EntropyPlugin(threshold=2.0)
        # Random-looking token of length >= 20
        result = plugin.evaluate("aB3xQ7mN2pL8kR4tY1sW5vZ")
        assert result.action == "warn"
        assert result.passed is False

    def test_score_bounded(self):
        plugin = EntropyPlugin(threshold=2.0)
        # Multiple high-entropy words
        words = " ".join(["aB3xQ7mN2pL8kR4tY1sW5vZ"] * 10)
        result = plugin.evaluate(words)
        assert 0.0 <= result.score <= 1.0

    def test_custom_threshold(self):
        plugin = EntropyPlugin(threshold=10.0)  # Impossibly high threshold
        result = plugin.evaluate("aB3xQ7mN2pL8kR4tY1sW5vZ9eU")
        assert result.action == "allow"  # Won't exceed threshold of 10

    def test_empty_text_allows(self):
        plugin = EntropyPlugin()
        result = plugin.evaluate("")
        assert result.action == "allow"

    def test_calculate_entropy_empty_string(self):
        plugin = EntropyPlugin()
        assert plugin._calculate_entropy("") == 0.0

    def test_calculate_entropy_single_char(self):
        plugin = EntropyPlugin()
        assert plugin._calculate_entropy("a") == pytest.approx(0.0)

    def test_calculate_entropy_two_distinct(self):
        plugin = EntropyPlugin()
        entropy = plugin._calculate_entropy("ab")
        assert entropy == pytest.approx(1.0)


# ── RepetitionPlugin ──────────────────────────────────────────────────────────

class TestRepetitionPlugin:
    def test_name_and_description(self):
        plugin = RepetitionPlugin()
        assert plugin.name == "repetition_detector"
        assert "repetition" in plugin.description.lower()

    def test_short_text_allows(self):
        plugin = RepetitionPlugin()
        result = plugin.evaluate("too short")
        assert result.action == "allow"
        assert result.passed is True

    def test_low_repetition_allows(self):
        plugin = RepetitionPlugin()
        text = "the quick brown fox jumps over the lazy dog near the river bank"
        result = plugin.evaluate(text)
        assert result.action == "allow"

    def test_high_repetition_warns(self):
        plugin = RepetitionPlugin(max_repetition_ratio=0.3)
        # ~80% repeated words
        text = " ".join(["repeat"] * 15 + ["other"] * 3)
        result = plugin.evaluate(text)
        assert result.action in ("warn", "block")
        assert result.passed is False

    def test_extreme_repetition_blocks(self):
        plugin = RepetitionPlugin()
        # All the same word (>80% repetition)
        text = " ".join(["spam"] * 20)
        result = plugin.evaluate(text)
        assert result.action == "block"
        assert result.passed is False

    def test_score_is_repetition_ratio(self):
        plugin = RepetitionPlugin(max_repetition_ratio=0.3)
        text = " ".join(["word"] * 15 + ["other"] * 5)
        result = plugin.evaluate(text)
        if not result.passed:
            assert "repetition_ratio" in result.details

    def test_boundary_ten_words(self):
        """Exactly 10 words — plugin should actually evaluate."""
        plugin = RepetitionPlugin(max_repetition_ratio=0.3)
        text = " ".join(["a"] * 10)
        result = plugin.evaluate(text)
        # All same word → high repetition
        assert result.action in ("warn", "block")


# ── LengthPlugin ──────────────────────────────────────────────────────────────

class TestLengthPlugin:
    def test_name_and_description(self):
        plugin = LengthPlugin(max_chars=500)
        assert plugin.name == "length_guard"
        assert "500" in plugin.description

    def test_text_within_limit_allows(self):
        plugin = LengthPlugin(max_chars=100)
        result = plugin.evaluate("short text")
        assert result.action == "allow"
        assert result.passed is True

    def test_text_at_limit_allows(self):
        plugin = LengthPlugin(max_chars=10)
        result = plugin.evaluate("a" * 10)
        assert result.action == "allow"

    def test_text_over_limit_blocks(self):
        plugin = LengthPlugin(max_chars=10)
        result = plugin.evaluate("a" * 11)
        assert result.action == "block"
        assert result.passed is False
        assert result.score == 1.0

    def test_details_contain_length_info(self):
        plugin = LengthPlugin(max_chars=5)
        result = plugin.evaluate("a" * 10)
        assert result.details["length"] == 10
        assert result.details["max_allowed"] == 5

    def test_empty_text_allows(self):
        plugin = LengthPlugin(max_chars=100)
        result = plugin.evaluate("")
        assert result.action == "allow"

    def test_custom_max_chars(self):
        plugin = LengthPlugin(max_chars=1)
        assert plugin.evaluate("a").action == "allow"
        assert plugin.evaluate("ab").action == "block"


# ── PluginEngine ──────────────────────────────────────────────────────────────

class TestPluginEngine:
    def test_register_and_unregister(self):
        engine = PluginEngine()
        plugin = LengthPlugin(max_chars=100)
        engine.register(plugin)
        assert "length_guard" in engine.plugins
        engine.unregister("length_guard")
        assert "length_guard" not in engine.plugins

    def test_unregister_nonexistent_no_error(self):
        engine = PluginEngine()
        engine.unregister("does_not_exist")  # Should not raise

    def test_evaluate_all_empty_returns_empty_list(self):
        engine = PluginEngine()
        results = engine.evaluate_all("some text")
        assert results == []

    def test_evaluate_all_returns_result_per_plugin(self):
        engine = PluginEngine()
        engine.register(LengthPlugin(max_chars=100))
        engine.register(RepetitionPlugin())
        results = engine.evaluate_all("hello world")
        assert len(results) == 2

    def test_evaluate_all_sets_execution_time(self):
        engine = PluginEngine()
        engine.register(LengthPlugin())
        results = engine.evaluate_all("hello")
        assert results[0].execution_time_ms >= 0

    def test_evaluate_all_catches_plugin_exceptions(self):
        class BrokenPlugin(GuardrailPlugin):
            @property
            def name(self):
                return "broken"

            @property
            def description(self):
                return "always raises"

            def evaluate(self, text, context=None):
                raise ValueError("intentional error")

        engine = PluginEngine()
        engine.register(BrokenPlugin())
        results = engine.evaluate_all("test")
        assert len(results) == 1
        assert results[0].error == "intentional error"
        assert results[0].action == "allow"  # Fail-open
        assert results[0].passed is True

    def test_get_final_action_all_allow(self):
        engine = PluginEngine()
        results = [
            PluginResult("a", True, 0.0, "allow"),
            PluginResult("b", True, 0.0, "allow"),
        ]
        assert engine.get_final_action(results) == "allow"

    def test_get_final_action_warn_wins_over_allow(self):
        engine = PluginEngine()
        results = [
            PluginResult("a", True, 0.0, "allow"),
            PluginResult("b", False, 0.5, "warn"),
        ]
        assert engine.get_final_action(results) == "warn"

    def test_get_final_action_block_wins_over_warn(self):
        engine = PluginEngine()
        results = [
            PluginResult("a", False, 0.5, "warn"),
            PluginResult("b", False, 1.0, "block"),
        ]
        assert engine.get_final_action(results) == "block"

    def test_get_final_action_empty_list_allows(self):
        engine = PluginEngine()
        assert engine.get_final_action([]) == "allow"

    def test_evaluate_all_with_context(self):
        engine = PluginEngine()
        engine.register(LengthPlugin(max_chars=100))
        context = {"user_id": "alice"}
        results = engine.evaluate_all("hello world", context=context)
        assert len(results) == 1


# ── create_default_plugin_engine ──────────────────────────────────────────────

class TestCreateDefaultPluginEngine:
    def test_returns_plugin_engine(self):
        engine = create_default_plugin_engine()
        assert isinstance(engine, PluginEngine)

    def test_has_entropy_plugin(self):
        engine = create_default_plugin_engine()
        assert "entropy_detector" in engine.plugins

    def test_has_repetition_plugin(self):
        engine = create_default_plugin_engine()
        assert "repetition_detector" in engine.plugins

    def test_has_length_plugin(self):
        engine = create_default_plugin_engine()
        assert "length_guard" in engine.plugins

    def test_default_engine_evaluates_text(self):
        engine = create_default_plugin_engine()
        results = engine.evaluate_all("Hello, this is normal text.")
        assert len(results) == 3
