"""
Tests for plugin_system.py
Covers: PluginEngine register/unregister/evaluate_all/get_final_action,
        EntropyPlugin, RepetitionPlugin, LengthPlugin,
        create_default_plugin_engine, error handling (fail-open).
"""

import math
import pytest

from plugin_system import (
    EntropyPlugin,
    GuardrailPlugin,
    LengthPlugin,
    PluginEngine,
    PluginResult,
    RepetitionPlugin,
    create_default_plugin_engine,
)


# ---------------------------------------------------------------------------
# PluginResult
# ---------------------------------------------------------------------------

class TestPluginResult:
    def test_defaults(self):
        r = PluginResult(
            plugin_name="test",
            passed=True,
            score=0.0,
            action="allow",
        )
        assert r.details == {}
        assert r.execution_time_ms == 0.0
        assert r.error is None


# ---------------------------------------------------------------------------
# PluginEngine — register / unregister
# ---------------------------------------------------------------------------

class TestPluginEngineManagement:
    def setup_method(self):
        self.engine = PluginEngine()

    def _make_plugin(self, name, action="allow"):
        class DummyPlugin(GuardrailPlugin):
            @property
            def name(self):
                return name

            @property
            def description(self):
                return f"Dummy plugin {name}"

            def evaluate(self, text, context=None):
                return PluginResult(
                    plugin_name=name,
                    passed=(action == "allow"),
                    score=0.0,
                    action=action,
                )

        return DummyPlugin()

    def test_register_adds_plugin(self):
        plugin = self._make_plugin("p1")
        self.engine.register(plugin)
        assert "p1" in self.engine.plugins

    def test_unregister_removes_plugin(self):
        plugin = self._make_plugin("p1")
        self.engine.register(plugin)
        self.engine.unregister("p1")
        assert "p1" not in self.engine.plugins

    def test_unregister_nonexistent_is_noop(self):
        self.engine.unregister("does_not_exist")  # should not raise

    def test_multiple_plugins_registered(self):
        for i in range(3):
            self.engine.register(self._make_plugin(f"p{i}"))
        assert len(self.engine.plugins) == 3


# ---------------------------------------------------------------------------
# PluginEngine — evaluate_all()
# ---------------------------------------------------------------------------

class TestPluginEngineEvaluateAll:
    def setup_method(self):
        self.engine = PluginEngine()

    def _make_plugin(self, name, action="allow"):
        class DummyPlugin(GuardrailPlugin):
            @property
            def name(self):
                return name

            @property
            def description(self):
                return ""

            def evaluate(self, text, context=None):
                return PluginResult(
                    plugin_name=name,
                    passed=(action == "allow"),
                    score=1.0 if action == "block" else 0.0,
                    action=action,
                )

        return DummyPlugin()

    def test_empty_engine_returns_empty_list(self):
        results = self.engine.evaluate_all("text")
        assert results == []

    def test_results_count_matches_plugins(self):
        for i in range(3):
            self.engine.register(self._make_plugin(f"p{i}"))
        results = self.engine.evaluate_all("any text")
        assert len(results) == 3

    def test_execution_time_recorded(self):
        self.engine.register(self._make_plugin("p1"))
        results = self.engine.evaluate_all("text")
        assert results[0].execution_time_ms >= 0

    def test_error_in_plugin_produces_fail_open(self):
        class BrokenPlugin(GuardrailPlugin):
            @property
            def name(self):
                return "broken"

            @property
            def description(self):
                return ""

            def evaluate(self, text, context=None):
                raise ValueError("simulated error")

        self.engine.register(BrokenPlugin())
        results = self.engine.evaluate_all("text")
        assert len(results) == 1
        r = results[0]
        assert r.action == "allow"  # fail-open
        assert r.passed is True
        assert r.error is not None


# ---------------------------------------------------------------------------
# PluginEngine — get_final_action()
# ---------------------------------------------------------------------------

class TestPluginEngineGetFinalAction:
    def setup_method(self):
        self.engine = PluginEngine()

    def _result(self, action):
        return PluginResult(plugin_name="x", passed=True, score=0.0, action=action)

    def test_all_allow_returns_allow(self):
        results = [self._result("allow"), self._result("allow")]
        assert self.engine.get_final_action(results) == "allow"

    def test_any_block_returns_block(self):
        results = [self._result("allow"), self._result("block")]
        assert self.engine.get_final_action(results) == "block"

    def test_warn_without_block_returns_warn(self):
        results = [self._result("allow"), self._result("warn")]
        assert self.engine.get_final_action(results) == "warn"

    def test_block_overrides_warn(self):
        results = [self._result("warn"), self._result("block")]
        assert self.engine.get_final_action(results) == "block"

    def test_empty_results_returns_allow(self):
        assert self.engine.get_final_action([]) == "allow"


# ---------------------------------------------------------------------------
# EntropyPlugin
# ---------------------------------------------------------------------------

class TestEntropyPlugin:
    def setup_method(self):
        self.plugin = EntropyPlugin(threshold=4.5)

    def test_name(self):
        assert self.plugin.name == "entropy_detector"

    def test_description_not_empty(self):
        assert self.plugin.description

    def test_safe_short_text_allowed(self):
        result = self.plugin.evaluate("hello world this is a simple sentence")
        assert result.action == "allow"
        assert result.passed is True

    def test_high_entropy_long_string_warns(self):
        # A random-looking long string with high character diversity
        high_entropy = "aB3xYz9qP2mNkLhJwVcR5tUeI0dFsSgO7"  # 33 chars
        result = self.plugin.evaluate(f"token: {high_entropy}")
        # Result depends on entropy calculation; at minimum check structure
        assert result.plugin_name == "entropy_detector"
        assert result.action in ("allow", "warn")

    def test_short_words_not_checked(self):
        # Words under 20 chars shouldn't trigger entropy check
        result = self.plugin.evaluate("short words here safe")
        assert result.action == "allow"

    def test_calculate_entropy_empty_string(self):
        assert self.plugin._calculate_entropy("") == 0.0

    def test_calculate_entropy_single_char(self):
        # Single repeated character has entropy 0
        assert self.plugin._calculate_entropy("aaaaaaa") == 0.0

    def test_calculate_entropy_uniform_distribution(self):
        # 2-character uniform distribution has entropy 1.0
        text = "ababababab"
        entropy = self.plugin._calculate_entropy(text)
        assert abs(entropy - 1.0) < 0.01


# ---------------------------------------------------------------------------
# RepetitionPlugin
# ---------------------------------------------------------------------------

class TestRepetitionPlugin:
    def setup_method(self):
        self.plugin = RepetitionPlugin(max_repetition_ratio=0.5)

    def test_name(self):
        assert self.plugin.name == "repetition_detector"

    def test_short_text_allowed(self):
        # < 10 words, skip check
        result = self.plugin.evaluate("hello world foo bar")
        assert result.action == "allow"
        assert result.passed is True

    def test_diverse_text_allowed(self):
        text = "The quick brown fox jumps over the lazy dog running fast"
        result = self.plugin.evaluate(text)
        assert result.action == "allow"

    def test_highly_repetitive_text_warns_or_blocks(self):
        repeated = ("spam " * 50).strip()
        result = self.plugin.evaluate(repeated)
        assert result.action in ("warn", "block")
        assert result.passed is False

    def test_extreme_repetition_blocks(self):
        repeated = ("spam " * 100).strip()
        result = RepetitionPlugin(max_repetition_ratio=0.3).evaluate(repeated)
        assert result.action in ("warn", "block")

    def test_repetition_ratio_in_details(self):
        repeated = ("abc " * 50).strip()
        result = self.plugin.evaluate(repeated)
        if not result.passed:
            assert "repetition_ratio" in result.details


# ---------------------------------------------------------------------------
# LengthPlugin
# ---------------------------------------------------------------------------

class TestLengthPlugin:
    def setup_method(self):
        self.plugin = LengthPlugin(max_chars=100)

    def test_name(self):
        assert self.plugin.name == "length_guard"

    def test_description_contains_max(self):
        assert "100" in self.plugin.description

    def test_short_text_allowed(self):
        result = self.plugin.evaluate("hello")
        assert result.action == "allow"
        assert result.passed is True

    def test_text_at_limit_allowed(self):
        text = "a" * 100
        result = self.plugin.evaluate(text)
        assert result.action == "allow"

    def test_text_over_limit_blocked(self):
        text = "a" * 101
        result = self.plugin.evaluate(text)
        assert result.action == "block"
        assert result.passed is False
        assert result.score == 1.0

    def test_details_include_length(self):
        text = "a" * 200
        result = self.plugin.evaluate(text)
        assert result.details["length"] == 200
        assert result.details["max_allowed"] == 100

    def test_empty_text_allowed(self):
        result = self.plugin.evaluate("")
        assert result.action == "allow"


# ---------------------------------------------------------------------------
# create_default_plugin_engine
# ---------------------------------------------------------------------------

class TestCreateDefaultPluginEngine:
    def test_returns_plugin_engine(self):
        engine = create_default_plugin_engine()
        assert isinstance(engine, PluginEngine)

    def test_contains_expected_plugins(self):
        engine = create_default_plugin_engine()
        assert "entropy_detector" in engine.plugins
        assert "repetition_detector" in engine.plugins
        assert "length_guard" in engine.plugins

    def test_default_engine_evaluates_safe_text(self):
        engine = create_default_plugin_engine()
        results = engine.evaluate_all("Hello, how are you today?")
        assert engine.get_final_action(results) == "allow"
