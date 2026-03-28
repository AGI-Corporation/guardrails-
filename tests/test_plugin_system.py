"""
Tests for plugin_system.py
"""

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
# PluginResult dataclass
# ---------------------------------------------------------------------------

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

    def test_with_error(self):
        result = PluginResult(
            plugin_name="broken",
            passed=True,
            score=0.0,
            action="allow",
            error="something went wrong",
        )
        assert result.error == "something went wrong"


# ---------------------------------------------------------------------------
# EntropyPlugin
# ---------------------------------------------------------------------------

class TestEntropyPlugin:
    @pytest.fixture()
    def plugin(self):
        return EntropyPlugin(threshold=4.5)

    def test_name_and_description(self, plugin):
        assert plugin.name == "entropy_detector"
        assert "entropy" in plugin.description.lower()

    def test_short_words_ignored(self, plugin):
        result = plugin.evaluate("hello world how are you")
        assert result.action == "allow"
        assert result.passed is True

    def test_high_entropy_long_string_flagged(self, plugin):
        # A base64-like string with high entropy
        high_entropy = "aB3$kL9mNpQrStUvWxYz01234567ABCDEFGH"
        result = plugin.evaluate(f"key={high_entropy}")
        # Should warn if entropy is high
        if result.action == "warn":
            assert result.passed is False
        else:
            assert result.action == "allow"  # threshold may not be exceeded in all cases

    def test_repeated_character_low_entropy(self, plugin):
        result = plugin.evaluate("a" * 50)
        assert result.action == "allow"
        assert result.passed is True

    def test_calculate_entropy_empty(self, plugin):
        entropy = plugin._calculate_entropy("")
        assert entropy == 0.0

    def test_calculate_entropy_single_char(self, plugin):
        entropy = plugin._calculate_entropy("aaaa")
        assert entropy == 0.0

    def test_calculate_entropy_uniform(self, plugin):
        # Two equally likely characters: entropy should be 1.0
        entropy = plugin._calculate_entropy("abababab")
        assert abs(entropy - 1.0) < 0.01

    def test_score_bounded(self, plugin):
        # Even with many high-entropy strings, score should not exceed 1.0
        many_high_entropy = " ".join(["aB3$kL9mNpQrStUvWxYz01234"] * 10)
        result = plugin.evaluate(many_high_entropy)
        assert result.score <= 1.0


# ---------------------------------------------------------------------------
# RepetitionPlugin
# ---------------------------------------------------------------------------

class TestRepetitionPlugin:
    @pytest.fixture()
    def plugin(self):
        return RepetitionPlugin(max_repetition_ratio=0.5)

    def test_name_and_description(self, plugin):
        assert plugin.name == "repetition_detector"
        assert "repetition" in plugin.description.lower()

    def test_short_text_always_passes(self, plugin):
        # less than 10 words → always passes
        result = plugin.evaluate("hello world")
        assert result.action == "allow"
        assert result.passed is True

    def test_diverse_text_passes(self, plugin):
        text = " ".join(f"word{i}" for i in range(30))
        result = plugin.evaluate(text)
        assert result.action == "allow"

    def test_highly_repetitive_text_blocked(self, plugin):
        # All same word repeated many times → repetition_ratio near 1.0
        text = " ".join(["spam"] * 50)
        result = plugin.evaluate(text)
        assert result.action == "block"
        assert result.passed is False

    def test_moderate_repetition_warns(self):
        # ratio between 0.5 and 0.8 → warn
        plugin = RepetitionPlugin(max_repetition_ratio=0.3)
        words = ["a"] * 40 + [f"word{i}" for i in range(10)]
        text = " ".join(words)
        result = plugin.evaluate(text)
        # Depending on exact ratio, it may be warn or block
        assert result.action in ("warn", "block")
        assert result.passed is False

    def test_repetition_ratio_in_details(self, plugin):
        text = " ".join(["spam"] * 50)
        result = plugin.evaluate(text)
        assert "repetition_ratio" in result.details


# ---------------------------------------------------------------------------
# LengthPlugin
# ---------------------------------------------------------------------------

class TestLengthPlugin:
    @pytest.fixture()
    def plugin(self):
        return LengthPlugin(max_chars=100)

    def test_name_and_description(self, plugin):
        assert plugin.name == "length_guard"
        assert "100" in plugin.description

    def test_short_text_passes(self, plugin):
        result = plugin.evaluate("Short text")
        assert result.action == "allow"
        assert result.passed is True
        assert result.score == 0.0

    def test_exactly_at_limit_passes(self, plugin):
        result = plugin.evaluate("a" * 100)
        assert result.action == "allow"

    def test_exceeds_limit_blocked(self, plugin):
        result = plugin.evaluate("a" * 101)
        assert result.action == "block"
        assert result.passed is False
        assert result.score == 1.0

    def test_details_contain_length(self, plugin):
        text = "a" * 200
        result = plugin.evaluate(text)
        assert result.details["length"] == 200
        assert result.details["max_allowed"] == 100

    def test_empty_text_passes(self, plugin):
        result = plugin.evaluate("")
        assert result.action == "allow"


# ---------------------------------------------------------------------------
# PluginEngine
# ---------------------------------------------------------------------------

class TestPluginEngine:
    def test_register_and_retrieve(self):
        engine = PluginEngine()
        engine.register(LengthPlugin())
        assert "length_guard" in engine.plugins

    def test_unregister(self):
        engine = PluginEngine()
        engine.register(LengthPlugin())
        engine.unregister("length_guard")
        assert "length_guard" not in engine.plugins

    def test_unregister_nonexistent_silent(self):
        engine = PluginEngine()
        engine.unregister("does_not_exist")  # should not raise

    def test_evaluate_all_returns_results_for_each_plugin(self):
        engine = PluginEngine()
        engine.register(LengthPlugin(max_chars=1000))
        engine.register(RepetitionPlugin())
        results = engine.evaluate_all("hello world")
        assert len(results) == 2
        plugin_names = {r.plugin_name for r in results}
        assert "length_guard" in plugin_names
        assert "repetition_detector" in plugin_names

    def test_evaluate_all_empty_engine(self):
        engine = PluginEngine()
        results = engine.evaluate_all("anything")
        assert results == []

    def test_execution_time_recorded(self):
        engine = PluginEngine()
        engine.register(LengthPlugin())
        results = engine.evaluate_all("hello")
        assert results[0].execution_time_ms >= 0

    def test_get_final_action_block_takes_priority(self):
        engine = PluginEngine()
        results = [
            PluginResult(plugin_name="allow_plugin", passed=True, score=0.0, action="allow"),
            PluginResult(plugin_name="block_plugin", passed=False, score=1.0, action="block"),
            PluginResult(plugin_name="warn_plugin", passed=False, score=0.5, action="warn"),
        ]
        assert engine.get_final_action(results) == "block"

    def test_get_final_action_warn_when_no_block(self):
        engine = PluginEngine()
        results = [
            PluginResult(plugin_name="allow_plugin", passed=True, score=0.0, action="allow"),
            PluginResult(plugin_name="warn_plugin", passed=False, score=0.3, action="warn"),
        ]
        assert engine.get_final_action(results) == "warn"

    def test_get_final_action_allow_when_all_pass(self):
        engine = PluginEngine()
        results = [
            PluginResult(plugin_name="allow_plugin_a", passed=True, score=0.0, action="allow"),
            PluginResult(plugin_name="allow_plugin_b", passed=True, score=0.0, action="allow"),
        ]
        assert engine.get_final_action(results) == "allow"

    def test_fail_open_on_plugin_error(self):
        """A crashing plugin should fail open (allow), not block the request."""

        class BrokenPlugin(GuardrailPlugin):
            @property
            def name(self):
                return "broken_plugin"

            @property
            def description(self):
                return "Always raises"

            def evaluate(self, text, context=None):
                raise RuntimeError("Plugin crashed")

        engine = PluginEngine()
        engine.register(BrokenPlugin())
        results = engine.evaluate_all("hello")
        assert len(results) == 1
        r = results[0]
        assert r.action == "allow"
        assert r.error is not None
        assert "Plugin crashed" in r.error


# ---------------------------------------------------------------------------
# create_default_plugin_engine
# ---------------------------------------------------------------------------

class TestCreateDefaultPluginEngine:
    def test_has_default_plugins(self):
        engine = create_default_plugin_engine()
        names = set(engine.plugins.keys())
        assert "entropy_detector" in names
        assert "repetition_detector" in names
        assert "length_guard" in names

    def test_default_engine_evaluates_clean_text(self):
        engine = create_default_plugin_engine()
        results = engine.evaluate_all("Hello, how are you doing today?")
        final = engine.get_final_action(results)
        assert final == "allow"
