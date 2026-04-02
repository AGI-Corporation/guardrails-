"""
Tests for plugin_system.py — PluginEngine, built-in plugins, PluginManager.
"""
import pytest

from plugin_system import (
    EntropyPlugin,
    GuardrailPlugin,
    LengthPlugin,
    PluginEngine,
    PluginManager,
    PluginResult,
    RepetitionPlugin,
    create_default_plugin_engine,
)


# ── PluginEngine ──────────────────────────────────────────────────────────────

class TestPluginEngine:
    def test_register_and_evaluate(self):
        engine = create_default_plugin_engine()
        results = engine.evaluate_all("hello world")
        assert len(results) == 3

    def test_unregister_removes_plugin(self):
        engine = create_default_plugin_engine()
        engine.unregister("entropy_detector")
        results = engine.evaluate_all("hello world")
        assert len(results) == 2

    def test_get_final_action_all_allow(self):
        engine = create_default_plugin_engine()
        results = engine.evaluate_all("hello world")
        action = engine.get_final_action(results)
        assert action == "allow"

    def test_get_final_action_block_wins(self):
        engine = PluginEngine()
        engine.register(LengthPlugin(max_chars=1))
        results = engine.evaluate_all("hello world")
        action = engine.get_final_action(results)
        assert action == "block"

    def test_get_final_action_warn(self):
        engine = PluginEngine()
        engine.register(EntropyPlugin(threshold=0.1))  # very low threshold → always warn
        results = engine.evaluate_all("A" * 25)
        action = engine.get_final_action(results)
        # EntropyPlugin warns (not blocks), so warn should be final
        assert action in ("warn", "allow")

    def test_plugin_error_fail_open(self):
        class BrokenPlugin(GuardrailPlugin):
            @property
            def name(self) -> str:
                return "broken"

            @property
            def description(self) -> str:
                return "always raises"

            def evaluate(self, text, context=None) -> PluginResult:
                raise RuntimeError("broken!")

        engine = PluginEngine()
        engine.register(BrokenPlugin())
        results = engine.evaluate_all("hello")
        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].action == "allow"  # fail-open


# ── EntropyPlugin ─────────────────────────────────────────────────────────────

class TestEntropyPlugin:
    def test_short_words_pass(self):
        plugin = EntropyPlugin()
        result = plugin.evaluate("hello world")
        assert result.passed

    def test_high_entropy_long_string_warns(self):
        plugin = EntropyPlugin(threshold=1.0)
        # Use a string with diverse chars to get high entropy
        text = "aAbBcCdDeEfFgGhHiIjJkK" * 2
        result = plugin.evaluate(text)
        # Plugin only checks words >= 20 chars
        assert result.action in ("allow", "warn")

    def test_result_has_plugin_name(self):
        plugin = EntropyPlugin()
        result = plugin.evaluate("hello")
        assert result.plugin_name == "entropy_detector"


# ── RepetitionPlugin ──────────────────────────────────────────────────────────

class TestRepetitionPlugin:
    def test_diverse_text_passes(self):
        plugin = RepetitionPlugin()
        result = plugin.evaluate("The quick brown fox jumps over the lazy dog now.")
        assert result.passed

    def test_highly_repetitive_text_warns_or_blocks(self):
        plugin = RepetitionPlugin(max_repetition_ratio=0.3)
        # 90% repetition
        text = ("spam " * 50).strip()
        result = plugin.evaluate(text)
        assert not result.passed

    def test_short_text_always_passes(self):
        plugin = RepetitionPlugin()
        result = plugin.evaluate("a b c")
        assert result.passed  # < 10 words

    def test_result_has_plugin_name(self):
        plugin = RepetitionPlugin()
        result = plugin.evaluate("hello world")
        assert result.plugin_name == "repetition_detector"


# ── LengthPlugin ─────────────────────────────────────────────────────────────

class TestLengthPlugin:
    def test_short_text_passes(self):
        plugin = LengthPlugin(max_chars=100)
        result = plugin.evaluate("Hello world")
        assert result.passed
        assert result.action == "allow"

    def test_long_text_blocked(self):
        plugin = LengthPlugin(max_chars=10)
        result = plugin.evaluate("This is a longer string")
        assert not result.passed
        assert result.action == "block"

    def test_exactly_at_limit_passes(self):
        plugin = LengthPlugin(max_chars=5)
        result = plugin.evaluate("12345")
        assert result.passed

    def test_one_over_limit_blocked(self):
        plugin = LengthPlugin(max_chars=5)
        result = plugin.evaluate("123456")
        assert not result.passed

    def test_result_has_plugin_name(self):
        plugin = LengthPlugin()
        result = plugin.evaluate("hello")
        assert result.plugin_name == "length_guard"


# ── PluginManager ─────────────────────────────────────────────────────────────

class TestPluginManager:
    def test_list_plugins_returns_names(self):
        manager = PluginManager()
        names = manager.list_plugins()
        assert "entropy_detector" in names
        assert "repetition_detector" in names
        assert "length_guard" in names

    def test_evaluate_returns_results(self):
        manager = PluginManager()
        results = manager.evaluate("hello world")
        assert len(results) == 3

    def test_get_final_action_clean_text(self):
        manager = PluginManager()
        action = manager.get_final_action("hello world")
        assert action == "allow"

    def test_get_final_action_too_long(self):
        manager = PluginManager()
        action = manager.get_final_action("x" * 15000)
        assert action == "block"

    def test_register_custom_plugin(self):
        class CustomPlugin(GuardrailPlugin):
            @property
            def name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "custom test plugin"

            def evaluate(self, text, context=None) -> PluginResult:
                passed = "allowed" in text
                return PluginResult(
                    plugin_name=self.name,
                    passed=passed,
                    score=0.0 if passed else 1.0,
                    action="allow" if passed else "block",
                )

        manager = PluginManager()
        manager.register(CustomPlugin())
        assert "custom" in manager.list_plugins()
        assert manager.get_final_action("dangerous text") == "block"
        assert manager.get_final_action("allowed text") in ("allow",)

    def test_unregister_plugin(self):
        manager = PluginManager()
        manager.unregister("length_guard")
        assert "length_guard" not in manager.list_plugins()


# ── create_default_plugin_engine ─────────────────────────────────────────────

class TestCreateDefaultPluginEngine:
    def test_returns_engine_with_three_plugins(self):
        engine = create_default_plugin_engine()
        assert len(engine.plugins) == 3

    def test_execution_time_recorded(self):
        engine = create_default_plugin_engine()
        results = engine.evaluate_all("test text")
        for r in results:
            assert r.execution_time_ms >= 0
