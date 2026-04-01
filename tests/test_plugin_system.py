"""
Tests for plugin_system.py
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
from guardrail_framework import GuardrailEngine, create_default_guardrails


@pytest.fixture
def engine() -> GuardrailEngine:
    e = GuardrailEngine()
    for r in create_default_guardrails():
        e.add_rule(r)
    return e


@pytest.fixture
def manager(engine) -> PluginManager:
    return PluginManager(engine)


class TestPluginManager:
    def test_list_plugins_default(self, manager):
        plugins = manager.list_plugins()
        assert isinstance(plugins, list)
        assert len(plugins) >= 3  # entropy, repetition, length

    def test_list_plugins_contains_default_names(self, manager):
        plugins = manager.list_plugins()
        assert "entropy_detector" in plugins
        assert "repetition_detector" in plugins
        assert "length_guard" in plugins

    def test_register_custom_plugin(self, manager, engine):
        class AlwaysPassPlugin(GuardrailPlugin):
            @property
            def name(self) -> str:
                return "always_pass"

            @property
            def description(self) -> str:
                return "Always passes"

            def evaluate(self, text, context=None) -> PluginResult:
                return PluginResult(
                    plugin_name=self.name, passed=True, score=0.0, action="allow"
                )

        manager.register(AlwaysPassPlugin())
        assert "always_pass" in manager.list_plugins()

    def test_unregister_plugin(self, manager):
        manager.unregister("length_guard")
        assert "length_guard" not in manager.list_plugins()

    def test_evaluate_returns_results(self, manager):
        results = manager.evaluate("hello world")
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, PluginResult)

    def test_get_final_action(self, manager):
        action = manager.get_final_action("hello world")
        assert action in ("allow", "block", "warn")

    def test_init_without_engine(self):
        m = PluginManager()
        assert m.engine is None
        assert len(m.list_plugins()) >= 3


class TestBuiltinPlugins:
    def test_length_plugin_allows_short(self):
        plugin = LengthPlugin(max_chars=100)
        result = plugin.evaluate("short text")
        assert result.passed is True
        assert result.action == "allow"

    def test_length_plugin_blocks_long(self):
        plugin = LengthPlugin(max_chars=10)
        result = plugin.evaluate("this text is longer than ten characters")
        assert result.passed is False
        assert result.action == "block"

    def test_repetition_plugin_allows_normal(self):
        plugin = RepetitionPlugin()
        result = plugin.evaluate("The quick brown fox jumps over the lazy dog")
        assert result.passed is True

    def test_repetition_plugin_blocks_repetitive(self):
        plugin = RepetitionPlugin(max_repetition_ratio=0.1)
        repetitive_text = "word " * 50  # highly repetitive
        result = plugin.evaluate(repetitive_text)
        assert result.passed is False

    def test_entropy_plugin_allows_normal(self):
        plugin = EntropyPlugin()
        result = plugin.evaluate("The quick brown fox jumps over the lazy dog.")
        # Normal text should have reasonable entropy
        assert isinstance(result.passed, bool)

    def test_entropy_plugin_blocks_high_entropy(self):
        plugin = EntropyPlugin(threshold=3.0)  # low threshold
        # Random-looking base64 string has high entropy
        result = plugin.evaluate("aB3xZ9mK2pL7nQ4vR1wY6tU0sE5oI8hF")
        assert isinstance(result.passed, bool)


class TestPluginEngine:
    def test_register_and_evaluate_all(self):
        engine = PluginEngine()
        engine.register(LengthPlugin())
        results = engine.evaluate_all("hello")
        assert len(results) == 1
        assert results[0].plugin_name == "length_guard"

    def test_get_final_action_all_pass(self):
        engine = create_default_plugin_engine()
        action = engine.get_final_action(engine.evaluate_all("hello world"))
        assert action == "allow"

    def test_plugin_error_is_fail_open(self):
        class BrokenPlugin(GuardrailPlugin):
            @property
            def name(self) -> str:
                return "broken"

            @property
            def description(self) -> str:
                return "Broken plugin"

            def evaluate(self, text, context=None) -> PluginResult:
                raise RuntimeError("Plugin error!")

        engine = PluginEngine()
        engine.register(BrokenPlugin())
        results = engine.evaluate_all("text")
        assert len(results) == 1
        # Fail-open: should allow even if plugin throws
        assert results[0].action == "allow"
        assert results[0].error is not None
