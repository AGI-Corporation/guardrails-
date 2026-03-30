"""
Tests for plugin_system.py
Covers: PluginEngine, EntropyPlugin, RepetitionPlugin, LengthPlugin,
        create_default_plugin_engine, custom plugin registration.
"""

import pytest

from plugin_system import (
    GuardrailPlugin,
    PluginEngine,
    PluginResult,
    EntropyPlugin,
    RepetitionPlugin,
    LengthPlugin,
    create_default_plugin_engine,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return create_default_plugin_engine()


# ── PluginEngine — registration ───────────────────────────────────────────

class TestPluginEngineRegistration:

    def test_register_plugin(self):
        pe = PluginEngine()
        pe.register(EntropyPlugin())
        assert "entropy_detector" in pe.plugins

    def test_unregister_plugin(self):
        pe = PluginEngine()
        pe.register(EntropyPlugin())
        pe.unregister("entropy_detector")
        assert "entropy_detector" not in pe.plugins

    def test_unregister_nonexistent_is_noop(self):
        pe = PluginEngine()
        pe.unregister("does_not_exist")  # should not raise


# ── PluginEngine — evaluate_all ───────────────────────────────────────────

class TestPluginEngineEvaluateAll:

    def test_returns_list(self, engine):
        results = engine.evaluate_all("Hello world")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_results_are_plugin_results(self, engine):
        results = engine.evaluate_all("Hello world")
        for r in results:
            assert isinstance(r, PluginResult)

    def test_timing_populated(self, engine):
        results = engine.evaluate_all("Hello world")
        for r in results:
            assert r.execution_time_ms >= 0

    def test_error_does_not_propagate(self):
        """A crashing plugin should not crash the engine — fail-open."""

        class BrokenPlugin(GuardrailPlugin):
            @property
            def name(self) -> str:
                return "broken"

            @property
            def description(self) -> str:
                return "Always crashes"

            def evaluate(self, text, context=None):
                raise RuntimeError("Plugin error")

        pe = PluginEngine()
        pe.register(BrokenPlugin())
        results = pe.evaluate_all("test text")
        assert len(results) == 1
        assert results[0].action == "allow"  # fail-open
        assert results[0].error is not None


# ── get_final_action ──────────────────────────────────────────────────────

class TestGetFinalAction:

    def _result(self, action: str) -> PluginResult:
        return PluginResult(plugin_name="p", passed=True, score=0.0, action=action)

    def test_all_allow_gives_allow(self):
        pe = PluginEngine()
        results = [self._result("allow"), self._result("allow")]
        assert pe.get_final_action(results) == "allow"

    def test_any_block_gives_block(self):
        pe = PluginEngine()
        results = [self._result("allow"), self._result("block")]
        assert pe.get_final_action(results) == "block"

    def test_warn_without_block_gives_warn(self):
        pe = PluginEngine()
        results = [self._result("allow"), self._result("warn")]
        assert pe.get_final_action(results) == "warn"

    def test_block_overrides_warn(self):
        pe = PluginEngine()
        results = [self._result("warn"), self._result("block")]
        assert pe.get_final_action(results) == "block"


# ── EntropyPlugin ─────────────────────────────────────────────────────────

class TestEntropyPlugin:

    def test_normal_text_passes(self):
        plugin = EntropyPlugin(threshold=4.5)
        result = plugin.evaluate("Hello, how are you today?")
        assert result.plugin_name == "entropy_detector"
        assert result.action == "allow"

    def test_high_entropy_long_token_triggers_warn(self):
        plugin = EntropyPlugin(threshold=2.0)
        # A long string (>= 20 chars) with many unique chars has high entropy
        high_entropy = "aB3!xY9&kQ2@mP7#nR5%zW"  # 22 chars, high entropy
        result = plugin.evaluate(high_entropy)
        assert result.action in ("warn", "block")
        assert result.score >= 0

    def test_result_has_score(self):
        plugin = EntropyPlugin()
        result = plugin.evaluate("test text here")
        assert 0.0 <= result.score <= 1.0


# ── RepetitionPlugin ──────────────────────────────────────────────────────

class TestRepetitionPlugin:

    def test_non_repetitive_text_passes(self):
        plugin = RepetitionPlugin()
        result = plugin.evaluate("The quick brown fox jumps over the lazy dog")
        assert result.action == "allow"

    def test_highly_repetitive_text_flagged(self):
        plugin = RepetitionPlugin(max_repetition_ratio=0.1)
        # Long text with extreme repetition (>= 10 words, mostly the same word)
        repetitive = " ".join(["word"] * 20)
        result = plugin.evaluate(repetitive)
        assert result.action in ("warn", "block")


# ── LengthPlugin ──────────────────────────────────────────────────────────

class TestLengthPlugin:

    def test_normal_length_passes(self):
        plugin = LengthPlugin()
        result = plugin.evaluate("Hello world")
        assert result.action == "allow"

    def test_too_long_blocked(self):
        plugin = LengthPlugin(max_chars=10)
        result = plugin.evaluate("This string is definitely longer than ten characters")
        assert result.action == "block"

    def test_empty_string_passes(self):
        plugin = LengthPlugin()
        result = plugin.evaluate("")
        assert result.action == "allow"


# ── create_default_plugin_engine ──────────────────────────────────────────

class TestCreateDefaultPluginEngine:

    def test_returns_plugin_engine(self, engine):
        assert isinstance(engine, PluginEngine)

    def test_has_plugins(self, engine):
        assert len(engine.plugins) > 0

    def test_evaluates_text(self, engine):
        results = engine.evaluate_all("Normal text that should be fine")
        assert len(results) == len(engine.plugins)
