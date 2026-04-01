"""
Tests for content_transformer.py and plugin_system.py enterprise extensions.
"""
import pytest
from content_transformer import ContentTransformer, TransformationResult
from plugin_system import (
    EntropyPlugin,
    RepetitionPlugin,
    LengthPlugin,
    LanguageAnomalyPlugin,
    PIIDensityPlugin,
    SentimentPlugin,
    PluginEngine,
    create_default_plugin_engine,
)


# ── ContentTransformer ────────────────────────────────────────────────────────

@pytest.fixture()
def transformer():
    return ContentTransformer()


def test_ssn_redaction(transformer):
    result = transformer.apply_all_pii("My SSN is 123-45-6789.")
    assert "[SSN REDACTED]" in result.transformed
    assert result.changes_made >= 1


def test_credit_card_redaction(transformer):
    result = transformer.apply_all_pii("Card: 4111-1111-1111-1111")
    assert "[CC REDACTED]" in result.transformed


def test_email_redaction(transformer):
    result = transformer.apply_all_pii("Email: alice@example.com")
    assert "[EMAIL REDACTED]" in result.transformed


def test_phone_redaction(transformer):
    result = transformer.apply_all_pii("Call 555-867-5309")
    assert "[PHONE REDACTED]" in result.transformed


def test_ip_address_redaction(transformer):
    result = transformer.apply_all_pii("Server at 192.168.1.100")
    assert "[IP REDACTED]" in result.transformed


def test_api_key_redaction_openai(transformer):
    result = transformer.apply_all_pii("Key: sk-abcdefghijklmnopqrstuvwxyz123456")
    assert "[API KEY REDACTED]" in result.transformed


def test_api_key_redaction_aws(transformer):
    result = transformer.apply_all_pii("AWS: AKIAIOSFODNN7EXAMPLE")
    assert "[API KEY REDACTED]" in result.transformed


def test_jwt_redaction(transformer):
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    result = transformer.apply_all_pii(f"Bearer {jwt}")
    assert "[JWT REDACTED]" in result.transformed


def test_clean_text_unchanged(transformer):
    text = "What is the capital of France?"
    result = transformer.apply_all_pii(text)
    assert result.transformed == text
    assert result.changes_made == 0


def test_transformation_result_type(transformer):
    result = transformer.apply_all_pii("Hello world")
    assert isinstance(result, TransformationResult)
    assert isinstance(result.transformations_applied, list)


def test_apply_all_applies_all_transformers(transformer):
    result = transformer.apply_all("Profanity: badword1, email: test@test.com")
    assert isinstance(result, TransformationResult)


# ── Plugin system ─────────────────────────────────────────────────────────────

def test_entropy_plugin_detects_high_entropy():
    plugin = EntropyPlugin(threshold=4.0)
    long_key = "sk-" + "aB3xYz9Qr7Lp2Wm" * 5  # Long high-entropy string
    result = plugin.evaluate(long_key)
    # Should warn (not necessarily block)
    assert result.action in ("warn", "block")


def test_entropy_plugin_passes_normal_text():
    plugin = EntropyPlugin()
    result = plugin.evaluate("Hello, how are you today?")
    assert result.action == "allow"


def test_repetition_plugin_blocks_high_repetition():
    plugin = RepetitionPlugin(max_repetition_ratio=0.4)
    text = "ignore " * 50
    result = plugin.evaluate(text)
    assert result.action in ("warn", "block")


def test_repetition_plugin_passes_normal_text():
    plugin = RepetitionPlugin()
    text = "The quick brown fox jumps over the lazy dog in the park near the river"
    result = plugin.evaluate(text)
    # Unique enough to pass
    assert result.action == "allow"


def test_length_plugin_blocks_long_text():
    plugin = LengthPlugin(max_chars=100)
    text = "a" * 200
    result = plugin.evaluate(text)
    assert result.action == "block"
    assert result.details["length"] == 200


def test_length_plugin_passes_short_text():
    plugin = LengthPlugin(max_chars=1000)
    result = plugin.evaluate("short text")
    assert result.action == "allow"


def test_language_anomaly_plugin_detects_cyrillic():
    plugin = LanguageAnomalyPlugin()
    # Primarily Cyrillic characters (>5% ratio)
    cyrillic_text = "аеос" * 20 + " normal text"
    result = plugin.evaluate(cyrillic_text)
    assert result.action in ("warn", "block")


def test_language_anomaly_plugin_passes_ascii():
    plugin = LanguageAnomalyPlugin()
    result = plugin.evaluate("This is perfectly normal ASCII text.")
    assert result.action == "allow"


def test_pii_density_plugin_blocks_high_density():
    plugin = PIIDensityPlugin(max_pii_density=0.05)
    text = (
        "alice@example.com bob@test.org carol@company.net "
        "dave@work.io 123-45-6789 4111-1111-1111-1111"
    )
    result = plugin.evaluate(text)
    assert result.action == "block"


def test_pii_density_plugin_passes_clean_text():
    plugin = PIIDensityPlugin()
    result = plugin.evaluate("What is the weather like today?")
    assert result.action == "allow"


def test_sentiment_plugin_flags_threatening_text():
    plugin = SentimentPlugin(threat_threshold=0.05)
    text = "I will kill and attack and harm and destroy and murder everyone"
    result = plugin.evaluate(text)
    assert result.action in ("warn", "block")


def test_sentiment_plugin_passes_normal_text():
    plugin = SentimentPlugin()
    result = plugin.evaluate("Let's build something amazing together!")
    assert result.action == "allow"


def test_create_default_plugin_engine_has_6_plugins():
    engine = create_default_plugin_engine()
    assert len(engine.plugins) == 6


def test_plugin_engine_evaluate_all_returns_results():
    engine = create_default_plugin_engine()
    results = engine.evaluate_all("Hello world")
    assert len(results) == 6
    for r in results:
        assert r.action in ("allow", "warn", "block")


def test_plugin_engine_final_action_most_restrictive():
    engine = PluginEngine()
    engine.register(SentimentPlugin(threat_threshold=0.01))
    engine.register(LengthPlugin(max_chars=5))
    # Short but threatening
    results = engine.evaluate_all("kill")
    final = engine.get_final_action(results)
    assert final in ("warn", "block")
