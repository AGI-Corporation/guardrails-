"""
Shared pytest fixtures for the Guardrails test suite.
"""
import pytest

from guardrail_framework import GuardrailEngine, create_default_guardrails


@pytest.fixture(scope="session")
def default_engine() -> GuardrailEngine:
    """A fully-configured GuardrailEngine with all default rules, session-scoped."""
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    return engine


@pytest.fixture()
def fresh_engine() -> GuardrailEngine:
    """A fresh GuardrailEngine with all default rules, function-scoped."""
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    return engine
