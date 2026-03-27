"""
Shared pytest fixtures for the guardrail test suite.

These fixtures are available to every test module automatically.
"""

import pytest

from guardrail_framework import (
    Action,
    GuardrailCategory,
    GuardrailEngine,
    GuardrailRule,
    Severity,
    create_default_guardrails,
    create_default_test_cases,
)


@pytest.fixture
def empty_engine():
    """A GuardrailEngine with no rules."""
    return GuardrailEngine()


@pytest.fixture
def default_engine():
    """A GuardrailEngine pre-loaded with the default guardrail rules."""
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    return engine


@pytest.fixture
def full_engine():
    """A GuardrailEngine pre-loaded with default rules AND test cases."""
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    for tc in create_default_test_cases():
        engine.add_test_case(tc)
    return engine


@pytest.fixture
def make_block_engine():
    """
    Factory fixture – call it with a keyword string to get an engine that
    blocks any text containing that keyword.

    Usage::

        def test_something(make_block_engine):
            engine = make_block_engine("forbidden")
            result = engine.evaluate("this is forbidden")
            assert result.action == "block"
    """
    def _factory(keyword: str) -> GuardrailEngine:
        engine = GuardrailEngine()
        rule = GuardrailRule(
            id="block_rule",
            name="Block Rule",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.HIGH,
            action=Action.BLOCK,
            keywords=[keyword],
        )
        engine.add_rule(rule)
        return engine

    return _factory
