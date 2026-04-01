"""
Tests for feedback_loop.py
"""
import pytest
from feedback_loop import (
    FeedbackLoop,
    FeedbackStore,
    FeedbackType,
    TuningSuggester,
    create_feedback_entry,
)
from guardrail_framework import GuardrailEngine, create_default_guardrails


@pytest.fixture
def engine() -> GuardrailEngine:
    e = GuardrailEngine()
    for r in create_default_guardrails():
        e.add_rule(r)
    return e


@pytest.fixture
def loop(engine) -> FeedbackLoop:
    return FeedbackLoop(engine, db_path=":memory:")


class TestFeedbackLoop:
    def test_record_block(self, loop):
        row_id = loop.record("test text", "block", ["ssn"])
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_record_allow(self, loop):
        row_id = loop.record("clean text", "allow")
        assert row_id >= 1

    def test_get_stats_empty(self, loop):
        stats = loop.get_stats()
        assert "total" in stats
        assert stats["total"] == 0

    def test_get_stats_after_records(self, loop):
        loop.record("blocked text", "block", ["ssn"])
        loop.record("allowed text", "allow")
        stats = loop.get_stats()
        assert stats["total"] == 2

    def test_mark_false_positive(self, loop):
        loop.mark_false_positive("text", matched_rules=["ssn"])
        stats = loop.get_stats()
        assert stats["false_positives"] >= 1

    def test_mark_false_negative(self, loop):
        loop.mark_false_negative("text")
        stats = loop.get_stats()
        assert stats["false_negatives"] >= 1

    def test_generate_report_returns_string(self, loop):
        loop.record("text", "block", ["ssn"])
        report = loop.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_init_without_engine(self):
        loop = FeedbackLoop(db_path=":memory:")
        assert loop.engine is None
        row_id = loop.record("text", "allow")
        assert row_id >= 1


class TestFeedbackStore:
    def test_add_and_get_all(self):
        store = FeedbackStore(db_path=":memory:")
        entry = create_feedback_entry("text", "block", FeedbackType.CORRECT_BLOCK)
        store.add(entry)
        entries = store.get_all()
        assert len(entries) == 1

    def test_get_stats_false_positives(self):
        store = FeedbackStore(db_path=":memory:")
        entry = create_feedback_entry("text", "block", FeedbackType.FALSE_POSITIVE)
        store.add(entry)
        stats = store.get_stats()
        assert stats["false_positives"] == 1
        assert stats["total"] == 1
