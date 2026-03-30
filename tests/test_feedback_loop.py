"""
Tests for feedback_loop.py
Covers: FeedbackStore (add, get_all, get_stats), TuningSuggester,
        create_feedback_entry helper.
"""

import pytest

from feedback_loop import (
    FeedbackStore,
    FeedbackEntry,
    FeedbackType,
    TuningSuggester,
    create_feedback_entry,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=str(tmp_path / "test_feedback.db"))


@pytest.fixture
def populated_store(tmp_path):
    s = FeedbackStore(db_path=str(tmp_path / "test_feedback_pop.db"))
    # 5 false-positive entries (blocked but should be allowed)
    for i in range(5):
        s.add(create_feedback_entry(
            text=f"safe text {i}",
            original_action="block",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            matched_rules=["hate_speech_basic"],
        ))
    # 3 correct-block entries
    for i in range(3):
        s.add(create_feedback_entry(
            text="bad text",
            original_action="block",
            feedback_type=FeedbackType.CORRECT_BLOCK,
            matched_rules=["pii_ssn"],
        ))
    return s


# ── create_feedback_entry ─────────────────────────────────────────────────

class TestCreateFeedbackEntry:

    def test_creates_entry(self):
        entry = create_feedback_entry(
            text="test",
            original_action="block",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            matched_rules=["r1"],
        )
        assert entry.text == "test"
        assert entry.original_action == "block"
        assert entry.feedback_type == FeedbackType.FALSE_POSITIVE
        assert "r1" in entry.matched_rules

    def test_timestamp_populated(self):
        entry = create_feedback_entry("t", "allow", FeedbackType.CORRECT_ALLOW, [])
        assert entry.timestamp

    def test_id_is_none(self):
        entry = create_feedback_entry("t", "allow", FeedbackType.CORRECT_ALLOW, [])
        assert entry.id is None


# ── FeedbackStore — add & get_all ─────────────────────────────────────────

class TestFeedbackStoreAdd:

    def test_add_returns_int_id(self, store):
        entry = create_feedback_entry("t", "block", FeedbackType.FALSE_POSITIVE, [])
        entry_id = store.add(entry)
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_get_all_returns_added_entries(self, store):
        store.add(create_feedback_entry("a", "block", FeedbackType.FALSE_POSITIVE, []))
        store.add(create_feedback_entry("b", "allow", FeedbackType.CORRECT_ALLOW, []))
        all_entries = store.get_all()
        assert len(all_entries) == 2

    def test_filter_by_feedback_type(self, populated_store):
        fps = populated_store.get_all(feedback_type=FeedbackType.FALSE_POSITIVE)
        assert len(fps) == 5
        for e in fps:
            assert e.feedback_type == FeedbackType.FALSE_POSITIVE

        cbs = populated_store.get_all(feedback_type=FeedbackType.CORRECT_BLOCK)
        assert len(cbs) == 3

    def test_empty_store_returns_empty_list(self, store):
        assert store.get_all() == []


# ── FeedbackStore — get_stats ─────────────────────────────────────────────

class TestFeedbackStoreStats:

    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats.get("total", 0) == 0

    def test_stats_total(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["total"] == 8

    def test_stats_by_type(self, populated_store):
        stats = populated_store.get_stats()
        assert stats.get("false_positives", 0) == 5
        assert stats.get("correct_blocks", 0) == 3


# ── TuningSuggester ───────────────────────────────────────────────────────

class TestTuningSuggester:

    def test_returns_suggestions_or_empty_list(self, populated_store):
        suggester = TuningSuggester(store=populated_store)
        suggestions = suggester.suggest_keyword_additions(min_occurrences=1)
        assert isinstance(suggestions, list)

    def test_suggest_rule_relaxation_returns_list(self, populated_store):
        suggester = TuningSuggester(store=populated_store)
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=1)
        assert isinstance(suggestions, list)

    def test_generate_report_returns_string(self, populated_store):
        suggester = TuningSuggester(store=populated_store)
        report = suggester.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0
