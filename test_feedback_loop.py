"""
Tests for feedback_loop.py
Covers FeedbackEntry, FeedbackStore, TuningSuggester, and helper functions.
"""

import os
import tempfile

import pytest

from feedback_loop import (
    FeedbackEntry,
    FeedbackType,
    FeedbackStore,
    TuningSuggester,
    create_feedback_entry,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def make_entry(
    text="sample text",
    original_action="block",
    feedback_type=FeedbackType.CORRECT_BLOCK,
    matched_rules=None,
    user_id=None,
    comment="",
) -> FeedbackEntry:
    return FeedbackEntry(
        id=None,
        timestamp="2024-01-01T00:00:00",
        text=text,
        original_action=original_action,
        feedback_type=feedback_type,
        user_id=user_id,
        matched_rules=matched_rules or [],
        expected_action="block",
        comment=comment,
    )


# ── FeedbackType ─────────────────────────────────────────────────────────────

class TestFeedbackType:
    def test_values(self):
        assert FeedbackType.FALSE_POSITIVE.value == "false_positive"
        assert FeedbackType.FALSE_NEGATIVE.value == "false_negative"
        assert FeedbackType.CORRECT_BLOCK.value == "correct_block"
        assert FeedbackType.CORRECT_ALLOW.value == "correct_allow"


# ── FeedbackStore ─────────────────────────────────────────────────────────────

class TestFeedbackStoreInit:
    def test_creates_database(self):
        db_path = make_temp_db()
        try:
            FeedbackStore(db_path)
            assert os.path.exists(db_path)
        finally:
            os.unlink(db_path)

    def test_idempotent_init(self):
        db_path = make_temp_db()
        try:
            FeedbackStore(db_path)
            FeedbackStore(db_path)
        finally:
            os.unlink(db_path)


class TestFeedbackStoreAdd:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.store = FeedbackStore(self.db_path)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_add_returns_integer_id(self):
        entry = make_entry()
        row_id = self.store.add(entry)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_add_increments_id(self):
        id1 = self.store.add(make_entry(text="first"))
        id2 = self.store.add(make_entry(text="second"))
        assert id2 > id1

    def test_add_persists_entry(self):
        entry = make_entry(
            text="suspicious content",
            original_action="allow",
            feedback_type=FeedbackType.FALSE_NEGATIVE,
            matched_rules=["rule_x"],
            user_id="bob",
            comment="missed this one",
        )
        self.store.add(entry)
        all_entries = self.store.get_all()
        assert len(all_entries) == 1
        stored = all_entries[0]
        assert stored.text == "suspicious content"
        assert stored.original_action == "allow"
        assert stored.feedback_type == FeedbackType.FALSE_NEGATIVE
        assert stored.matched_rules == ["rule_x"]
        assert stored.user_id == "bob"
        assert stored.comment == "missed this one"

    def test_add_empty_matched_rules(self):
        self.store.add(make_entry(matched_rules=[]))
        entries = self.store.get_all()
        assert entries[0].matched_rules == []


class TestFeedbackStoreGetAll:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.store = FeedbackStore(self.db_path)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_empty_returns_empty_list(self):
        assert self.store.get_all() == []

    def test_returns_all_entries(self):
        for i in range(4):
            self.store.add(make_entry(text=f"entry {i}"))
        assert len(self.store.get_all()) == 4

    def test_filter_by_feedback_type(self):
        self.store.add(make_entry(feedback_type=FeedbackType.FALSE_POSITIVE,
                                  original_action="block"))
        self.store.add(make_entry(feedback_type=FeedbackType.FALSE_NEGATIVE,
                                  original_action="allow"))
        self.store.add(make_entry(feedback_type=FeedbackType.CORRECT_BLOCK))

        fp_entries = self.store.get_all(FeedbackType.FALSE_POSITIVE)
        assert len(fp_entries) == 1
        assert fp_entries[0].feedback_type == FeedbackType.FALSE_POSITIVE

    def test_no_match_type_filter_returns_empty(self):
        self.store.add(make_entry(feedback_type=FeedbackType.CORRECT_ALLOW,
                                  original_action="allow"))
        results = self.store.get_all(FeedbackType.FALSE_POSITIVE)
        assert results == []

    def test_returns_feedback_entry_objects(self):
        self.store.add(make_entry())
        entries = self.store.get_all()
        assert all(isinstance(e, FeedbackEntry) for e in entries)


class TestFeedbackStoreGetStats:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.store = FeedbackStore(self.db_path)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_empty_stats(self):
        stats = self.store.get_stats()
        assert stats["total"] == 0
        assert stats["false_positives"] == 0
        assert stats["false_negatives"] == 0
        assert stats["correct_blocks"] == 0
        assert stats["correct_allows"] == 0

    def test_counts_correctly(self):
        self.store.add(make_entry(feedback_type=FeedbackType.FALSE_POSITIVE,
                                  original_action="block"))
        self.store.add(make_entry(feedback_type=FeedbackType.FALSE_POSITIVE,
                                  original_action="block"))
        self.store.add(make_entry(feedback_type=FeedbackType.FALSE_NEGATIVE,
                                  original_action="allow"))
        self.store.add(make_entry(feedback_type=FeedbackType.CORRECT_BLOCK))
        self.store.add(make_entry(feedback_type=FeedbackType.CORRECT_ALLOW,
                                  original_action="allow"))
        stats = self.store.get_stats()
        assert stats["total"] == 5
        assert stats["false_positives"] == 2
        assert stats["false_negatives"] == 1
        assert stats["correct_blocks"] == 1
        assert stats["correct_allows"] == 1


# ── TuningSuggester ───────────────────────────────────────────────────────────

class TestTuningSuggester:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.store = FeedbackStore(self.db_path)
        self.suggester = TuningSuggester(self.store)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_keyword_suggestions_empty_when_no_false_negatives(self):
        suggestions = self.suggester.suggest_keyword_additions()
        assert suggestions == []

    def test_keyword_suggestions_based_on_false_negatives(self):
        # Add many false negatives with repeating words
        for _ in range(5):
            self.store.add(FeedbackEntry(
                id=None,
                timestamp="2024-01-01T00:00:00",
                text="malicious attack content with malicious attack words",
                original_action="allow",
                feedback_type=FeedbackType.FALSE_NEGATIVE,
                user_id=None,
                matched_rules=[],
                expected_action="block",
            ))
        suggestions = self.suggester.suggest_keyword_additions(min_occurrences=3)
        keywords = [s["keyword"] for s in suggestions]
        # "malicious", "attack", "content", "words" should appear
        assert any(k in keywords for k in ["malicious", "attack", "content"])

    def test_relaxation_suggestions_empty_when_no_false_positives(self):
        suggestions = self.suggester.suggest_rule_relaxation()
        assert suggestions == []

    def test_relaxation_suggestions_based_on_false_positives(self):
        for _ in range(6):
            self.store.add(FeedbackEntry(
                id=None,
                timestamp="2024-01-01T00:00:00",
                text="safe text",
                original_action="block",
                feedback_type=FeedbackType.FALSE_POSITIVE,
                user_id=None,
                matched_rules=["overly_strict_rule"],
                expected_action="allow",
            ))
        suggestions = self.suggester.suggest_rule_relaxation(min_occurrences=5)
        assert len(suggestions) >= 1
        assert suggestions[0]["rule_id"] == "overly_strict_rule"

    def test_generate_report_contains_sections(self):
        report = self.suggester.generate_report()
        assert "Guardrail Tuning Report" in report
        assert "Feedback Summary" in report
        assert "Total feedback entries:" in report

    def test_generate_report_includes_keyword_suggestions_section(self):
        """Trigger the 'Suggested Keyword Additions' branch in generate_report."""
        for _ in range(4):
            self.store.add(FeedbackEntry(
                id=None,
                timestamp="2024-01-01T00:00:00",
                text="malicious attack content harmful attack",
                original_action="allow",
                feedback_type=FeedbackType.FALSE_NEGATIVE,
                user_id=None,
                matched_rules=[],
                expected_action="block",
            ))
        report = self.suggester.generate_report()
        assert "Suggested Keyword Additions" in report

    def test_generate_report_includes_rule_relaxation_section(self):
        """Trigger the 'Suggested Rule Relaxations' branch in generate_report."""
        for _ in range(6):
            self.store.add(FeedbackEntry(
                id=None,
                timestamp="2024-01-01T00:00:00",
                text="safe content",
                original_action="block",
                feedback_type=FeedbackType.FALSE_POSITIVE,
                user_id=None,
                matched_rules=["overly_strict_rule"],
                expected_action="allow",
            ))
        report = self.suggester.generate_report()
        assert "Suggested Rule Relaxations" in report


# ── create_feedback_entry ─────────────────────────────────────────────────────

class TestCreateFeedbackEntry:
    def test_false_positive_sets_expected_allow(self):
        entry = create_feedback_entry(
            text="text",
            original_action="block",
            feedback_type=FeedbackType.FALSE_POSITIVE,
        )
        assert entry.expected_action == "allow"

    def test_false_negative_sets_expected_block(self):
        entry = create_feedback_entry(
            text="text",
            original_action="allow",
            feedback_type=FeedbackType.FALSE_NEGATIVE,
        )
        assert entry.expected_action == "block"

    def test_correct_block_sets_expected_block(self):
        entry = create_feedback_entry(
            text="text",
            original_action="block",
            feedback_type=FeedbackType.CORRECT_BLOCK,
        )
        assert entry.expected_action == "block"

    def test_correct_allow_sets_expected_allow(self):
        entry = create_feedback_entry(
            text="text",
            original_action="allow",
            feedback_type=FeedbackType.CORRECT_ALLOW,
        )
        assert entry.expected_action == "allow"

    def test_timestamp_set_automatically(self):
        entry = create_feedback_entry(
            text="text",
            original_action="allow",
            feedback_type=FeedbackType.CORRECT_ALLOW,
        )
        assert entry.timestamp != ""

    def test_optional_fields(self):
        entry = create_feedback_entry(
            text="text",
            original_action="allow",
            feedback_type=FeedbackType.CORRECT_ALLOW,
            matched_rules=["r1"],
            user_id="alice",
            comment="looks good",
        )
        assert entry.matched_rules == ["r1"]
        assert entry.user_id == "alice"
        assert entry.comment == "looks good"

    def test_id_is_none(self):
        entry = create_feedback_entry(
            text="text",
            original_action="allow",
            feedback_type=FeedbackType.CORRECT_ALLOW,
        )
        assert entry.id is None
