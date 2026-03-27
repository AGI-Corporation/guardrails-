"""
Tests for feedback_loop.py
Covers: FeedbackStore init/add/get_all/get_stats,
        TuningSuggester keyword/relaxation suggestions and report,
        create_feedback_entry helper.
"""

import pytest

from feedback_loop import (
    FeedbackEntry,
    FeedbackStore,
    FeedbackType,
    TuningSuggester,
    create_feedback_entry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=str(tmp_path / "feedback.db"))


def _make_entry(
    text="test text",
    original_action="allow",
    feedback_type=FeedbackType.FALSE_NEGATIVE,
    user_id=None,
    matched_rules=None,
    expected_action="block",
    comment="",
):
    return FeedbackEntry(
        id=None,
        timestamp="2024-01-01T00:00:00",
        text=text,
        original_action=original_action,
        feedback_type=feedback_type,
        user_id=user_id,
        matched_rules=matched_rules or [],
        expected_action=expected_action,
        comment=comment,
    )


# ---------------------------------------------------------------------------
# FeedbackStore — init
# ---------------------------------------------------------------------------

class TestFeedbackStoreInit:
    def test_creates_db_file(self, tmp_path):
        db_path = str(tmp_path / "fb.db")
        FeedbackStore(db_path=db_path)
        import os
        assert os.path.exists(db_path)

    def test_reinit_is_idempotent(self, tmp_path):
        db_path = str(tmp_path / "fb.db")
        FeedbackStore(db_path=db_path)
        FeedbackStore(db_path=db_path)  # should not raise


# ---------------------------------------------------------------------------
# FeedbackStore — add()
# ---------------------------------------------------------------------------

class TestFeedbackStoreAdd:
    def test_add_returns_integer_id(self, store):
        entry = _make_entry()
        row_id = store.add(entry)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_add_increments_id(self, store):
        id1 = store.add(_make_entry())
        id2 = store.add(_make_entry())
        assert id2 > id1

    def test_add_preserves_fields(self, store):
        entry = _make_entry(
            text="my text",
            original_action="block",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            user_id="alice",
            matched_rules=["rule_x"],
            expected_action="allow",
            comment="not harmful",
        )
        store.add(entry)
        results = store.get_all()
        r = results[0]
        assert r.text == "my text"
        assert r.original_action == "block"
        assert r.feedback_type == FeedbackType.FALSE_POSITIVE
        assert r.user_id == "alice"
        assert r.matched_rules == ["rule_x"]
        assert r.expected_action == "allow"
        assert r.comment == "not harmful"


# ---------------------------------------------------------------------------
# FeedbackStore — get_all()
# ---------------------------------------------------------------------------

class TestFeedbackStoreGetAll:
    def test_get_all_empty(self, store):
        assert store.get_all() == []

    def test_get_all_returns_all(self, store):
        for _ in range(4):
            store.add(_make_entry())
        results = store.get_all()
        assert len(results) == 4

    def test_get_all_filter_by_type(self, store):
        store.add(_make_entry(feedback_type=FeedbackType.FALSE_POSITIVE))
        store.add(_make_entry(feedback_type=FeedbackType.FALSE_NEGATIVE))
        store.add(_make_entry(feedback_type=FeedbackType.CORRECT_BLOCK))

        fp = store.get_all(FeedbackType.FALSE_POSITIVE)
        assert len(fp) == 1
        assert fp[0].feedback_type == FeedbackType.FALSE_POSITIVE

        fn = store.get_all(FeedbackType.FALSE_NEGATIVE)
        assert len(fn) == 1

    def test_get_all_returns_feedback_entry_objects(self, store):
        store.add(_make_entry())
        results = store.get_all()
        assert isinstance(results[0], FeedbackEntry)

    def test_get_all_no_filter_returns_all_types(self, store):
        for ft in FeedbackType:
            store.add(_make_entry(feedback_type=ft))
        results = store.get_all()
        assert len(results) == len(list(FeedbackType))


# ---------------------------------------------------------------------------
# FeedbackStore — get_stats()
# ---------------------------------------------------------------------------

class TestFeedbackStoreGetStats:
    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total"] == 0
        assert stats["false_positives"] == 0
        assert stats["false_negatives"] == 0
        assert stats["correct_blocks"] == 0
        assert stats["correct_allows"] == 0

    def test_stats_counts(self, store):
        store.add(_make_entry(feedback_type=FeedbackType.FALSE_POSITIVE))
        store.add(_make_entry(feedback_type=FeedbackType.FALSE_POSITIVE))
        store.add(_make_entry(feedback_type=FeedbackType.FALSE_NEGATIVE))
        store.add(_make_entry(feedback_type=FeedbackType.CORRECT_BLOCK))
        store.add(_make_entry(feedback_type=FeedbackType.CORRECT_ALLOW))

        stats = store.get_stats()
        assert stats["total"] == 5
        assert stats["false_positives"] == 2
        assert stats["false_negatives"] == 1
        assert stats["correct_blocks"] == 1
        assert stats["correct_allows"] == 1


# ---------------------------------------------------------------------------
# TuningSuggester — suggest_keyword_additions()
# ---------------------------------------------------------------------------

class TestTuningSuggesterKeywords:
    def test_no_suggestions_when_empty(self, store):
        suggester = TuningSuggester(store)
        assert suggester.suggest_keyword_additions() == []

    def test_no_suggestions_below_threshold(self, store):
        # Only 2 entries, threshold is 3
        store.add(_make_entry(text="harmful attack attempt", feedback_type=FeedbackType.FALSE_NEGATIVE))
        store.add(_make_entry(text="harmful attack attempt", feedback_type=FeedbackType.FALSE_NEGATIVE))
        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_keyword_additions(min_occurrences=3)
        assert suggestions == []

    def test_suggestions_above_threshold(self, store):
        for _ in range(5):
            store.add(_make_entry(
                text="dangerous harmful attack content",
                feedback_type=FeedbackType.FALSE_NEGATIVE,
            ))
        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_keyword_additions(min_occurrences=3)
        assert len(suggestions) > 0
        # All should have keys: keyword, occurrences, suggestion
        for s in suggestions:
            assert "keyword" in s
            assert "occurrences" in s
            assert s["occurrences"] >= 3

    def test_only_false_negatives_used(self, store):
        # False positives should not contribute to keyword suggestions
        for _ in range(10):
            store.add(_make_entry(
                text="innocent text content here",
                feedback_type=FeedbackType.FALSE_POSITIVE,
            ))
        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_keyword_additions(min_occurrences=3)
        assert suggestions == []


# ---------------------------------------------------------------------------
# TuningSuggester — suggest_rule_relaxation()
# ---------------------------------------------------------------------------

class TestTuningSuggesterRelaxation:
    def test_no_suggestions_when_empty(self, store):
        suggester = TuningSuggester(store)
        assert suggester.suggest_rule_relaxation() == []

    def test_no_suggestions_below_threshold(self, store):
        for _ in range(3):
            store.add(_make_entry(
                feedback_type=FeedbackType.FALSE_POSITIVE,
                matched_rules=["over_eager_rule"],
            ))
        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=5)
        assert suggestions == []

    def test_suggests_relaxation_above_threshold(self, store):
        for _ in range(6):
            store.add(_make_entry(
                feedback_type=FeedbackType.FALSE_POSITIVE,
                matched_rules=["over_eager_rule"],
            ))
        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=5)
        assert len(suggestions) == 1
        assert suggestions[0]["rule_id"] == "over_eager_rule"
        assert suggestions[0]["false_positive_count"] == 6

    def test_only_false_positives_used(self, store):
        for _ in range(10):
            store.add(_make_entry(
                feedback_type=FeedbackType.FALSE_NEGATIVE,
                matched_rules=["some_rule"],
            ))
        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=5)
        assert suggestions == []


# ---------------------------------------------------------------------------
# TuningSuggester — generate_report()
# ---------------------------------------------------------------------------

class TestTuningSuggesterReport:
    def test_report_contains_headers(self, store):
        suggester = TuningSuggester(store)
        report = suggester.generate_report()
        assert "Guardrail Tuning Report" in report
        assert "Feedback Summary" in report

    def test_report_with_data(self, store):
        store.add(_make_entry(feedback_type=FeedbackType.FALSE_POSITIVE))
        store.add(_make_entry(feedback_type=FeedbackType.FALSE_NEGATIVE))
        suggester = TuningSuggester(store)
        report = suggester.generate_report()
        assert "1" in report  # counts appear

    def test_report_includes_keyword_suggestions_section(self, store):
        # Add enough false negatives to trigger the keyword suggestions section (threshold=3)
        for _ in range(5):
            store.add(_make_entry(
                text="dangerous harmful attack content here",
                feedback_type=FeedbackType.FALSE_NEGATIVE,
            ))
        suggester = TuningSuggester(store)
        report = suggester.generate_report()
        assert "Suggested Keyword Additions" in report

    def test_report_includes_relaxation_suggestions_section(self, store):
        # Add enough false positives for rule relaxation suggestion (threshold=5)
        for _ in range(6):
            store.add(_make_entry(
                feedback_type=FeedbackType.FALSE_POSITIVE,
                matched_rules=["over_eager_rule"],
            ))
        suggester = TuningSuggester(store)
        report = suggester.generate_report()
        assert "Suggested Rule Relaxations" in report


# ---------------------------------------------------------------------------
# create_feedback_entry helper
# ---------------------------------------------------------------------------

class TestCreateFeedbackEntry:
    def test_returns_feedback_entry(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.FALSE_NEGATIVE)
        assert isinstance(entry, FeedbackEntry)

    def test_id_is_none(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.FALSE_NEGATIVE)
        assert entry.id is None

    def test_timestamp_set(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.FALSE_NEGATIVE)
        assert entry.timestamp

    def test_false_positive_expected_action(self):
        entry = create_feedback_entry("text", "block", FeedbackType.FALSE_POSITIVE)
        assert entry.expected_action == "allow"

    def test_false_negative_expected_action(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.FALSE_NEGATIVE)
        assert entry.expected_action == "block"

    def test_correct_block_expected_action(self):
        entry = create_feedback_entry("text", "block", FeedbackType.CORRECT_BLOCK)
        assert entry.expected_action == "block"

    def test_optional_fields(self):
        entry = create_feedback_entry(
            "text", "allow", FeedbackType.FALSE_NEGATIVE,
            matched_rules=["r1"], user_id="user1", comment="a comment",
        )
        assert entry.matched_rules == ["r1"]
        assert entry.user_id == "user1"
        assert entry.comment == "a comment"

    def test_defaults(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.CORRECT_ALLOW)
        assert entry.matched_rules == []
        assert entry.user_id is None
        assert entry.comment == ""
