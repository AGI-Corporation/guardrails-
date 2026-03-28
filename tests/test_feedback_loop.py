"""
Tests for feedback_loop.py
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

@pytest.fixture()
def store(tmp_path):
    """Fresh FeedbackStore backed by a temp SQLite DB."""
    return FeedbackStore(db_path=str(tmp_path / "feedback.db"))


@pytest.fixture()
def suggester(store):
    return TuningSuggester(store=store)


# ---------------------------------------------------------------------------
# create_feedback_entry helper
# ---------------------------------------------------------------------------

class TestCreateFeedbackEntry:
    def test_false_positive_sets_expected_allow(self):
        entry = create_feedback_entry(
            text="hello",
            original_action="block",
            feedback_type=FeedbackType.FALSE_POSITIVE,
        )
        assert entry.expected_action == "allow"
        assert entry.id is None
        assert entry.timestamp != ""

    def test_false_negative_sets_expected_block(self):
        entry = create_feedback_entry(
            text="bad content",
            original_action="allow",
            feedback_type=FeedbackType.FALSE_NEGATIVE,
        )
        assert entry.expected_action == "block"

    def test_optional_fields_default(self):
        entry = create_feedback_entry(
            text="test",
            original_action="allow",
            feedback_type=FeedbackType.CORRECT_ALLOW,
        )
        assert entry.user_id is None
        assert entry.comment == ""
        assert entry.matched_rules == []

    def test_custom_fields(self):
        entry = create_feedback_entry(
            text="test",
            original_action="block",
            feedback_type=FeedbackType.CORRECT_BLOCK,
            matched_rules=["rule_1"],
            user_id="alice",
            comment="Looks right",
        )
        assert entry.user_id == "alice"
        assert entry.comment == "Looks right"
        assert entry.matched_rules == ["rule_1"]


# ---------------------------------------------------------------------------
# FeedbackType enum
# ---------------------------------------------------------------------------

class TestFeedbackType:
    def test_values(self):
        assert FeedbackType.FALSE_POSITIVE.value == "false_positive"
        assert FeedbackType.FALSE_NEGATIVE.value == "false_negative"
        assert FeedbackType.CORRECT_BLOCK.value == "correct_block"
        assert FeedbackType.CORRECT_ALLOW.value == "correct_allow"


# ---------------------------------------------------------------------------
# FeedbackStore — add and retrieve
# ---------------------------------------------------------------------------

class TestFeedbackStoreAdd:
    def test_add_returns_positive_id(self, store):
        entry = create_feedback_entry(
            text="test", original_action="allow", feedback_type=FeedbackType.CORRECT_ALLOW
        )
        row_id = store.add(entry)
        assert row_id > 0

    def test_added_entry_is_retrievable(self, store):
        entry = create_feedback_entry(
            text="specific text",
            original_action="block",
            feedback_type=FeedbackType.FALSE_POSITIVE,
            user_id="alice",
        )
        store.add(entry)
        all_entries = store.get_all()
        assert len(all_entries) == 1
        assert all_entries[0].text == "specific text"
        assert all_entries[0].user_id == "alice"
        assert all_entries[0].feedback_type == FeedbackType.FALSE_POSITIVE

    def test_matched_rules_persisted(self, store):
        entry = create_feedback_entry(
            text="test",
            original_action="block",
            feedback_type=FeedbackType.CORRECT_BLOCK,
            matched_rules=["rule_a", "rule_b"],
        )
        store.add(entry)
        retrieved = store.get_all()[0]
        assert "rule_a" in retrieved.matched_rules
        assert "rule_b" in retrieved.matched_rules

    def test_multiple_entries(self, store):
        for i in range(5):
            store.add(
                create_feedback_entry(
                    text=f"text {i}",
                    original_action="allow",
                    feedback_type=FeedbackType.CORRECT_ALLOW,
                )
            )
        assert len(store.get_all()) == 5


# ---------------------------------------------------------------------------
# FeedbackStore — get_all with filter
# ---------------------------------------------------------------------------

class TestFeedbackStoreGetAll:
    def test_filter_by_false_positive(self, store):
        store.add(create_feedback_entry("t1", "block", FeedbackType.FALSE_POSITIVE))
        store.add(create_feedback_entry("t2", "allow", FeedbackType.CORRECT_ALLOW))
        fps = store.get_all(FeedbackType.FALSE_POSITIVE)
        assert len(fps) == 1
        assert fps[0].text == "t1"

    def test_filter_by_false_negative(self, store):
        store.add(create_feedback_entry("t1", "allow", FeedbackType.FALSE_NEGATIVE))
        store.add(create_feedback_entry("t2", "allow", FeedbackType.CORRECT_ALLOW))
        fns = store.get_all(FeedbackType.FALSE_NEGATIVE)
        assert len(fns) == 1

    def test_no_filter_returns_all(self, store):
        for ft in FeedbackType:
            store.add(create_feedback_entry("x", "allow", ft))
        all_entries = store.get_all()
        assert len(all_entries) == len(list(FeedbackType))

    def test_empty_store_returns_empty_list(self, store):
        assert store.get_all() == []


# ---------------------------------------------------------------------------
# FeedbackStore — get_stats
# ---------------------------------------------------------------------------

class TestFeedbackStoreStats:
    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total"] == 0
        assert stats["false_positives"] == 0
        assert stats["false_negatives"] == 0
        assert stats["correct_blocks"] == 0
        assert stats["correct_allows"] == 0

    def test_stats_counts(self, store):
        store.add(create_feedback_entry("t1", "block", FeedbackType.FALSE_POSITIVE))
        store.add(create_feedback_entry("t2", "block", FeedbackType.FALSE_POSITIVE))
        store.add(create_feedback_entry("t3", "allow", FeedbackType.FALSE_NEGATIVE))
        store.add(create_feedback_entry("t4", "block", FeedbackType.CORRECT_BLOCK))
        store.add(create_feedback_entry("t5", "allow", FeedbackType.CORRECT_ALLOW))
        stats = store.get_stats()
        assert stats["total"] == 5
        assert stats["false_positives"] == 2
        assert stats["false_negatives"] == 1
        assert stats["correct_blocks"] == 1
        assert stats["correct_allows"] == 1


# ---------------------------------------------------------------------------
# TuningSuggester — suggest_keyword_additions
# ---------------------------------------------------------------------------

class TestTuningSuggesterKeywords:
    def test_no_false_negatives_no_suggestions(self, suggester):
        suggestions = suggester.suggest_keyword_additions(min_occurrences=1)
        assert suggestions == []

    def test_common_words_suggested(self, store, suggester):
        # Add many false negatives with "badword" appearing repeatedly
        for _ in range(5):
            store.add(
                create_feedback_entry(
                    text="contains badword in this text",
                    original_action="allow",
                    feedback_type=FeedbackType.FALSE_NEGATIVE,
                )
            )
        suggestions = suggester.suggest_keyword_additions(min_occurrences=3)
        keywords = [s["keyword"] for s in suggestions]
        assert "badword" in keywords

    def test_rare_words_below_threshold_excluded(self, store, suggester):
        store.add(
            create_feedback_entry(
                text="uniqueword only once",
                original_action="allow",
                feedback_type=FeedbackType.FALSE_NEGATIVE,
            )
        )
        suggestions = suggester.suggest_keyword_additions(min_occurrences=3)
        keywords = [s["keyword"] for s in suggestions]
        assert "uniqueword" not in keywords

    def test_suggestion_structure(self, store, suggester):
        for _ in range(4):
            store.add(
                create_feedback_entry(
                    text="harmful dangerous stuff",
                    original_action="allow",
                    feedback_type=FeedbackType.FALSE_NEGATIVE,
                )
            )
        suggestions = suggester.suggest_keyword_additions(min_occurrences=3)
        for s in suggestions:
            assert "keyword" in s
            assert "occurrences" in s
            assert "suggestion" in s


# ---------------------------------------------------------------------------
# TuningSuggester — suggest_rule_relaxation
# ---------------------------------------------------------------------------

class TestTuningSuggesterRelaxation:
    def test_no_false_positives_no_suggestions(self, suggester):
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=1)
        assert suggestions == []

    def test_frequently_triggering_rule_suggested(self, store, suggester):
        for _ in range(6):
            store.add(
                create_feedback_entry(
                    text="false alarm",
                    original_action="block",
                    feedback_type=FeedbackType.FALSE_POSITIVE,
                    matched_rules=["overly_strict_rule"],
                )
            )
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=5)
        rule_ids = [s["rule_id"] for s in suggestions]
        assert "overly_strict_rule" in rule_ids

    def test_infrequent_rule_excluded(self, store, suggester):
        store.add(
            create_feedback_entry(
                text="edge case",
                original_action="block",
                feedback_type=FeedbackType.FALSE_POSITIVE,
                matched_rules=["rare_rule"],
            )
        )
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=5)
        rule_ids = [s["rule_id"] for s in suggestions]
        assert "rare_rule" not in rule_ids


# ---------------------------------------------------------------------------
# TuningSuggester — generate_report
# ---------------------------------------------------------------------------

class TestTuningSuggesterReport:
    def test_report_contains_summary(self, store, suggester):
        store.add(create_feedback_entry("t", "block", FeedbackType.FALSE_POSITIVE))
        report = suggester.generate_report()
        assert "Tuning Report" in report
        assert "false positives" in report.lower()

    def test_empty_report_still_generates(self, suggester):
        report = suggester.generate_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_includes_keyword_suggestion_section(self, store, suggester):
        """Cover the 'if keyword_suggestions:' branch in generate_report."""
        for _ in range(5):
            store.add(
                create_feedback_entry(
                    text="contains badthing in this sentence",
                    original_action="allow",
                    feedback_type=FeedbackType.FALSE_NEGATIVE,
                )
            )
        report = suggester.generate_report()
        assert "Keyword" in report or "keyword" in report

    def test_report_includes_relaxation_section(self, store, suggester):
        """Cover the 'if relaxation_suggestions:' branch in generate_report."""
        for _ in range(6):
            store.add(
                create_feedback_entry(
                    text="false alarm content",
                    original_action="block",
                    feedback_type=FeedbackType.FALSE_POSITIVE,
                    matched_rules=["too_strict_rule"],
                )
            )
        report = suggester.generate_report()
        assert "Relaxation" in report or "Relax" in report or "too_strict_rule" in report
