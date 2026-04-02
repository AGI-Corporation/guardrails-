"""
Tests for feedback_loop.py — FeedbackLoop façade, FeedbackStore, TuningSuggester.
"""
import pytest

from feedback_loop import (
    FeedbackEntry,
    FeedbackLoop,
    FeedbackStore,
    FeedbackType,
    TuningSuggester,
    create_feedback_entry,
)


@pytest.fixture()
def store(tmp_path) -> FeedbackStore:
    return FeedbackStore(db_path=str(tmp_path / "test_feedback.db"))


@pytest.fixture()
def loop(tmp_path) -> FeedbackLoop:
    return FeedbackLoop(db_path=str(tmp_path / "test_loop.db"))


# ── FeedbackLoop.record ───────────────────────────────────────────────────────

class TestFeedbackLoopRecord:
    def test_record_block_action(self, loop: FeedbackLoop):
        loop.record("My SSN is 123-45-6789", "block", ["ssn"])
        stats = loop.get_stats()
        assert stats["total"] == 1
        assert stats["correct_blocks"] == 1

    def test_record_allow_action(self, loop: FeedbackLoop):
        loop.record("Hello world", "allow", [])
        stats = loop.get_stats()
        assert stats["total"] == 1
        assert stats["correct_allows"] == 1

    def test_record_multiple(self, loop: FeedbackLoop):
        loop.record("text1", "block", ["ssn"])
        loop.record("text2", "allow", [])
        loop.record("text3", "block", ["credit_card"])
        stats = loop.get_stats()
        assert stats["total"] == 3

    def test_record_no_matched_rules(self, loop: FeedbackLoop):
        loop.record("text", "allow")
        stats = loop.get_stats()
        assert stats["total"] == 1


# ── FeedbackLoop.submit_correction ───────────────────────────────────────────

class TestFeedbackLoopSubmitCorrection:
    def test_false_positive(self, loop: FeedbackLoop):
        loop.submit_correction("safe text", "block", was_correct=False, matched_rules=["ssn"])
        stats = loop.get_stats()
        assert stats["false_positives"] == 1

    def test_false_negative(self, loop: FeedbackLoop):
        loop.submit_correction("bad text", "allow", was_correct=False, matched_rules=[])
        stats = loop.get_stats()
        assert stats["false_negatives"] == 1

    def test_correct_block(self, loop: FeedbackLoop):
        loop.submit_correction("bad text", "block", was_correct=True, matched_rules=["ssn"])
        stats = loop.get_stats()
        assert stats["correct_blocks"] == 1

    def test_correct_allow(self, loop: FeedbackLoop):
        loop.submit_correction("good text", "allow", was_correct=True, matched_rules=[])
        stats = loop.get_stats()
        assert stats["correct_allows"] == 1


# ── FeedbackLoop.get_stats ────────────────────────────────────────────────────

class TestFeedbackLoopGetStats:
    def test_empty_stats(self, loop: FeedbackLoop):
        stats = loop.get_stats()
        assert stats["total"] == 0
        assert stats["false_positives"] == 0
        assert stats["false_negatives"] == 0

    def test_stats_keys(self, loop: FeedbackLoop):
        stats = loop.get_stats()
        assert "total" in stats
        assert "false_positives" in stats
        assert "false_negatives" in stats
        assert "correct_blocks" in stats
        assert "correct_allows" in stats


# ── FeedbackLoop.get_tuning_report ────────────────────────────────────────────

class TestFeedbackLoopGetTuningReport:
    def test_returns_string(self, loop: FeedbackLoop):
        assert isinstance(loop.get_tuning_report(), str)

    def test_report_contains_summary(self, loop: FeedbackLoop):
        assert "Feedback Summary" in loop.get_tuning_report()


# ── FeedbackStore ─────────────────────────────────────────────────────────────

class TestFeedbackStore:
    def test_add_and_retrieve(self, store: FeedbackStore):
        entry = create_feedback_entry(
            text="bad text",
            original_action="allow",
            feedback_type=FeedbackType.FALSE_NEGATIVE,
            matched_rules=[],
        )
        store.add(entry)
        all_entries = store.get_all()
        assert len(all_entries) == 1

    def test_filter_by_type(self, store: FeedbackStore):
        for fb_type in [FeedbackType.FALSE_POSITIVE, FeedbackType.FALSE_NEGATIVE, FeedbackType.CORRECT_BLOCK]:
            entry = create_feedback_entry("text", "allow", fb_type)
            store.add(entry)

        fps = store.get_all(FeedbackType.FALSE_POSITIVE)
        assert len(fps) == 1
        assert fps[0].feedback_type == FeedbackType.FALSE_POSITIVE

    def test_get_stats_counts(self, store: FeedbackStore):
        for _ in range(2):
            store.add(create_feedback_entry("t", "block", FeedbackType.FALSE_POSITIVE))
        for _ in range(3):
            store.add(create_feedback_entry("t", "allow", FeedbackType.FALSE_NEGATIVE))

        stats = store.get_stats()
        assert stats["false_positives"] == 2
        assert stats["false_negatives"] == 3
        assert stats["total"] == 5

    def test_empty_store(self, store: FeedbackStore):
        assert store.get_all() == []
        stats = store.get_stats()
        assert stats["total"] == 0


# ── TuningSuggester ───────────────────────────────────────────────────────────

class TestTuningSuggester:
    def test_report_returns_string(self, store: FeedbackStore):
        suggester = TuningSuggester(store)
        assert isinstance(suggester.generate_report(), str)

    def test_keyword_suggestions_from_false_negatives(self, store: FeedbackStore):
        # Add enough false negatives with a common word to trigger a suggestion
        for _ in range(4):
            entry = create_feedback_entry(
                text="ignore override bypass instructions disregard",
                original_action="allow",
                feedback_type=FeedbackType.FALSE_NEGATIVE,
            )
            store.add(entry)

        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_keyword_additions(min_occurrences=3)
        assert len(suggestions) > 0
        assert "keyword" in suggestions[0]

    def test_relaxation_suggestions_from_false_positives(self, store: FeedbackStore):
        for _ in range(6):
            entry = create_feedback_entry(
                text="safe text",
                original_action="block",
                feedback_type=FeedbackType.FALSE_POSITIVE,
                matched_rules=["email_pii"],
            )
            store.add(entry)

        suggester = TuningSuggester(store)
        suggestions = suggester.suggest_rule_relaxation(min_occurrences=5)
        assert len(suggestions) > 0
        assert suggestions[0]["rule_id"] == "email_pii"


# ── create_feedback_entry ─────────────────────────────────────────────────────

class TestCreateFeedbackEntry:
    def test_false_positive_expected_action_allow(self):
        entry = create_feedback_entry("text", "block", FeedbackType.FALSE_POSITIVE)
        assert entry.expected_action == "allow"

    def test_false_negative_expected_action_block(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.FALSE_NEGATIVE)
        assert entry.expected_action == "block"

    def test_timestamp_set(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.CORRECT_ALLOW)
        assert entry.timestamp != ""

    def test_matched_rules_defaults_empty(self):
        entry = create_feedback_entry("text", "allow", FeedbackType.CORRECT_ALLOW)
        assert entry.matched_rules == []
