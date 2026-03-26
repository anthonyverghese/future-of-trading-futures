"""
Tests for outcome_tracker.OutcomeEvaluator.

Uses dependency injection (noop/recording stubs) for on_hit_fn and on_outcome_fn
so no real DB or cache module is needed.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

import pytest

# Allow imports from the mnq_alerts package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import outcome_tracker  # noqa: E402
from outcome_tracker import OutcomeEvaluator  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt(
    minutes: float = 0, base_hour: int = 10, base_min: int = 0
) -> datetime.datetime:
    """Return a datetime offset by *minutes* from a base time."""
    base = datetime.datetime(2026, 3, 25, base_hour, base_min, 0)
    return base + datetime.timedelta(minutes=minutes)


def _make_evaluator(
    prior_outcomes: list[str] | None = None,
    tmp_path=None,
):
    """Create an OutcomeEvaluator with recording stubs."""
    hits: list[tuple] = []
    outcomes: list[tuple] = []

    def record_hit(alert_id, hit_time):
        hits.append((alert_id, hit_time))

    def record_outcome(alert_id, outcome, date_str):
        outcomes.append((alert_id, outcome, date_str))

    ev = OutcomeEvaluator(
        prior_outcomes=prior_outcomes,
        on_hit_fn=record_hit,
        on_outcome_fn=record_outcome,
    )
    return ev, hits, outcomes


# ---------------------------------------------------------------------------
# 1. Price hits target → "correct"
# ---------------------------------------------------------------------------


class TestTargetHit:
    def test_buy_target_correct(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=1,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        # Price reaches the line (within HIT_THRESHOLD=1.0).
        ev.update(line + 0.5, _dt(1))
        assert len(hits) == 1
        assert hits[0][0] == 1

        # Price moves up to target (line + 8).
        ev.update(line + 8.0, _dt(2))
        assert len(outcomes) == 1
        assert outcomes[0] == (1, "correct", "2026-03-25")
        assert ev.consecutive_wins == 1

    def test_sell_target_correct(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=2,
            line_price=line,
            direction="down",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        ev.update(line - 0.5, _dt(1))  # hit
        ev.update(line - 8.0, _dt(2))  # target
        assert outcomes[0] == (2, "correct", "2026-03-25")


# ---------------------------------------------------------------------------
# 2. Price hits stop → "incorrect"
# ---------------------------------------------------------------------------


class TestStopHit:
    def test_buy_stop_incorrect(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=3,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        ev.update(line, _dt(1))  # hit the line
        ev.update(line - 20.0, _dt(2))  # stop hit (line - 20)
        assert outcomes[0] == (3, "incorrect", "2026-03-25")
        assert ev.consecutive_losses == 1

    def test_sell_stop_incorrect(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=4,
            line_price=line,
            direction="down",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        ev.update(line, _dt(1))  # hit
        ev.update(line + 20.0, _dt(2))  # stop (line + 20)
        assert outcomes[0] == (4, "incorrect", "2026-03-25")


# ---------------------------------------------------------------------------
# 3. Timeout after hit → "incorrect"
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_after_hit_is_incorrect(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=5,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        ev.update(line, _dt(1))  # hit
        # Price drifts slightly but never reaches target or stop.
        ev.update(line + 2.0, _dt(5))
        ev.update(line + 1.0, _dt(10))
        assert len(outcomes) == 0

        # 15 minutes after hit_time → timeout.
        ev.update(line + 1.0, _dt(16))
        assert outcomes[0] == (5, "incorrect", "2026-03-25")
        assert ev.consecutive_losses == 1


# ---------------------------------------------------------------------------
# 4. Price never reaches line → "inconclusive"
# ---------------------------------------------------------------------------


class TestInconclusive:
    def test_never_hits_line_is_inconclusive(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=6,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        # Price stays far from the line for 15+ minutes.
        ev.update(line + 50.0, _dt(5))
        ev.update(line + 50.0, _dt(10))
        assert len(outcomes) == 0

        ev.update(line + 50.0, _dt(15))
        assert outcomes[0] == (6, "inconclusive", "2026-03-25")
        # Inconclusive should NOT affect streak.
        assert ev.consecutive_wins == 0
        assert ev.consecutive_losses == 0


# ---------------------------------------------------------------------------
# 5. BUY direction thresholds
# ---------------------------------------------------------------------------


class TestBuyDirection:
    def test_buy_target_is_line_plus_8(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=10,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.update(line, _dt(1))  # hit

        # Just below target — no resolution yet.
        ev.update(line + 7.9, _dt(2))
        assert len(outcomes) == 0

        # At target.
        ev.update(line + 8.0, _dt(3))
        assert outcomes[0][1] == "correct"

    def test_buy_stop_is_line_minus_20(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=11,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.update(line, _dt(1))  # hit

        # Just above stop — no resolution.
        ev.update(line - 19.9, _dt(2))
        assert len(outcomes) == 0

        # At stop.
        ev.update(line - 20.0, _dt(3))
        assert outcomes[0][1] == "incorrect"


# ---------------------------------------------------------------------------
# 6. SELL direction thresholds
# ---------------------------------------------------------------------------


class TestSellDirection:
    def test_sell_target_is_line_minus_8(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=20,
            line_price=line,
            direction="down",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.update(line, _dt(1))  # hit

        ev.update(line - 7.9, _dt(2))
        assert len(outcomes) == 0

        ev.update(line - 8.0, _dt(3))
        assert outcomes[0][1] == "correct"

    def test_sell_stop_is_line_plus_20(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=21,
            line_price=line,
            direction="down",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.update(line, _dt(1))  # hit

        ev.update(line + 19.9, _dt(2))
        assert len(outcomes) == 0

        ev.update(line + 20.0, _dt(3))
        assert outcomes[0][1] == "incorrect"


# ---------------------------------------------------------------------------
# 7. consecutive_wins / consecutive_losses tracking
# ---------------------------------------------------------------------------


class TestStreaks:
    def test_consecutive_wins(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, _ = _make_evaluator(prior_outcomes=["correct", "correct", "correct"])
        assert ev.consecutive_wins == 3
        assert ev.consecutive_losses == 0

    def test_consecutive_losses(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, _ = _make_evaluator(prior_outcomes=["incorrect", "incorrect"])
        assert ev.consecutive_losses == 2
        assert ev.consecutive_wins == 0

    def test_streak_broken(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, _ = _make_evaluator(prior_outcomes=["correct", "incorrect", "correct"])
        assert ev.consecutive_wins == 1
        assert ev.consecutive_losses == 0

    def test_streak_updates_with_new_outcomes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, outcomes = _make_evaluator(prior_outcomes=["correct"])
        line = 20000.0

        # Add and resolve a correct trade.
        ev.add(
            alert_id=30,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.update(line, _dt(1))
        ev.update(line + 8.0, _dt(2))
        assert ev.consecutive_wins == 2

        # Now add and resolve an incorrect trade — breaks the win streak.
        ev.add(
            alert_id=31,
            line_price=line,
            direction="up",
            alert_time=_dt(3),
            date_str="2026-03-25",
        )
        ev.update(line, _dt(4))
        ev.update(line - 20.0, _dt(5))
        assert ev.consecutive_wins == 0
        assert ev.consecutive_losses == 1


# ---------------------------------------------------------------------------
# 8. add_untracked feeds streak but doesn't call DB functions
# ---------------------------------------------------------------------------


class TestUntracked:
    def test_untracked_no_db_calls(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add_untracked(line_price=line, direction="up", alert_time=_dt(0))

        ev.update(line, _dt(1))  # hit — should NOT call on_hit
        ev.update(line + 8.0, _dt(2))  # target — should NOT call on_outcome

        assert len(hits) == 0
        assert len(outcomes) == 0
        # But streak should still update.
        assert ev.consecutive_wins == 1

    def test_untracked_inconclusive_no_db(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add_untracked(line_price=line, direction="up", alert_time=_dt(0))

        # Price never reaches line.
        ev.update(line + 50.0, _dt(15))
        assert len(hits) == 0
        assert len(outcomes) == 0

    def test_untracked_stop_feeds_streak(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add_untracked(line_price=line, direction="up", alert_time=_dt(0))
        ev.update(line, _dt(1))
        ev.update(line - 20.0, _dt(2))

        assert len(hits) == 0
        assert len(outcomes) == 0
        assert ev.consecutive_losses == 1


# ---------------------------------------------------------------------------
# 9. Tracked entries call on_hit_fn and on_outcome_fn
# ---------------------------------------------------------------------------


class TestTrackedCallbacks:
    def test_tracked_calls_on_hit_and_on_outcome(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, hits, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=40,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        ev.update(line, _dt(1))
        assert len(hits) == 1
        assert hits[0][0] == 40

        ev.update(line + 8.0, _dt(2))
        assert len(outcomes) == 1
        assert outcomes[0][0] == 40
        assert outcomes[0][1] == "correct"


# ---------------------------------------------------------------------------
# 10. close_session marks remaining as inconclusive
# ---------------------------------------------------------------------------


class TestCloseSession:
    def test_close_session_marks_tracked_inconclusive(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, outcomes = _make_evaluator()
        line = 20000.0

        ev.add(
            alert_id=50,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.add(
            alert_id=51,
            line_price=line + 100,
            direction="down",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        ev.close_session()

        assert len(outcomes) == 2
        assert all(o[1] == "inconclusive" for o in outcomes)

    def test_close_session_skips_untracked(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, outcomes = _make_evaluator()
        line = 20000.0

        ev.add_untracked(line_price=line, direction="up", alert_time=_dt(0))
        ev.add(
            alert_id=52,
            line_price=line,
            direction="down",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )

        ev.close_session()

        # Only the tracked one should have called on_outcome.
        assert len(outcomes) == 1
        assert outcomes[0][0] == 52

    def test_close_session_clears_pending(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, _ = _make_evaluator()

        ev.add(
            alert_id=53,
            line_price=20000.0,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.close_session()

        # After close, update should do nothing.
        ev, _, outcomes2 = _make_evaluator()
        ev.update(20000.0, _dt(1))
        assert len(outcomes2) == 0


# ---------------------------------------------------------------------------
# 11. Streak file persistence (tmp_path fixture)
# ---------------------------------------------------------------------------


class TestStreakFilePersistence:
    def test_save_and_load_streak(self, monkeypatch, tmp_path):
        streak_file = str(tmp_path / "streak.json")
        monkeypatch.setattr(outcome_tracker, "_STREAK_FILE", streak_file)

        ev, _, _ = _make_evaluator()
        line = 20000.0

        # Generate a correct outcome to write the streak file.
        ev.add(
            alert_id=60,
            line_price=line,
            direction="up",
            alert_time=_dt(0),
            date_str="2026-03-25",
        )
        ev.update(line, _dt(1))
        ev.update(line + 8.0, _dt(2))

        # The streak file should now exist with ["correct"].
        assert os.path.exists(streak_file)
        with open(streak_file) as f:
            data = json.load(f)
        assert data == ["correct"]

        # A new evaluator should load from the file, ignoring prior_outcomes.
        ev2, _, _ = _make_evaluator(prior_outcomes=["incorrect", "incorrect"])
        assert ev2.consecutive_wins == 1
        assert ev2.consecutive_losses == 0

    def test_no_streak_file_uses_prior_outcomes(self, monkeypatch, tmp_path):
        streak_file = str(tmp_path / "nonexistent_streak.json")
        monkeypatch.setattr(outcome_tracker, "_STREAK_FILE", streak_file)

        ev, _, _ = _make_evaluator(prior_outcomes=["incorrect", "incorrect"])
        assert ev.consecutive_losses == 2

    def test_corrupt_streak_file_uses_prior_outcomes(self, monkeypatch, tmp_path):
        streak_file = str(tmp_path / "streak.json")
        monkeypatch.setattr(outcome_tracker, "_STREAK_FILE", streak_file)

        with open(streak_file, "w") as f:
            f.write("NOT VALID JSON{{{")

        ev, _, _ = _make_evaluator(prior_outcomes=["correct"])
        assert ev.consecutive_wins == 1

    def test_streak_file_respects_limit(self, monkeypatch, tmp_path):
        streak_file = str(tmp_path / "streak.json")
        monkeypatch.setattr(outcome_tracker, "_STREAK_FILE", streak_file)

        # Pre-seed with 25 outcomes (limit is 20).
        with open(streak_file, "w") as f:
            json.dump(["correct"] * 25, f)

        ev, _, _ = _make_evaluator()
        # Should truncate to last 20.
        assert len(ev._recent_outcomes) == 20


# ---------------------------------------------------------------------------
# 12. Prior outcomes seeding
# ---------------------------------------------------------------------------


class TestPriorOutcomes:
    def test_prior_outcomes_seed_streak(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, _ = _make_evaluator(prior_outcomes=["incorrect", "correct", "correct"])
        assert ev.consecutive_wins == 2
        assert ev.consecutive_losses == 0

    def test_empty_prior_outcomes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, _ = _make_evaluator(prior_outcomes=[])
        assert ev.consecutive_wins == 0
        assert ev.consecutive_losses == 0

    def test_none_prior_outcomes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            outcome_tracker, "_STREAK_FILE", str(tmp_path / "streak.json")
        )
        ev, _, _ = _make_evaluator(prior_outcomes=None)
        assert ev.consecutive_wins == 0
        assert ev.consecutive_losses == 0

    def test_streak_file_takes_precedence_over_prior(self, monkeypatch, tmp_path):
        streak_file = str(tmp_path / "streak.json")
        monkeypatch.setattr(outcome_tracker, "_STREAK_FILE", streak_file)

        with open(streak_file, "w") as f:
            json.dump(["incorrect", "incorrect", "incorrect"], f)

        ev, _, _ = _make_evaluator(prior_outcomes=["correct", "correct"])
        # File should win.
        assert ev.consecutive_losses == 3
        assert ev.consecutive_wins == 0
