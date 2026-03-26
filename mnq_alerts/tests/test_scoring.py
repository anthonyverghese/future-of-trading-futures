"""Comprehensive tests for mnq_alerts/scoring.py."""

import datetime
import sys
from pathlib import Path

# Ensure the package is importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from scoring import MIN_SCORE, TIER_LABELS, composite_score, score_tier

# ---------------------------------------------------------------------------
# Helpers — a baseline call with neutral defaults that scores 0
# ---------------------------------------------------------------------------


def _base(**overrides):
    """Return kwargs for composite_score with all-neutral defaults."""
    defaults = dict(
        level_name="UNKNOWN",
        entry_count=4,  # count 4 adds 0
        now_et=datetime.time(12, 0),  # midday, not power hour
        tick_rate=1500.0,  # outside 1750-2000
        session_move_pts=None,
        direction=None,
        consecutive_wins=0,
        consecutive_losses=0,
    )
    defaults.update(overrides)
    return defaults


# ===================================================================
# 1. Level quality
# ===================================================================


class TestLevelQuality:
    def test_fib_ext_hi_plus2(self):
        assert composite_score(**_base(level_name="FIB_EXT_HI_1.272")) == 2

    def test_ibl_plus1(self):
        assert composite_score(**_base(level_name="IBL")) == 1

    def test_fib_ext_lo_plus1(self):
        assert composite_score(**_base(level_name="FIB_EXT_LO_1.272")) == 1

    def test_vwap_minus1(self):
        assert composite_score(**_base(level_name="VWAP")) == -1

    def test_ibh_minus1(self):
        assert composite_score(**_base(level_name="IBH")) == -1

    def test_unknown_level_zero(self):
        assert composite_score(**_base(level_name="SOMETHING_ELSE")) == 0


# ===================================================================
# 2. Direction x Level interactions
# ===================================================================


class TestDirectionLevel:
    # Strong combos
    def test_fib_hi_up_plus2(self):
        assert (
            composite_score(**_base(level_name="FIB_EXT_HI_1.272", direction="up"))
            == 2 + 2
        )

    def test_fib_lo_down_plus1(self):
        assert (
            composite_score(**_base(level_name="FIB_EXT_LO_1.272", direction="down"))
            == 1 + 1
        )

    def test_ibl_down_plus1(self):
        assert composite_score(**_base(level_name="IBL", direction="down")) == 1 + 1

    def test_vwap_up_plus1(self):
        assert composite_score(**_base(level_name="VWAP", direction="up")) == -1 + 1

    # Weak combos
    def test_ibh_up_minus1(self):
        assert composite_score(**_base(level_name="IBH", direction="up")) == -1 + (-1)

    def test_ibl_up_minus1(self):
        assert composite_score(**_base(level_name="IBL", direction="up")) == 1 + (-1)

    def test_fib_lo_up_minus1(self):
        assert composite_score(
            **_base(level_name="FIB_EXT_LO_1.272", direction="up")
        ) == 1 + (-1)

    def test_fib_hi_down_minus1(self):
        assert composite_score(
            **_base(level_name="FIB_EXT_HI_1.272", direction="down")
        ) == 2 + (-1)

    def test_vwap_down_minus1(self):
        assert composite_score(**_base(level_name="VWAP", direction="down")) == -1 + (
            -1
        )

    # No direction — direction block skipped entirely
    def test_no_direction(self):
        assert composite_score(**_base(level_name="IBL", direction=None)) == 1

    # Direction present but combo not listed (IBH + down) — no interaction bonus
    def test_ibh_down_no_interaction(self):
        # IBH level = -1, direction combo (IBH, down) not in any list => 0
        assert composite_score(**_base(level_name="IBH", direction="down")) == -1


# ===================================================================
# 3. Time of day — power hour
# ===================================================================


class TestTimeOfDay:
    def test_power_hour_1500_plus2(self):
        assert composite_score(**_base(now_et=datetime.time(15, 0))) == 2

    def test_power_hour_1530_plus2(self):
        assert composite_score(**_base(now_et=datetime.time(15, 30))) == 2

    def test_power_hour_1559_plus2(self):
        assert composite_score(**_base(now_et=datetime.time(15, 59))) == 2

    def test_before_power_hour_1459_zero(self):
        assert composite_score(**_base(now_et=datetime.time(14, 59))) == 0

    def test_midday_zero(self):
        assert composite_score(**_base(now_et=datetime.time(12, 0))) == 0

    def test_morning_zero(self):
        assert composite_score(**_base(now_et=datetime.time(9, 30))) == 0

    def test_none_time_zero(self):
        assert composite_score(**_base(now_et=None)) == 0


# ===================================================================
# 4. Tick rate
# ===================================================================


class TestTickRate:
    def test_in_band_1750_plus2(self):
        assert composite_score(**_base(tick_rate=1750.0)) == 2

    def test_in_band_1999_plus2(self):
        assert composite_score(**_base(tick_rate=1999.9)) == 2

    def test_in_band_1875_plus2(self):
        assert composite_score(**_base(tick_rate=1875.0)) == 2

    def test_below_band_1749_zero(self):
        assert composite_score(**_base(tick_rate=1749.9)) == 0

    def test_at_2000_zero(self):
        assert composite_score(**_base(tick_rate=2000.0)) == 0

    def test_above_band_zero(self):
        assert composite_score(**_base(tick_rate=2500.0)) == 0

    def test_none_tick_rate_zero(self):
        assert composite_score(**_base(tick_rate=None)) == 0


# ===================================================================
# 5. Test count (entry_count)
# ===================================================================


class TestEntryCount:
    def test_count_1_minus1(self):
        assert composite_score(**_base(entry_count=1)) == -1

    def test_count_2_plus1(self):
        assert composite_score(**_base(entry_count=2)) == 1

    def test_count_3_minus1(self):
        assert composite_score(**_base(entry_count=3)) == -1

    def test_count_4_zero(self):
        assert composite_score(**_base(entry_count=4)) == 0

    def test_count_5_plus1(self):
        assert composite_score(**_base(entry_count=5)) == 1

    def test_count_6_zero(self):
        assert composite_score(**_base(entry_count=6)) == 0

    def test_count_10_zero(self):
        assert composite_score(**_base(entry_count=10)) == 0


# ===================================================================
# 6. Session move
# ===================================================================


class TestSessionMove:
    def test_strongly_red_minus50_plus1(self):
        assert composite_score(**_base(session_move_pts=-50.0)) == 1

    def test_strongly_red_minus100_plus1(self):
        assert composite_score(**_base(session_move_pts=-100.0)) == 1

    def test_mildly_red_minus49_minus1(self):
        assert composite_score(**_base(session_move_pts=-49.0)) == -1

    def test_mildly_red_minus1_minus1(self):
        assert composite_score(**_base(session_move_pts=-1.0)) == -1

    def test_mildly_red_zero_minus1(self):
        assert composite_score(**_base(session_move_pts=0.0)) == -1

    def test_mildly_green_1_minus3(self):
        assert composite_score(**_base(session_move_pts=1.0)) == -3

    def test_mildly_green_50_minus3(self):
        assert composite_score(**_base(session_move_pts=50.0)) == -3

    def test_strongly_green_51_plus1(self):
        assert composite_score(**_base(session_move_pts=51.0)) == 1

    def test_strongly_green_100_plus1(self):
        assert composite_score(**_base(session_move_pts=100.0)) == 1

    def test_none_session_move_zero(self):
        assert composite_score(**_base(session_move_pts=None)) == 0


# ===================================================================
# 7. Streak
# ===================================================================


class TestStreak:
    def test_2_wins_plus3(self):
        assert composite_score(**_base(consecutive_wins=2)) == 3

    def test_5_wins_plus3(self):
        assert composite_score(**_base(consecutive_wins=5)) == 3

    def test_1_win_zero(self):
        assert composite_score(**_base(consecutive_wins=1)) == 0

    def test_2_losses_minus4(self):
        assert composite_score(**_base(consecutive_losses=2)) == -4

    def test_5_losses_minus4(self):
        assert composite_score(**_base(consecutive_losses=5)) == -4

    def test_1_loss_zero(self):
        assert composite_score(**_base(consecutive_losses=1)) == 0

    def test_no_streak_zero(self):
        assert composite_score(**_base(consecutive_wins=0, consecutive_losses=0)) == 0

    def test_wins_take_precedence_over_losses(self):
        # If both are >= 2 (unusual but possible), wins branch hits first
        result = composite_score(**_base(consecutive_wins=2, consecutive_losses=2))
        assert result == 3  # wins branch fires, losses branch is elif


# ===================================================================
# 8. score_tier
# ===================================================================


class TestScoreTier:
    def test_score_7_elite(self):
        label, wr = score_tier(7)
        assert label == "Elite"
        assert wr == "~88%"

    def test_score_10_elite(self):
        label, wr = score_tier(10)
        assert label == "Elite"

    def test_score_6_strong(self):
        label, wr = score_tier(6)
        assert label == "Strong"
        assert wr == "~85%"

    def test_score_5_good(self):
        label, wr = score_tier(5)
        assert label == "Good"
        assert wr == "~85%"

    def test_score_4_good(self):
        # Below MIN_SCORE still returns a tier (caller decides suppression)
        label, _ = score_tier(4)
        assert label == "Good"

    def test_score_0_good(self):
        label, _ = score_tier(0)
        assert label == "Good"

    def test_score_negative_good(self):
        label, _ = score_tier(-5)
        assert label == "Good"


# ===================================================================
# 9. Constants
# ===================================================================


class TestConstants:
    def test_min_score(self):
        assert MIN_SCORE == 5

    def test_tier_labels_keys(self):
        assert set(TIER_LABELS.keys()) == {5, 6, 7}


# ===================================================================
# 10. Combined / integration scenarios
# ===================================================================


class TestCombined:
    def test_best_case_scenario(self):
        """Stack every positive factor."""
        score = composite_score(
            level_name="FIB_EXT_HI_1.272",  # +2
            direction="up",  # +2 (strong combo)
            entry_count=2,  # +1
            now_et=datetime.time(15, 30),  # +2 (power hour)
            tick_rate=1800.0,  # +2
            session_move_pts=-80.0,  # +1 (strongly red)
            consecutive_wins=3,  # +3
        )
        assert score == 2 + 2 + 1 + 2 + 2 + 1 + 3  # 13
        label, _ = score_tier(score)
        assert label == "Elite"

    def test_worst_case_scenario(self):
        """Stack every negative factor."""
        score = composite_score(
            level_name="IBH",  # -1
            direction="up",  # -1 (weak combo)
            entry_count=1,  # -1
            now_et=datetime.time(10, 30),  # 0
            tick_rate=1000.0,  # 0
            session_move_pts=25.0,  # -3 (mildly green)
            consecutive_losses=3,  # -4
        )
        assert score == -1 + (-1) + (-1) + 0 + 0 + (-3) + (-4)  # -10
        label, _ = score_tier(score)
        assert label == "Good"

    def test_mixed_scenario_above_threshold(self):
        """A realistic above-threshold scenario."""
        score = composite_score(
            level_name="FIB_EXT_HI_1.272",  # +2
            direction="up",  # +2
            entry_count=5,  # +1
            now_et=datetime.time(15, 0),  # +2
            tick_rate=1500.0,  # 0
            session_move_pts=-60.0,  # +1
            consecutive_wins=0,
            consecutive_losses=0,
        )
        assert score == 8
        assert score >= MIN_SCORE

    def test_mixed_scenario_below_threshold(self):
        """A realistic below-threshold scenario."""
        score = composite_score(
            level_name="VWAP",  # -1
            direction="down",  # -1
            entry_count=1,  # -1
            now_et=datetime.time(11, 0),  # 0
            tick_rate=1500.0,  # 0
            session_move_pts=30.0,  # -3
            consecutive_losses=0,
        )
        assert score == -6
        assert score < MIN_SCORE

    def test_all_none_optionals(self):
        """All optional params are None/default."""
        score = composite_score(
            level_name="UNKNOWN",
            entry_count=4,
            now_et=None,
            tick_rate=None,
            session_move_pts=None,
        )
        assert score == 0

    def test_boundary_exactly_min_score(self):
        """Construct a scenario that scores exactly MIN_SCORE (5)."""
        score = composite_score(
            level_name="FIB_EXT_HI_1.272",  # +2
            direction="up",  # +2
            entry_count=2,  # +1
            now_et=datetime.time(12, 0),  # 0
            tick_rate=None,  # 0
            session_move_pts=None,  # 0
        )
        assert score == 5
        assert score == MIN_SCORE
