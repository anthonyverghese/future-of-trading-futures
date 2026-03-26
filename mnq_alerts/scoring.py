"""
scoring.py — Composite alert quality scoring (data-driven weights from 206-day backtest).

Separated from alert_manager.py so scoring logic can be tested, tuned, and
backtested independently of the zone state machine and notification plumbing.

Components (each contributes an integer score):
  Level:           FIB_EXT_HI +2, IBL/FIB_EXT_LO +1, VWAP/IBH -1
  Direction×Level: strong combos +1/+2, weak combos -1
  Time of day:     power hour +2, all others 0
  Tick rate:       1750-2000 +2, all others 0
  Test count:      #2/#5 +1, #1/#3 -1
  Session move:    strongly red/green +1, mildly red -1, mildly green -3
  Streak:          2+ wins +3, 2+ losses -4
"""

from __future__ import annotations

import datetime

# Minimum composite score to fire a notification.
# Score ≥5 → ~85% win rate at ~4.9 alerts/day (206-day backtest, 1,004 samples).
MIN_SCORE = 5

# Signal strength tiers shown in notifications (206-day backtest, data-driven weights).
TIER_LABELS: dict[int, tuple[str, str]] = {
    # score_min: (tier_label, backtest_win_rate for this bucket)
    5: ("Good", "~85%"),  # score 5
    6: ("Strong", "~85%"),  # score 6
    7: ("Elite", "~88%"),  # score 7+
}


def composite_score(
    level_name: str,
    entry_count: int,
    now_et: datetime.time | None,
    tick_rate: float | None,
    session_move_pts: float | None,
    direction: str | None = None,
    consecutive_wins: int = 0,
    consecutive_losses: int = 0,
) -> int:
    """Compute composite alert quality score (206-day backtest, data-driven weights).

    Returns an integer score; alerts with score < MIN_SCORE should be suppressed.
    """
    s = 0

    # Level quality (206-day WR: FIB_HI 79.0%, FIB_LO 78.0%, IBL 76.8%,
    #                              VWAP 72.6%, IBH 71.9%)
    if level_name == "FIB_EXT_HI_1.272":
        s += 2
    elif level_name in ("IBL", "FIB_EXT_LO_1.272"):
        s += 1
    elif level_name in ("VWAP", "IBH"):
        s -= 1

    # Direction × Level interaction (206-day backtest)
    if direction is not None:
        combo = (level_name, direction)
        # Strong combos
        if combo == ("FIB_EXT_HI_1.272", "up"):  # 83.9% (174 trades)
            s += 2
        elif combo in (
            ("FIB_EXT_LO_1.272", "down"),  # 80.2% (283 trades)
            ("IBL", "down"),  # 80.1% (321 trades)
            ("VWAP", "up"),  # 74.7% (704 trades)
        ):
            s += 1
        # Weak combos
        elif combo in (
            ("IBH", "up"),  # 70.1% (284 trades)
            ("IBL", "up"),  # 74.4% (433 trades)
            ("FIB_EXT_LO_1.272", "up"),  # 76.2% (349 trades)
            ("FIB_EXT_HI_1.272", "down"),  # 76.2% (298 trades)
            ("VWAP", "down"),  # 70.4% (673 trades)
        ):
            s -= 1

    # Time of day (only power hour is meaningfully above baseline)
    if now_et is not None:
        mins = now_et.hour * 60 + now_et.minute
        if mins >= 15 * 60:
            s += 2  # power hour: 78.6% (529 trades)

    # Tick rate (only the 1750-2000 band shows signal: 79.9%)
    if tick_rate is not None:
        if 1750 <= tick_rate < 2000:
            s += 2

    # Test count (206-day WR: #2 77.1%, #5 77.0%, #4 75.9%, #6+ 74.8%,
    #                          #3 72.8%, #1 72.5%)
    if entry_count == 1:
        s -= 1
    elif entry_count == 2:
        s += 1
    elif entry_count == 3:
        s -= 1
    elif entry_count == 5:
        s += 1

    # Session context (206-day WR: strongly red 76.2%, strongly green 76.7%,
    #                                mildly red 73.1%, mildly green 66.9%)
    if session_move_pts is not None:
        if session_move_pts <= -50:
            s += 1  # strongly red
        elif -50 < session_move_pts <= 0:
            s -= 1  # mildly red
        elif session_move_pts > 50:
            s += 1  # strongly green
        else:
            s -= 3  # mildly green: 66.9% — worst bucket

    # Outcome streak (206-day: 2+W → 81.3%, 2+L → 56.9%, mixed → 68.0%)
    if consecutive_wins >= 2:
        s += 3
    elif consecutive_losses >= 2:
        s -= 4

    return s


def score_tier(score: int) -> tuple[str, str]:
    """Return (tier_label, backtest_win_rate) for a given composite score."""
    if score >= 7:
        return TIER_LABELS[7]
    elif score >= 6:
        return TIER_LABELS[6]
    return TIER_LABELS[5]
