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
  Session move:    sweet spots (10-20 green, -20 to -10 red) +2,
                   strongly red +1, near-zero green -3, rest 0
  Streak:          2+ wins +3, 2+ losses -4
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

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


@dataclass
class ScoreBreakdown:
    """Per-factor breakdown of a composite score."""

    level: int = 0
    combo: int = 0
    time: int = 0
    tick: int = 0
    test: int = 0
    move: int = 0
    streak: int = 0

    @property
    def total(self) -> int:
        return (
            self.level
            + self.combo
            + self.time
            + self.tick
            + self.test
            + self.move
            + self.streak
        )

    def __str__(self) -> str:
        parts = []
        for name in ("level", "combo", "time", "tick", "test", "move", "streak"):
            val = getattr(self, name)
            if val != 0:
                parts.append(f"{name}={val:+d}")
        return ", ".join(parts) if parts else "all zero"


def composite_score(
    level_name: str,
    entry_count: int,
    now_et: datetime.time | None,
    tick_rate: float | None,
    session_move_pts: float | None,
    direction: str | None = None,
    consecutive_wins: int = 0,
    consecutive_losses: int = 0,
    breakdown: bool = False,
) -> int | tuple[int, ScoreBreakdown]:
    """Compute composite alert quality score (206-day backtest, data-driven weights).

    Returns an integer score; alerts with score < MIN_SCORE should be suppressed.
    If breakdown=True, returns (score, ScoreBreakdown) for logging/debugging.
    """
    bd = ScoreBreakdown()

    # Level quality (206-day WR: FIB_HI 79.0%, FIB_LO 78.0%, IBL 76.8%,
    #                              VWAP 72.6%, IBH 71.9%)
    if level_name == "FIB_EXT_HI_1.272":
        bd.level = 2
    elif level_name in ("IBL", "FIB_EXT_LO_1.272"):
        bd.level = 1
    elif level_name in ("VWAP", "IBH"):
        bd.level = -1

    # Direction × Level interaction (206-day backtest)
    if direction is not None:
        c = (level_name, direction)
        # Strong combos
        if c == ("FIB_EXT_HI_1.272", "up"):  # 83.9% (174 trades)
            bd.combo = 2
        elif c in (
            ("FIB_EXT_LO_1.272", "down"),  # 80.2% (283 trades)
            ("IBL", "down"),  # 80.1% (321 trades)
            ("VWAP", "up"),  # 74.7% (704 trades)
        ):
            bd.combo = 1
        # Weak combos
        elif c in (
            ("IBH", "up"),  # 70.1% (284 trades)
            ("IBL", "up"),  # 74.4% (433 trades)
            ("FIB_EXT_LO_1.272", "up"),  # 76.2% (349 trades)
            ("FIB_EXT_HI_1.272", "down"),  # 76.2% (298 trades)
            ("VWAP", "down"),  # 70.4% (673 trades)
        ):
            bd.combo = -1

    # Time of day (only power hour is meaningfully above baseline)
    if now_et is not None:
        mins = now_et.hour * 60 + now_et.minute
        if mins >= 15 * 60:
            bd.time = 2  # power hour: 78.6% (529 trades)

    # Tick rate (only the 1750-2000 band shows signal: 79.9%)
    if tick_rate is not None:
        if 1750 <= tick_rate < 2000:
            bd.tick = 2

    # Test count (206-day WR: #2 77.1%, #5 77.0%, #4 75.9%, #6+ 74.8%,
    #                          #3 72.8%, #1 72.5%)
    if entry_count == 1:
        bd.test = -1
    elif entry_count == 2:
        bd.test = 1
    elif entry_count == 3:
        bd.test = -1
    elif entry_count == 5:
        bd.test = 1

    # Session context (214-day sub-bucket analysis):
    #   (-20,-10]: 80.6%, (-10,0]: 74.7%, (10,20]: 80.7%, (0,10]: 68.9%
    #   strongly red/green: ~75%, (20-50]: ~74%, rest: ~72%
    if session_move_pts is not None:
        if 10 < session_move_pts <= 20:
            bd.move = 2  # mildly green sweet spot: 80.7% WR
        elif -20 < session_move_pts <= -10:
            bd.move = 2  # mildly red sweet spot: 80.6% WR
        elif session_move_pts <= -50:
            bd.move = 1  # strongly red: 75.7%
        elif session_move_pts > 50:
            bd.move = 0  # strongly green: 73.1% — near baseline
        elif 0 < session_move_pts <= 10:
            bd.move = -3  # near-zero green: 68.9% — worst bucket
        else:
            bd.move = 0  # remaining buckets: ~74% — baseline

    # Outcome streak (206-day: 2+W → 81.3%, 2+L → 56.9%, mixed → 68.0%)
    if consecutive_wins >= 2:
        bd.streak = 3
    elif consecutive_losses >= 2:
        bd.streak = -4

    if breakdown:
        return bd.total, bd
    return bd.total


def score_tier(score: int) -> tuple[str, str]:
    """Return (tier_label, backtest_win_rate) for a given composite score."""
    if score >= 7:
        return TIER_LABELS[7]
    elif score >= 6:
        return TIER_LABELS[6]
    return TIER_LABELS[5]
