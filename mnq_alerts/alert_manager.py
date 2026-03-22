"""
alert_manager.py — Tracks per-level alert state to prevent notification spam.

Rule: alert once when price enters the zone (within ALERT_THRESHOLD_POINTS).
Stay silent while price remains in the zone. Reset when price exits, so the
next entry triggers a fresh alert.

Composite scoring (180-day backtest, +8 target / -20 stop):
  Score ≥ 3 → 79.5% win rate (410 trades)
  Score ≥ 4 → 81.3% win rate (252 trades)
  Score ≥ 5 → 80.5% win rate (133 trades)
  Minimum cutoff = 3; alerts below are suppressed.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from cache import log_alert
from config import ALERT_EXIT_POINTS, ALERT_THRESHOLD_POINTS
from notifications import send_notification

_TICKER = "MNQ"

# Minimum composite score to fire a notification.
# 180-day backtest: ≥ 3 → 79.5% (410 trades), ≥ 4 → 81.3% (252 trades).
_MIN_SCORE = 3

# Signal strength tiers shown in notifications (180-day backtest).
_TIER_LABELS: dict[int, tuple[str, str]] = {
    # score_min: (tier_label, backtest_win_rate for this bucket)
    3: ("Good", "76.6%"),  # score 3: 121W/37L
    4: ("Strong", "82.4%"),  # score 4: 98W/21L
    5: ("Elite", "82.8%"),  # score 5+: 107W/26L
}


@dataclass
class LevelState:
    """Alert state for a single price level (IBH, IBL, or VWAP)."""

    name: str
    price: float
    in_zone: bool = False
    reference_price: float | None = (
        None  # level price locked at zone entry; used for exit check
    )
    entry_count: int = 0  # cumulative zone entries this session

    def update(self, current_price: float) -> bool:
        """Returns True if an alert should fire (price just entered the zone).

        On zone entry, reference_price is locked to the current level price.
        Exit requires price to move ALERT_EXIT_POINTS (20) away from the
        reference — wider than the entry threshold (10) to reduce re-triggering.
        """
        if self.in_zone:
            if abs(current_price - self.reference_price) > ALERT_EXIT_POINTS:
                self.in_zone = False
                self.reference_price = None
            return False

        if abs(current_price - self.price) <= ALERT_THRESHOLD_POINTS:
            self.in_zone = True
            self.reference_price = self.price
            self.entry_count += 1
            return True

        return False


def _composite_score(
    level_name: str,
    entry_count: int,
    now_et: datetime.time | None,
    tick_rate: float | None,
    session_move_pts: float | None,
) -> int:
    """Compute composite alert quality score (90-day backtest-derived weights).

    Components (each contributes an integer score):
      Level:         IBL +3, VWAP 0, IBH -1
      Time of day:   afternoon +2, power hour +1, lunch -1, first hour -3
      Tick rate:     ≥2000 +2, ≥1750 +1, <1000 -2
      Test count:    #1 -4, #3 +2, #4 +1, #5+ -1
      Session move:  mildly red +2, strongly green -1
    """
    s = 0

    # Level quality
    if level_name == "IBL":
        s += 3
    elif level_name == "IBH":
        s -= 1
    elif level_name == "FIB_EXT_LO_1.272":
        s += 2  # 78.5% win rate, EV +2.0
    elif level_name in ("FIB_RET_0.236", "FIB_EXT_HI_1.272"):
        s += 1  # 77.8% win rate, EV +1.8

    # Time of day
    if now_et is not None:
        mins = now_et.hour * 60 + now_et.minute
        if (13 * 60) <= mins < (15 * 60):
            s += 2  # afternoon
        elif (10 * 60 + 30) <= mins < (11 * 60 + 30):
            s -= 3  # first hour post-IB
        elif (11 * 60 + 30) <= mins < (13 * 60):
            s -= 1  # lunch
        else:
            s += 1  # power hour

    # Tick rate
    if tick_rate is not None:
        if tick_rate >= 2000:
            s += 2
        elif tick_rate >= 1750:
            s += 1
        elif tick_rate < 1000:
            s -= 2

    # Test count
    if entry_count == 1:
        s -= 4  # first test
    elif entry_count == 3:
        s += 2
    elif entry_count == 4:
        s += 1
    elif entry_count >= 5:
        s -= 1

    # Session context
    if session_move_pts is not None:
        if -50 < session_move_pts <= 0:
            s += 2  # mildly red
        elif session_move_pts > 50:
            s -= 1  # strongly green

    return s


def _score_tier(score: int) -> tuple[str, str]:
    """Return (tier_label, backtest_win_rate) for a given composite score."""
    if score >= 5:
        return _TIER_LABELS[5]
    elif score >= 4:
        return _TIER_LABELS[4]
    return _TIER_LABELS[3]


class AlertManager:
    """Coordinates alert state across IBH, IBL, VWAP, and Fib levels for one session."""

    def __init__(self):
        self._levels: dict[str, LevelState] = {}

    def update_levels(
        self, ibh: float | None, ibl: float | None, vwap: float | None
    ) -> None:
        """Register or update price levels. Pass None to skip a level."""
        for name, price in {"IBH": ibh, "IBL": ibl, "VWAP": vwap}.items():
            if price is None:
                continue
            if name not in self._levels:
                self._levels[name] = LevelState(name=name, price=price)
                print(f"[AlertManager] {name} registered at {price:.2f}")
            else:
                self._levels[name].price = price  # VWAP drifts; always update

    def update_fib_levels(self, fib_levels: dict[str, float]) -> None:
        """Register Fibonacci levels (fixed for the session, like IBH/IBL)."""
        for name, price in fib_levels.items():
            if name not in self._levels:
                self._levels[name] = LevelState(name=name, price=price)
                print(f"[AlertManager] {name} registered at {price:.2f}")

    def advance_state(self, current_price: float) -> None:
        """Update in_zone state for all levels without sending notifications.
        Use during historical replay to prime the state machine."""
        for level in self._levels.values():
            level.update(current_price)

    def check_and_notify(
        self,
        current_price: float,
        now_et: datetime.time | None = None,
        tick_rate: float | None = None,
        session_move_pts: float | None = None,
    ) -> list[tuple[int, str, float, str]]:
        """
        Fire a notification for any level whose zone is newly entered.

        All filtering is done via composite score (first-test and first-hour
        are penalized in the score rather than hard-blocked):
          - Score < 3 → suppressed
          - Score 3 → Good (76.6%)
          - Score 4 → Strong (82.4%)
          - Score 5+ → Elite (82.8%)

        Returns a list of (alert_id, line_name, line_price, direction) for each
        fired alert so the caller can register them with OutcomeEvaluator.
        """
        fired: list[tuple[int, str, float, str]] = []
        for level in self._levels.values():
            if level.update(current_price):
                score = _composite_score(
                    level.name,
                    level.entry_count,
                    now_et,
                    tick_rate,
                    session_move_pts,
                )

                if score < _MIN_SCORE:
                    print(
                        f"[ALERT suppressed — low score ({score})] {level.name} zone entered "
                        f"(test #{level.entry_count}), price {current_price:.2f}"
                    )
                    continue

                direction = "up" if current_price > level.price else "down"
                tier_label, tier_wr = _score_tier(score)
                title, body = _build_message(
                    level.name,
                    level.price,
                    current_price,
                    level.entry_count,
                    now_et,
                    score=score,
                    tier_label=tier_label,
                    tier_wr=tier_wr,
                )
                print(f"[ALERT] {title} | {body}")
                send_notification(title, body)
                alert_id = log_alert(
                    ticker=_TICKER,
                    line=level.name,
                    line_price=level.price,
                    current_price=current_price,
                    direction=direction,
                )
                fired.append((alert_id, level.name, level.price, direction))
        return fired


def _ordinal(n: int) -> str:
    """Return ordinal string: 2 → '2nd', 3 → '3rd', 4 → '4th', etc."""
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n if n <= 20 else n % 10, "th")
    return f"{n}{suffix}"


def _time_bucket(t: datetime.time | None) -> str:
    """Return the backtest time-of-day label for the given ET time."""
    if t is None:
        return "unknown"
    mins = t.hour * 60 + t.minute
    if mins < 11 * 60 + 30:
        return "late morning"  # 10:30–11:30 (score -3 penalty)
    elif mins < 13 * 60:
        return "lunch"  # 11:30–13:00, 70.9%
    elif mins < 15 * 60:
        return "afternoon"  # 13:00–15:00, 77.6%
    else:
        return "power hour"  # 15:00–16:00, 75.7%


def _build_message(
    level_name: str,
    level_price: float,
    current_price: float,
    entry_count: int,
    now_et: datetime.time | None = None,
    *,
    score: int = 0,
    tier_label: str = "",
    tier_wr: str = "",
) -> tuple[str, str]:
    """
    Build the notification title and body.

    Title format:  ↑ BUY (Strong) — IBL support bounce
    Body format:
        IBL @ 24595.00 | MNQ @ 24602.50 (5.25 pts above)
        Afternoon, 2nd retest | Score 5 — 88.3% win rate
    """
    distance = abs(current_price - level_price)
    above_line = current_price > level_price
    time_label = _time_bucket(now_et)

    # Setup name: concise description of the trade thesis.
    setups = {
        "IBH": ("BUY", "IBH support retest", "SELL", "IBH resistance fade"),
        "IBL": ("BUY", "IBL support bounce", "SELL", "IBL breakdown retest"),
        "VWAP": ("BUY", "VWAP support hold", "SELL", "VWAP resistance fade"),
        "FIB_RET_0.236": ("BUY", "Fib 23.6% support", "SELL", "Fib 23.6% breakdown"),
        "FIB_EXT_LO_1.272": (
            "BUY",
            "Fib 1.272 ext bounce",
            "SELL",
            "Fib 1.272 ext breakdown",
        ),
        "FIB_EXT_HI_1.272": (
            "BUY",
            "Fib 1.272 ext breakout",
            "SELL",
            "Fib 1.272 ext rejection",
        ),
    }
    if level_name in setups:
        buy_action, buy_setup, sell_action, sell_setup = setups[level_name]
        if above_line:
            action, setup_name = buy_action, buy_setup
        else:
            action, setup_name = sell_action, sell_setup
    else:
        action = "WATCH"
        setup_name = f"{level_name} test"

    arrow = "↑" if action == "BUY" else "↓"
    side_str = f"{distance:.1f} pts {'above' if above_line else 'below'}"

    title = f"{arrow} {action} ({tier_label}) — {setup_name}"
    body = (
        f"{level_name} @ {level_price:.2f} | MNQ @ {current_price:.2f} ({side_str})\n"
        f"{time_label.capitalize()}, {_ordinal(entry_count - 1)} retest | "
        f"Score {score} — {tier_wr} win rate"
    )
    return title, body
