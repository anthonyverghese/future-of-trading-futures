"""
alert_manager.py — Tracks per-level alert state to prevent notification spam.

Rule: alert once when price enters the zone (within ALERT_THRESHOLD_POINTS).
Stay silent while price remains in the zone. Reset when price exits, so the
next entry triggers a fresh alert.

Composite scoring (206-day backtest, +8 target / -20 stop, all 7 factors):
  Score ≥ 5 → ~85% win rate at ~4.9 alerts/day (1,004 samples)
  Score ≥ 6 → ~85% win rate at ~3.2 alerts/day (654 samples)
  Score ≥ 7 → ~88% win rate at ~1.4 alerts/day (297 samples)
  Minimum cutoff = 5; alerts below are suppressed.
  Split-half stable: 86.4% / 84.1% on each half at ≥5.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from cache import log_alert
from config import ALERT_EXIT_POINTS, ALERT_THRESHOLD_POINTS
from notifications import send_notification

_TICKER = "MNQ"

# Minimum composite score to fire a notification.
# Score ≥5 → ~85% win rate at ~4.9 alerts/day (206-day backtest, 1,004 samples).
_MIN_SCORE = 5


# Signal strength tiers shown in notifications (206-day backtest, data-driven weights).
_TIER_LABELS: dict[int, tuple[str, str]] = {
    # score_min: (tier_label, backtest_win_rate for this bucket)
    5: ("Good", "~85%"),  # score 5
    6: ("Strong", "~85%"),  # score 6
    7: ("Elite", "~88%"),  # score 7+
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
    direction: str | None = None,
    consecutive_wins: int = 0,
    consecutive_losses: int = 0,
) -> int:
    """Compute composite alert quality score (206-day backtest, data-driven weights).

    Components (each contributes an integer score):
      Level:           FIB_EXT_HI +2, IBL/FIB_EXT_LO +1, VWAP/IBH -1
      Direction×Level: strong combos +1/+2, weak combos -1
      Time of day:     power hour +2, all others 0
      Tick rate:       1750-2000 +2, all others 0
      Test count:      #2/#5 +1, #1/#3 -1
      Session move:    strongly red/green +1, mildly red -1
      Streak:          2+ wins +3, 2+ losses -4
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


def _score_tier(score: int) -> tuple[str, str]:
    """Return (tier_label, backtest_win_rate) for a given composite score."""
    if score >= 7:
        return _TIER_LABELS[7]
    elif score >= 6:
        return _TIER_LABELS[6]
    return _TIER_LABELS[5]


class AlertManager:
    """Coordinates alert state across IBH, IBL, VWAP, and Fib levels for one session."""

    def __init__(self):
        self._levels: dict[str, LevelState] = {}

    def update_levels(
        self, ibh: float | None, ibl: float | None, vwap: float | None
    ) -> None:
        """Register or update price levels. Pass None to skip a level.

        IBH is excluded — 206-day backtest shows IBH at 71.9% WR (score -1),
        and IBH×up at 70.1% (score -1). Rarely clears ≥5 threshold.
        """
        for name, price in {"IBL": ibl, "VWAP": vwap}.items():
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
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
        trade_ts: datetime.datetime | None = None,
    ) -> list[tuple[int, str, float, str]]:
        """
        Fire a notification for any level whose zone is newly entered.

        All filtering is done via composite score (data-driven weights):
          - Score < 5 → suppressed
          - Score 5 → Good (~85%)
          - Score 6 → Strong (~85%)
          - Score 7+ → Elite (~88%)

        Returns (fired, all_zone_entries) where:
          fired: list of (alert_id, line_name, line_price, direction) for notified alerts
          all_zone_entries: list of (line_name, line_price, direction) for ALL zone
            entries (including suppressed), so the caller can track outcomes for
            streak computation — matching what the backtest does.
        """
        fired: list[tuple[int, str, float, str]] = []
        all_zone_entries: list[tuple[str, float, str]] = []
        for level in self._levels.values():
            if level.update(current_price):
                # Defensive: never notify if price drifted beyond threshold
                # (shouldn't happen, but guards against stale-state edge cases).
                if abs(current_price - level.price) > ALERT_THRESHOLD_POINTS:
                    continue
                direction = "up" if current_price > level.price else "down"

                # Track every zone entry for streak computation.
                all_zone_entries.append((level.name, level.price, direction))

                score = _composite_score(
                    level.name,
                    level.entry_count,
                    now_et,
                    tick_rate,
                    session_move_pts,
                    direction=direction,
                    consecutive_wins=consecutive_wins,
                    consecutive_losses=consecutive_losses,
                )

                if score < _MIN_SCORE:
                    print(
                        f"[ALERT suppressed — low score ({score})] {level.name} zone entered "
                        f"(test #{level.entry_count}), price {current_price:.2f}"
                    )
                    continue
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
                alert_id = log_alert(
                    ticker=_TICKER,
                    line=level.name,
                    line_price=level.price,
                    current_price=current_price,
                    direction=direction,
                    trade_ts=trade_ts,
                )
                send_notification(title, body)
                fired.append((alert_id, level.name, level.price, direction))
        return fired, all_zone_entries


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
        return "late morning"
    elif mins < 13 * 60:
        return "lunch"
    elif mins < 15 * 60:
        return "afternoon"
    else:
        return "power hour"  # 78.6% WR, +2 score


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

    dot = "🟢" if action == "BUY" else "🔴"

    title = f"{dot} {action} {dot} MNQ @ {current_price:.2f}"
    body = (
        f"{level_name} @ {level_price:.2f} | {tier_label} ({tier_wr})\n"
        f"{_ordinal(entry_count - 1)} retest, {time_label}"
    )
    return title, body
