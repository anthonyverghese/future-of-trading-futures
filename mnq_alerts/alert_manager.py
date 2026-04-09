"""
alert_manager.py — Tracks per-level alert state to prevent notification spam.

Rule: alert once when price enters the zone (within ALERT_THRESHOLD_POINTS).
Stay silent while price remains in the zone. Reset when price exits, so the
next entry triggers a fresh alert.

Scoring logic lives in scoring.py; this module handles the zone state machine,
notification dispatch, and message formatting.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from typing import Callable

from config import ALERT_EXIT_POINTS, ALERT_THRESHOLD_POINTS
from scoring import MIN_SCORE, composite_score, score_tier

_ZONE_STATE_FILE = os.path.join(os.path.dirname(__file__), ".zone_state.json")

_TICKER = "MNQ"

# Type aliases for injectable dependencies.
LogAlertFn = Callable[..., int]
SendNotificationFn = Callable[[str, str], bool]


@dataclass
class LevelState:
    """Alert state for a single price level (IBH, IBL, Fib, or VWAP).

    For fixed levels (IBH/IBL/Fib): exit check uses the locked reference_price.
    For drifting levels (VWAP): exit check uses the current level price so the
    zone tracks VWAP as it moves, preventing rapid re-triggering when VWAP drifts
    toward price after a zone exit.
    """

    name: str
    price: float
    in_zone: bool = False
    reference_price: float | None = (
        None  # level price locked at zone entry; used for exit check
    )
    entry_count: int = 0  # cumulative zone entries this session
    drifts: bool = False  # True for VWAP — exit checks current price, not locked ref

    def update(self, current_price: float) -> bool:
        """Returns True if an alert should fire (price just entered the zone).

        On zone entry, reference_price is locked to the current level price.
        Exit requires price to move ALERT_EXIT_POINTS (20) away from the
        reference (fixed levels) or current level price (drifting levels like VWAP).
        """
        if self.in_zone:
            exit_ref = self.price if self.drifts else self.reference_price
            if abs(current_price - exit_ref) > ALERT_EXIT_POINTS:
                self.in_zone = False
                self.reference_price = None
            return False

        if abs(current_price - self.price) <= ALERT_THRESHOLD_POINTS:
            self.in_zone = True
            self.reference_price = self.price
            self.entry_count += 1
            return True

        return False


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


def build_message(
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

    Title format:  🟢 BUY 🟢 MNQ @ 24602.50
    Body format:
        IBL @ 24595.00 | Strong (~85%)
        2nd retest, afternoon
    """
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


class AlertManager:
    """Coordinates alert state across IBH, IBL, VWAP, and Fib levels for one session.

    Accepts optional dependency injection for logging and notifications,
    defaulting to the production implementations (cache.log_alert and
    notifications.send_notification). Pass stubs for testing.
    """

    def __init__(
        self,
        log_fn: LogAlertFn | None = None,
        notify_fn: SendNotificationFn | None = None,
    ):
        self._levels: dict[str, LevelState] = {}
        if log_fn is not None:
            self._log_fn = log_fn
        else:
            from cache import log_alert

            self._log_fn = log_alert
        if notify_fn is not None:
            self._notify_fn = notify_fn
        else:
            from notifications import send_notification

            self._notify_fn = send_notification

    def save_zone_state(self) -> None:
        """Persist zone entry counts and in_zone state to survive restarts."""
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        state = {
            "date": today,
            "levels": {
                name: {
                    "entry_count": ls.entry_count,
                    "in_zone": ls.in_zone,
                    "reference_price": ls.reference_price,
                }
                for name, ls in self._levels.items()
            },
        }
        try:
            with open(_ZONE_STATE_FILE, "w") as f:
                json.dump(state, f)
        except OSError:
            pass

    def restore_zone_state(self) -> None:
        """Restore zone entry counts from disk (same-day only)."""
        try:
            with open(_ZONE_STATE_FILE) as f:
                state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        if state.get("date") != today:
            return
        restored = 0
        for name, saved in state.get("levels", {}).items():
            if name in self._levels:
                ls = self._levels[name]
                ls.entry_count = saved.get("entry_count", 0)
                ls.in_zone = saved.get("in_zone", False)
                ls.reference_price = saved.get("reference_price")
                restored += 1
        if restored:
            counts = ", ".join(
                f"{n}={ls.entry_count}"
                for n, ls in self._levels.items()
                if ls.entry_count > 0
            )
            print(f"[zone] Restored state for {restored} levels ({counts})")

    def update_levels(
        self, ibh: float | None, ibl: float | None, vwap: float | None
    ) -> None:
        """Register or update price levels. Pass None to skip a level."""
        for name, price in {"IBH": ibh, "IBL": ibl, "VWAP": vwap}.items():
            if price is None:
                continue
            if name not in self._levels:
                self._levels[name] = LevelState(
                    name=name, price=price, drifts=(name == "VWAP")
                )
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
        range_30m: float | None = None,
    ) -> tuple[list[tuple[int, str, float, str]], list[tuple[str, float, str]]]:
        """
        Fire a notification for any level whose zone is newly entered.

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

                score, bd = composite_score(
                    level.name,
                    level.entry_count,
                    now_et,
                    tick_rate,
                    session_move_pts,
                    direction=direction,
                    consecutive_wins=consecutive_wins,
                    consecutive_losses=consecutive_losses,
                    range_30m=range_30m,
                    breakdown=True,
                )

                if score < MIN_SCORE:
                    print(
                        f"[ALERT suppressed — score {score}] {level.name} zone entered "
                        f"(test #{level.entry_count}), price {current_price:.2f} | {bd}"
                    )
                    continue
                tier_label, tier_wr = score_tier(score)
                title, body = build_message(
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
                alert_id = self._log_fn(
                    ticker=_TICKER,
                    line=level.name,
                    line_price=level.price,
                    current_price=current_price,
                    direction=direction,
                    trade_ts=trade_ts,
                )
                self._notify_fn(title, body)
                fired.append((alert_id, level.name, level.price, direction))
        if all_zone_entries:
            self.save_zone_state()
        return fired, all_zone_entries
