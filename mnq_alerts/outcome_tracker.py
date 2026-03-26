"""
outcome_tracker.py — Evaluates whether alert recommendations were correct.

A recommendation is correct if, within EVAL_WINDOW_MINS minutes of price
hitting the line, price moves TARGET_POINTS in the recommended direction
before moving STOP_POINTS against (stop loss).
A recommendation is incorrect if price hits the stop first, or fails to reach
the target within the evaluation window.
A recommendation is inconclusive if price never reaches the line within
EVAL_WINDOW_MINS minutes of the alert triggering.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Callable

from config import EVAL_WINDOW_MINS, HIT_THRESHOLD, STOP_POINTS, TARGET_POINTS

# Persists recent outcomes (tracked + untracked) so streaks survive restarts.
_STREAK_FILE = os.path.join(os.path.dirname(__file__), "streak_outcomes.json")
_STREAK_LIMIT = 20  # keep the last N outcomes

# Type aliases for injectable DB callbacks.
UpdateHitFn = Callable[[int, str], None]
UpdateOutcomeFn = Callable[[int, str, str], None]


def _noop_hit(alert_id: int, hit_time: str) -> None:
    pass


def _noop_outcome(alert_id: int, outcome: str, date_str: str) -> None:
    pass


@dataclass
class _PendingEval:
    alert_id: int
    line_price: float
    direction: str  # 'up' or 'down'
    alert_time: datetime.datetime
    date_str: str
    hit_time: datetime.datetime | None = field(default=None)


class OutcomeEvaluator:
    """
    Tracks pending alert outcomes. Call update() on every live trade tick.
    Call close_session() when RTH ends to mark remaining alerts 'unresolved'.

    Accepts optional dependency injection for DB callbacks (on_hit_fn,
    on_outcome_fn). Defaults to the production implementations from cache.py.
    Pass stubs/noops for testing.
    """

    def __init__(
        self,
        prior_outcomes: list[str] | None = None,
        on_hit_fn: UpdateHitFn | None = None,
        on_outcome_fn: UpdateOutcomeFn | None = None,
    ) -> None:
        self._pending: list[_PendingEval] = []

        # DB callbacks — lazy-import production defaults.
        if on_hit_fn is not None:
            self._on_hit = on_hit_fn
        else:
            from cache import update_alert_hit

            self._on_hit = update_alert_hit
        if on_outcome_fn is not None:
            self._on_outcome = on_outcome_fn
        else:
            from cache import update_alert_outcome

            self._on_outcome = update_alert_outcome

        # Load streaks from file (includes untracked outcomes), falling back
        # to DB-only outcomes passed in by the caller.
        saved = self._load_streak_file()
        if saved is not None:
            self._recent_outcomes: list[str] = saved
        else:
            self._recent_outcomes: list[str] = (
                list(prior_outcomes) if prior_outcomes else []
            )

    def add(
        self,
        alert_id: int,
        line_price: float,
        direction: str,
        alert_time: datetime.datetime,
        date_str: str,
    ) -> None:
        self._pending.append(
            _PendingEval(alert_id, line_price, direction, alert_time, date_str)
        )

    def add_untracked(
        self,
        line_price: float,
        direction: str,
        alert_time: datetime.datetime,
    ) -> None:
        """Track a suppressed zone entry for streak computation only.

        These don't get logged to the DB or sent as notifications, but their
        outcomes feed into consecutive_wins/losses — matching how the backtest
        computes streaks across ALL zone entries, not just filtered ones.
        """
        self._pending.append(
            _PendingEval(
                alert_id=-1,  # sentinel: not in DB
                line_price=line_price,
                direction=direction,
                alert_time=alert_time,
                date_str="",
            )
        )

    def restore(self, pending_alerts: list[dict]) -> None:
        """Re-populate pending evaluations from DB rows after a restart."""
        for a in pending_alerts:
            self._pending.append(
                _PendingEval(
                    alert_id=a["alert_id"],
                    line_price=a["line_price"],
                    direction=a["direction"],
                    alert_time=a["alert_time"],
                    date_str=a["date_str"],
                    hit_time=a.get("hit_time"),
                )
            )
        if pending_alerts:
            print(
                f"[outcome] Restored {len(pending_alerts)} pending "
                f"evaluation(s) from DB."
            )

    @staticmethod
    def _load_streak_file() -> list[str] | None:
        """Load recent outcomes from streak file, or None if not found."""
        try:
            with open(_STREAK_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data[-_STREAK_LIMIT:]
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            pass
        return None

    def _save_streak_file(self) -> None:
        """Persist recent outcomes to disk so streaks survive restarts."""
        try:
            with open(_STREAK_FILE, "w") as f:
                json.dump(self._recent_outcomes[-_STREAK_LIMIT:], f)
        except OSError:
            pass

    def _record_outcome(self, outcome: str) -> None:
        """Append outcome and persist to disk."""
        self._recent_outcomes.append(outcome)
        self._save_streak_file()

    @property
    def consecutive_wins(self) -> int:
        """Count of consecutive 'correct' outcomes at the tail of the history."""
        count = 0
        for outcome in reversed(self._recent_outcomes):
            if outcome == "correct":
                count += 1
            else:
                break
        return count

    @property
    def consecutive_losses(self) -> int:
        """Count of consecutive 'incorrect' outcomes at the tail of the history."""
        count = 0
        for outcome in reversed(self._recent_outcomes):
            if outcome == "incorrect":
                count += 1
            else:
                break
        return count

    def update(self, current_price: float, current_time: datetime.datetime) -> None:
        """Process one trade tick against all pending evaluations."""
        resolved: list[_PendingEval] = []
        tracked = lambda ev: ev.alert_id != -1  # noqa: E731

        for ev in self._pending:
            if ev.hit_time is None:
                if abs(current_price - ev.line_price) <= HIT_THRESHOLD:
                    # Price touched the line — start the move evaluation window.
                    ev.hit_time = current_time
                    if tracked(ev):
                        self._on_hit(ev.alert_id, current_time.isoformat())
                elif (
                    current_time - ev.alert_time
                ).total_seconds() / 60 >= EVAL_WINDOW_MINS:
                    # Price never reached the line within the window — inconclusive.
                    if tracked(ev):
                        self._on_outcome(ev.alert_id, "inconclusive", ev.date_str)
                    resolved.append(ev)
            else:
                elapsed_mins = (current_time - ev.hit_time).total_seconds() / 60

                if ev.direction == "up":
                    target_hit = current_price >= ev.line_price + TARGET_POINTS
                    stop_hit = current_price <= ev.line_price - STOP_POINTS
                else:
                    target_hit = current_price <= ev.line_price - TARGET_POINTS
                    stop_hit = current_price >= ev.line_price + STOP_POINTS

                if target_hit:
                    if tracked(ev):
                        self._on_outcome(ev.alert_id, "correct", ev.date_str)
                    self._record_outcome("correct")
                    resolved.append(ev)
                elif stop_hit:
                    if tracked(ev):
                        self._on_outcome(ev.alert_id, "incorrect", ev.date_str)
                    self._record_outcome("incorrect")
                    resolved.append(ev)
                elif elapsed_mins >= EVAL_WINDOW_MINS:
                    if tracked(ev):
                        self._on_outcome(ev.alert_id, "incorrect", ev.date_str)
                    self._record_outcome("incorrect")
                    resolved.append(ev)

        for ev in resolved:
            self._pending.remove(ev)

    def close_session(self) -> None:
        """Mark all still-pending evaluations as 'inconclusive' at session end."""
        for ev in self._pending:
            if ev.alert_id != -1:
                self._on_outcome(ev.alert_id, "inconclusive", ev.date_str)
        self._pending.clear()
