"""
outcome_tracker.py — Evaluates whether alert recommendations were correct.

A recommendation is correct if, within 15 minutes of price hitting the line,
price moves 8 points in the recommended direction before moving 20 points
against (stop loss).
A recommendation is incorrect if price hits the stop first, or fails to reach
the +8 target within 15 minutes.
A recommendation is inconclusive if price never reaches the line within
15 minutes of the alert triggering.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from cache import update_alert_hit, update_alert_outcome

HIT_THRESHOLD = 1.0  # points — price within this distance = "hit the line"
MOVE_POINTS = 8.0  # points price must move in recommended direction (target)
STOP_POINTS = 20.0  # points against — stopped out before target = incorrect
EVAL_WINDOW_MINS = 15  # minutes after hitting line to evaluate outcome


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
    """

    def __init__(self, prior_outcomes: list[str] | None = None) -> None:
        self._pending: list[_PendingEval] = []
        # chronological: "correct"/"incorrect" — seeded from DB for cross-session streaks
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

    def restore(self, pending_alerts: list[dict], now: datetime.datetime) -> None:
        """Re-populate pending evaluations from DB rows after a restart.

        Alerts whose evaluation window has already expired are resolved
        immediately rather than resumed:
          - No hit + 15 min past alert_time → inconclusive
          - Hit + 15 min past hit_time → incorrect (target not reached)
        """
        resumed = 0
        expired = 0
        for a in pending_alerts:
            alert_time = a["alert_time"]
            hit_time = a.get("hit_time")
            alert_id = a["alert_id"]
            date_str = a["date_str"]

            if hit_time is None:
                # Waiting for line hit — check if alert window expired.
                if (now - alert_time).total_seconds() / 60 >= EVAL_WINDOW_MINS:
                    update_alert_outcome(alert_id, "inconclusive", date_str)
                    expired += 1
                    continue
            else:
                # Line was hit — check if move window expired.
                if (now - hit_time).total_seconds() / 60 >= EVAL_WINDOW_MINS:
                    update_alert_outcome(alert_id, "incorrect", date_str)
                    self._recent_outcomes.append("incorrect")
                    expired += 1
                    continue

            # Still within window — resume tracking.
            self._pending.append(
                _PendingEval(
                    alert_id=alert_id,
                    line_price=a["line_price"],
                    direction=a["direction"],
                    alert_time=alert_time,
                    date_str=date_str,
                    hit_time=hit_time,
                )
            )
            resumed += 1

        if resumed or expired:
            print(
                f"[outcome] Restored {resumed} pending evaluation(s), "
                f"resolved {expired} expired."
            )

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

        for ev in self._pending:
            if ev.hit_time is None:
                if abs(current_price - ev.line_price) <= HIT_THRESHOLD:
                    # Price touched the line — start the move evaluation window.
                    ev.hit_time = current_time
                    update_alert_hit(ev.alert_id, current_time.isoformat())
                elif (
                    current_time - ev.alert_time
                ).total_seconds() / 60 >= EVAL_WINDOW_MINS:
                    # Price never reached the line within 15 minutes — inconclusive.
                    update_alert_outcome(ev.alert_id, "inconclusive", ev.date_str)
                    resolved.append(ev)
            else:
                elapsed_mins = (current_time - ev.hit_time).total_seconds() / 60

                if ev.direction == "up":
                    target_hit = current_price >= ev.line_price + MOVE_POINTS
                    stop_hit = current_price <= ev.line_price - STOP_POINTS
                else:
                    target_hit = current_price <= ev.line_price - MOVE_POINTS
                    stop_hit = current_price >= ev.line_price + STOP_POINTS

                if target_hit:
                    update_alert_outcome(ev.alert_id, "correct", ev.date_str)
                    self._recent_outcomes.append("correct")
                    resolved.append(ev)
                elif stop_hit:
                    update_alert_outcome(ev.alert_id, "incorrect", ev.date_str)
                    self._recent_outcomes.append("incorrect")
                    resolved.append(ev)
                elif elapsed_mins >= EVAL_WINDOW_MINS:
                    update_alert_outcome(ev.alert_id, "incorrect", ev.date_str)
                    self._recent_outcomes.append("incorrect")
                    resolved.append(ev)

        for ev in resolved:
            self._pending.remove(ev)

    def close_session(self) -> None:
        """Mark all still-pending evaluations as 'inconclusive' at session end."""
        for ev in self._pending:
            update_alert_outcome(ev.alert_id, "inconclusive", ev.date_str)
        self._pending.clear()
