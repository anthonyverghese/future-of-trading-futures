"""Pluggable zone state machines.

Each zone class tracks whether price is "in the zone" for a single level.
Different strategies use different entry/exit logic.

All classes implement:
  update(price) → bool   (True = fresh zone entry)
  reset()                (force zone exit)
  in_zone: bool
  entry_count: int
"""

from __future__ import annotations


class HumanZone:
    """Human alert zone: 7-pt entry, 20-pt exit.

    Used by the human alert app. Zone enters when price is within 7 pts
    of the level, exits when price moves 20+ pts from the reference.
    """

    def __init__(self, price: float, drifts: bool = False):
        self.price = price
        self.drifts = drifts
        self.in_zone = False
        self.entry_count = 0
        self._ref: float | None = None

    def update(self, current_price: float) -> bool:
        if self.in_zone:
            ref = self.price if self.drifts else self._ref
            if ref is not None and abs(current_price - ref) > 20.0:
                self.in_zone = False
                self._ref = None
            return False
        if abs(current_price - self.price) <= 7.0:
            self.in_zone = True
            self._ref = self.price
            self.entry_count += 1
            return True
        return False

    def reset(self):
        self.in_zone = False
        self._ref = None


class BotZoneTradeReset:
    """Bot zone: 1-pt entry, resets when trade closes.

    Zone stays active while a trade is open. Does NOT exit on price
    distance. The caller must call reset() when the trade resolves.
    This is the current live bot logic (deployed 2026-04-19).
    """

    def __init__(self, price: float, drifts: bool = False):
        self.price = price
        self.drifts = drifts  # True for VWAP
        self.in_zone = False
        self.entry_count = 0

    def update(self, current_price: float) -> bool:
        if self.in_zone:
            return False
        if abs(current_price - self.price) <= 1.0:
            self.in_zone = True
            self.entry_count += 1
            return True
        return False

    def reset(self):
        self.in_zone = False


class HumanZoneTradeReset:
    """Human alert zone: 7-pt entry, resets when outcome is decided.

    Like HumanZone but no fixed 20-pt exit threshold. Instead, the zone
    resets when the alert outcome is decided (correct/incorrect/inconclusive).
    This allows faster re-alerting on the same level.
    """

    def __init__(self, price: float, drifts: bool = False):
        self.price = price
        self.drifts = drifts
        self.in_zone = False
        self.entry_count = 0

    def update(self, current_price: float) -> bool:
        if self.in_zone:
            return False
        if abs(current_price - self.price) <= 7.0:
            self.in_zone = True
            self.entry_count += 1
            return True
        return False

    def reset(self):
        self.in_zone = False


class BotZoneFixedExit:
    """Bot zone: 1-pt entry, fixed exit threshold.

    Legacy logic — zone exits when price moves exit_pts from the level.
    Superseded by BotZoneTradeReset but kept for comparison backtests.
    """

    def __init__(self, price: float, exit_pts: float = 20.0, drifts: bool = False):
        self.price = price
        self.exit_pts = exit_pts
        self.drifts = drifts
        self.in_zone = False
        self.entry_count = 0
        self._ref: float | None = None

    def update(self, current_price: float) -> bool:
        if self.in_zone:
            ref = self.price if self.drifts else self._ref
            if ref is not None and abs(current_price - ref) > self.exit_pts:
                self.in_zone = False
                self._ref = None
            return False
        if abs(current_price - self.price) <= 1.0:
            self.in_zone = True
            self._ref = self.price
            self.entry_count += 1
            return True
        return False

    def reset(self):
        self.in_zone = False
        self._ref = None
