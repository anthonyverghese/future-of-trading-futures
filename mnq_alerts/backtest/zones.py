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


class BotZoneConfirmed:
    """Bot zone: 1-pt entry, requires N confirmed bounces before trading.

    First N touches observe only — price must bounce from the level
    to confirm it holds. After N confirmations, trades normally.

    Tolerance = 5pts: price within 5pts of level is "testing" it.
    Bounce = 8pts: price must reverse 8pts from its extreme to confirm.
    Deep penetration = price goes >5pts through: sets deep_penetration
    flag, reducing trust in the level.

    Example: level=27241, price dips to 27237 (4pts, within tolerance),
    bounces to 27245 (8pts from low). Confirmed.
    """

    PENETRATION_TOLERANCE = 5.0  # max pts price can go within and still be "testing"

    def __init__(self, price: float, drifts: bool = False,
                 required_confirms: int = 1, bounce_pts: float = 8.0):
        self.price = price
        self.drifts = drifts
        self.in_zone = False
        self.entry_count = 0
        self._required = required_confirms
        self._bounce_pts = bounce_pts
        self._confirms = 0
        self._observing = False
        self._obs_min = 0.0  # min price during observation
        self._obs_max = 0.0  # max price during observation
        self.deep_penetration = False  # set True if price breaks > tolerance

    def update(self, current_price: float) -> bool:
        dist = abs(current_price - self.price)

        # If observing a potential confirmation bounce...
        if self._observing:
            if current_price < self._obs_min:
                self._obs_min = current_price
            if current_price > self._obs_max:
                self._obs_max = current_price

            if dist > self.PENETRATION_TOLERANCE:
                # Price left the tolerance zone — evaluate the bounce.
                # Don't include the exit price in obs tracking — it's
                # outside the tolerance zone and would inflate metrics.
                self._evaluate_bounce(current_price)
                self._observing = False
                self.in_zone = False
            return False

        if self.in_zone:
            return False

        if dist <= 1.0:
            if self._confirms >= self._required:
                # Level is confirmed — trade normally.
                self.in_zone = True
                self.entry_count += 1
                return True
            else:
                # Not confirmed yet — observe this touch.
                self._observing = True
                self._obs_min = current_price
                self._obs_max = current_price
                self.in_zone = True
                return False

        return False

    def _evaluate_bounce(self, final_price: float):
        """Check if the observation period showed a valid bounce.

        Uses obs_min/obs_max (prices within tolerance zone) for
        penetration. Uses final_price direction to determine bounce
        direction. obs_range = reversal size within the zone.

        Bounce UP: dipped below level (within tolerance), price then
        exited above level. obs_range ≥ bounce_pts.
        Bounce DOWN: spiked above level (within tolerance), price then
        exited below level. obs_range ≥ bounce_pts.
        Deep penetration: set when the exit direction shows price
        broke through in a way that exceeds tolerance.
        """
        level = self.price
        dip_below = max(0, level - self._obs_min)
        spike_above = max(0, self._obs_max - level)
        obs_range = self._obs_max - self._obs_min

        # Deep penetration check based on exit direction.
        # If price exited above: the "test" was below, check if
        # the dip was too deep. If exited below: check the spike.
        if final_price > level and dip_below > self.PENETRATION_TOLERANCE:
            self.deep_penetration = True
        if final_price < level and spike_above > self.PENETRATION_TOLERANCE:
            self.deep_penetration = True
        # Also check if BOTH sides exceeded tolerance (price went
        # through the level in both directions during observation).
        if dip_below > self.PENETRATION_TOLERANCE and spike_above > self.PENETRATION_TOLERANCE:
            self.deep_penetration = True

        # Bounce UP: exited above, dipped below within tolerance.
        up_confirmed = (
            final_price > level
            and self._obs_min < level
            and dip_below <= self.PENETRATION_TOLERANCE
            and obs_range >= self._bounce_pts
        )
        # Bounce DOWN: exited below, spiked above within tolerance.
        down_confirmed = (
            final_price < level
            and self._obs_max > level
            and spike_above <= self.PENETRATION_TOLERANCE
            and obs_range >= self._bounce_pts
        )

        if up_confirmed or down_confirmed:
            self._confirms += 1

    def reset(self):
        self.in_zone = False
        self._observing = False
        self._obs_min = 0.0
        self._obs_max = 0.0


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
