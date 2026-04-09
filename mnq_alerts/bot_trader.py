"""
bot_trader.py — Bot trading logic, separate from human alert system.

Manages its own zone tracking (1pt entry, 15pt exit) and delegates
order execution to IBKRBroker. Main.py calls into this module without
needing to know bot internals.

Bot parameters (validated via walk-forward in walk_forward.py):
  - Entry: price within 1 pt of level (vs 7 pt for human alerts)
  - Exit zone reset: 15 pts away (vs 20 pt for human alerts)
  - Target: +12 pts, Stop: -25 pts, 15-min per-trade timeout
  - Risk: $150/day loss limit, 4 consecutive loss stop, 1 position at a time
"""

from __future__ import annotations

import datetime
from collections import deque

import pytz

from broker import IBKRBroker
from config import (
    BOT_ENTRY_THRESHOLD,
    BOT_EXIT_THRESHOLD,
    BOT_MAX_ENTRIES_PER_LEVEL,
    BOT_MIN_SCORE,
    BOT_TREND_LOOKBACK_MIN,
    BOT_TREND_PENALTY,
    BOT_TREND_THRESHOLD,
)

_ET = pytz.timezone("America/New_York")


class BotZone:
    """Zone tracker for a single level using bot thresholds (1pt/15pt).

    For drifting levels (VWAP), exit check uses the current level price
    instead of the locked entry reference, preventing rapid re-triggering
    as VWAP drifts toward price after a zone exit.
    """

    def __init__(self, name: str, price: float, drifts: bool = False) -> None:
        self.name = name
        self.price = price
        self.in_zone = False
        self.entry_count = 0
        self.drifts = drifts  # True for VWAP
        self._ref_price: float | None = None

    def update(self, current_price: float) -> bool:
        """Returns True on fresh zone entry (price within BOT_ENTRY_THRESHOLD)."""
        if self.in_zone:
            exit_ref = self.price if self.drifts else self._ref_price
            if (
                exit_ref is not None
                and abs(current_price - exit_ref) > BOT_EXIT_THRESHOLD
            ):
                self.in_zone = False
                self._ref_price = None
            return False
        if abs(current_price - self.price) <= BOT_ENTRY_THRESHOLD:
            self.in_zone = True
            self._ref_price = self.price
            self.entry_count += 1
            return True
        return False


def bot_entry_score(
    level: str, direction: str, entry_count: int, trend_60m: float = 0.0
) -> int:
    """Score a bot zone entry. Higher = better quality.

    Walk-forward validated over 318 days (VWAP-corrected):
    - Score >= 1 filter: 4.0x P&L/DD
    - Adding 60m trend penalty + max 5/level: 8.1x P&L/DD
    """
    score = 0
    # Level quality
    if level == "IBL":
        score += 2
    elif level in ("FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"):
        score += 1
    elif level == "IBH":
        score -= 1

    # Direction + level combos
    if level == "IBL" and direction == "up":
        score += 1
    if level == "IBH" and direction == "down":
        score += 1
    if level == "IBH" and direction == "up":
        score -= 1

    # Entry count (retest)
    if entry_count == 1:
        score -= 2
    elif entry_count >= 3:
        score += 1

    # Time of day
    now_et = datetime.datetime.now(_ET)
    hour = now_et.hour + now_et.minute / 60.0
    if hour >= 15.0:  # power hour
        score += 2
    elif 13.0 <= hour < 15.0:  # afternoon
        score += 1
    elif 10.5 <= hour < 11.5:  # post-IB weakness
        score -= 1

    # Counter-trend penalty: don't buy into a falling market or sell into a rising one.
    if direction == "up" and trend_60m < -BOT_TREND_THRESHOLD:
        score += BOT_TREND_PENALTY
    elif direction == "down" and trend_60m > BOT_TREND_THRESHOLD:
        score += BOT_TREND_PENALTY

    return score


class BotTrader:
    """Coordinates bot zone tracking and order submission.

    Keeps bot logic isolated from the human alert system in main.py.
    """

    def __init__(self) -> None:
        self._broker = IBKRBroker()
        self._zones: dict[str, BotZone] = {}
        # Rolling 60-min price window for trend calculation.
        self._price_window: deque[tuple[datetime.datetime, float]] = deque()
        # Per-level daily trade count (reset each day via reset_daily_state).
        self._level_trade_counts: dict[str, int] = {}

    def connect(self) -> bool:
        """Connect to IBKR. Returns True on success."""
        return self._broker.connect()

    @property
    def is_connected(self) -> bool:
        return self._broker.is_connected

    def process_events(self) -> None:
        """Pump ib_insync event loop so fill callbacks fire.

        Also checks if the open position (if any) has exceeded the per-trade
        timeout and closes it at market. This matches the 15-min window
        assumed by bot_risk_backtest.py.
        """
        if self._broker.is_connected:
            self._broker.process_events()
            self._broker.check_position_timeout()

    def update_level(self, name: str, price: float) -> None:
        """Register or update a price level for bot zone tracking."""
        self._zones[name] = BotZone(name, price)

    def update_levels(
        self,
        ibh: float | None = None,
        ibl: float | None = None,
        vwap: float | None = None,
    ) -> None:
        """Bulk update levels. Updates price on existing zones without resetting state.

        VWAP is excluded from bot trading — walk-forward over 318 days showed
        VWAP is net negative (-$68 P&L, $1,710 MaxDD) while IBH/IBL/Fib are
        all solidly positive.
        """
        for name, price in {"IBH": ibh, "IBL": ibl}.items():
            if price is not None:
                if name in self._zones:
                    self._zones[name].price = price
                else:
                    self._zones[name] = BotZone(name, price)

    def update_fib_levels(self, fib_levels: dict[str, float]) -> None:
        """Register fib levels for bot zone tracking."""
        for name, price in fib_levels.items():
            if name not in self._zones:
                self._zones[name] = BotZone(name, price)

    def on_tick(self, price: float) -> None:
        """Check all bot zones and submit orders on fresh entries."""
        if not self._broker.is_connected:
            return

        # Update 60-min price window for trend calculation.
        now = datetime.datetime.now(_ET)
        self._price_window.append((now, price))
        cutoff = now - datetime.timedelta(minutes=BOT_TREND_LOOKBACK_MIN)
        while self._price_window and self._price_window[0][0] < cutoff:
            self._price_window.popleft()
        # Compute trend: price change over the window.
        if len(self._price_window) >= 2:
            trend_60m = self._price_window[-1][1] - self._price_window[0][1]
        else:
            trend_60m = 0.0

        for bz in self._zones.values():
            if bz.update(price):
                direction = "up" if price > bz.price else "down"

                # Per-level daily trade cap.
                level_trades = self._level_trade_counts.get(bz.name, 0)
                if level_trades >= BOT_MAX_ENTRIES_PER_LEVEL:
                    print(
                        f"[bot] Skipped {bz.name} (max {BOT_MAX_ENTRIES_PER_LEVEL} "
                        f"trades/level/day reached)"
                    )
                    continue

                score = bot_entry_score(bz.name, direction, bz.entry_count, trend_60m)
                if score < BOT_MIN_SCORE:
                    print(
                        f"[bot] Skipped {bz.name} (score {score} < {BOT_MIN_SCORE}) | "
                        f"test #{bz.entry_count}, {direction}, trend={trend_60m:+.0f}"
                    )
                    continue
                allowed, reason = self._broker.can_trade()
                if allowed:
                    result = self._broker.submit_bracket(
                        direction=direction,
                        current_price=price,
                        line_price=bz.price,
                        level_name=bz.name,
                    )
                    if result.success:
                        self._level_trade_counts[bz.name] = level_trades + 1
                    else:
                        print(f"[broker] Trade failed: {result.error}")
                else:
                    print(f"[broker] Skipped {bz.name}: {reason}")

    def advance_zones(self, price: float) -> None:
        """Update zone state without trading (used during replay)."""
        for bz in self._zones.values():
            bz.update(price)

    def reset_daily_state(self) -> None:
        """Reset risk counters and clear zones for a new session."""
        self._broker.reset_daily_state()
        self._zones.clear()
        self._price_window.clear()
        self._level_trade_counts.clear()

    def eod_flatten(self) -> None:
        """Flatten open position a few minutes before market close.

        Does not disconnect — close_session() still runs at 4pm for summary.
        Blocks any new trades after this is called.
        """
        self._broker.eod_flatten()

    def close_session(self) -> None:
        """Cancel open orders, flatten positions, disconnect."""
        if self._broker.is_connected:
            self._broker.cancel_all_mnq_orders()
            self._broker.flatten_positions()
            self._broker.disconnect()

    @property
    def daily_stats(self) -> str:
        return self._broker.daily_stats

    @property
    def daily_summary(self) -> str:
        """Multi-line summary for end-of-day push notification."""
        b = self._broker
        total = b._trades_today
        if total == 0:
            return "No trades today"
        wr = b._wins_today / total * 100 if total > 0 else 0
        lines = [
            f"{total} trades",
            f"W {b._wins_today} / L {b._losses_today}",
            f"Win rate: {wr:.0f}%",
            f"P&L: ${b._daily_pnl_usd:+.2f}",
        ]
        if b._stopped_for_day:
            lines.append(f"Stopped: {b._stop_reason}")
        return "\n".join(lines)
