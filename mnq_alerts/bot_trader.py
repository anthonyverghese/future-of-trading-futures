"""
bot_trader.py — Bot trading logic, separate from human alert system.

Manages its own zone tracking (1pt entry, 15pt exit) and delegates
order execution to IBKRBroker. Main.py calls into this module without
needing to know bot internals.

Bot parameters (validated over 214 days in bot_risk_backtest.py):
  - Entry: price within 1 pt of level (vs 7 pt for human alerts)
  - Exit zone reset: 15 pts away (vs 20 pt for human alerts)
  - Target: +12 pts, Stop: -25 pts
  - Risk: $150/day loss limit, 3 consecutive loss stop, 1 position at a time
"""

from __future__ import annotations

from broker import IBKRBroker
from config import BOT_ENTRY_THRESHOLD, BOT_EXIT_THRESHOLD


class BotZone:
    """Zone tracker for a single level using bot thresholds (1pt/15pt)."""

    def __init__(self, name: str, price: float) -> None:
        self.name = name
        self.price = price
        self.in_zone = False
        self._ref_price: float | None = None

    def update(self, current_price: float) -> bool:
        """Returns True on fresh zone entry (price within BOT_ENTRY_THRESHOLD)."""
        if self.in_zone:
            if (
                self._ref_price is not None
                and abs(current_price - self._ref_price) > BOT_EXIT_THRESHOLD
            ):
                self.in_zone = False
                self._ref_price = None
            return False
        if abs(current_price - self.price) <= BOT_ENTRY_THRESHOLD:
            self.in_zone = True
            self._ref_price = self.price
            return True
        return False


class BotTrader:
    """Coordinates bot zone tracking and order submission.

    Keeps bot logic isolated from the human alert system in main.py.
    """

    def __init__(self) -> None:
        self._broker = IBKRBroker()
        self._zones: dict[str, BotZone] = {}

    def connect(self) -> bool:
        """Connect to IBKR. Returns True on success."""
        return self._broker.connect()

    @property
    def is_connected(self) -> bool:
        return self._broker.is_connected

    def process_events(self) -> None:
        """Pump ib_insync event loop so fill callbacks fire."""
        if self._broker.is_connected:
            self._broker.process_events()

    def update_level(self, name: str, price: float) -> None:
        """Register or update a price level for bot zone tracking."""
        self._zones[name] = BotZone(name, price)

    def update_levels(
        self,
        ibh: float | None = None,
        ibl: float | None = None,
        vwap: float | None = None,
    ) -> None:
        """Bulk update levels (mirrors AlertManager.update_levels)."""
        for name, price in {"IBH": ibh, "IBL": ibl, "VWAP": vwap}.items():
            if price is not None:
                self._zones[name] = BotZone(name, price)

    def update_fib_levels(self, fib_levels: dict[str, float]) -> None:
        """Register fib levels for bot zone tracking."""
        for name, price in fib_levels.items():
            self._zones[name] = BotZone(name, price)

    def on_tick(self, price: float) -> None:
        """Check all bot zones and submit orders on fresh entries."""
        if not self._broker.is_connected:
            return
        for bz in self._zones.values():
            if bz.update(price):
                direction = "up" if price > bz.price else "down"
                allowed, reason = self._broker.can_trade()
                if allowed:
                    result = self._broker.submit_bracket(
                        direction=direction,
                        current_price=price,
                        line_price=bz.price,
                        level_name=bz.name,
                    )
                    if not result.success:
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
