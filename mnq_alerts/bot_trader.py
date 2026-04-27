"""
bot_trader.py — Bot trading logic, separate from human alert system.

Manages its own zone tracking (1pt entry, 15pt exit) and delegates
order execution to IBKRBroker. Main.py calls into this module without
needing to know bot internals.

Bot parameters (walk-forward validated 2026-04-17 over 319 days):
  - Entry: price within 1 pt of level (vs 7 pt for human alerts)
  - Exit zone reset: 15 pts away (vs 20 pt for human alerts)
  - Target/Stop: IB-range-normalized (7%/20% of IB range)
  - Risk: $100/day loss limit, 1 position at a time, 13:30-14:00 ET suppressed
  - Levels: IBH, FIB_EXT_HI, FIB_EXT_LO, FIB_0.236, FIB_0.5, FIB_0.618, FIB_0.786
    (IBL and VWAP excluded — weak OOS performance)
  - Scoring: unscored (scoring hurts OOS, validated 2026-04-26)
"""

from __future__ import annotations

import datetime
from collections import deque

import pytz

from broker import IBKRBroker
from cache import load_bot_daily_level_counts
from config import (
    BOT_ENTRY_THRESHOLD,
    BOT_INCLUDE_IBL,
    BOT_INCLUDE_INTERIOR_FIBS,
    BOT_INCLUDE_VWAP,
    BOT_MAX_ENTRIES_PER_LEVEL,
    BOT_MIN_SCORE,
    BOT_PER_LEVEL_TS,
    BOT_STOP_POINTS,
    BOT_TARGET_POINTS,
    BOT_TREND_LOOKBACK_MIN,
    BOT_VOL_FILTER_MIN_RANGE_PCT,
)
from scoring import SUPPRESSED_WINDOWS

_ET = pytz.timezone("America/New_York")


class BotZone:
    """Zone tracker for a single level.

    Zone lifecycle = trade lifecycle:
    - Zone enters when price within BOT_ENTRY_THRESHOLD (1 pt) of level
    - Zone stays active while a trade is open on this level
    - Zone resets when the trade closes (via reset() call from BotTrader)
    - No fixed exit threshold — the zone only resets on trade close
    """

    def __init__(self, name: str, price: float, drifts: bool = False) -> None:
        self.name = name
        self.price = price
        self.in_zone = False
        self.suppressed = False  # True when entry was blocked; clears on price exit
        self.entry_count = 0
        self.drifts = drifts  # True for VWAP

    def update(self, current_price: float) -> bool:
        """Returns True on fresh zone entry (price within BOT_ENTRY_THRESHOLD).

        Zone lifecycle:
        - Price enters 1pt threshold → fires (returns True)
        - If trade opens → zone stays in_zone, reset on trade close
        - If trade blocked/skipped → zone is suppressed, won't re-fire
          until price leaves the threshold and comes back
        """
        near = abs(current_price - self.price) <= BOT_ENTRY_THRESHOLD
        if self.in_zone or self.suppressed:
            if not near:
                self.in_zone = False
                self.suppressed = False
            return False
        if near:
            self.in_zone = True
            self.entry_count += 1
            return True
        return False

    def suppress(self) -> None:
        """Suppress zone after a skipped/blocked entry.

        Zone won't re-fire until price leaves the threshold area.
        """
        self.in_zone = False
        self.suppressed = True

    def reset(self) -> None:
        """Reset zone after trade closes. Allows re-entry on next approach."""
        self.in_zone = False
        self.suppressed = False


def bot_entry_score(
    level: str,
    direction: str,
    entry_count: int,
    trend_60m: float = 0.0,
    tick_rate: float = 0.0,
    session_move_pct: float = 0.0,
    range_30m_pct: float | None = None,
    now_et: datetime.time | None = None,
) -> int:
    """Score a bot zone entry using bot-specific weights.

    Weights derived from factor analysis on 319 days of 1-pt bot entry
    outcomes (bot_pct_backtest.py, 2026-04-17). These differ from the
    human alert weights — e.g., power hour is -2 for bot entries (67.3%
    WR, worst time bucket) but +2 for human alerts.
    """
    score = 0

    # Level quality (bot: IBL +1, FIB_LO +1, others 0 or -1)
    if level == "IBL":
        score += 1
    elif level == "FIB_EXT_LO_1.272":
        score += 1
    elif level == "IBH":
        score -= 1
    # VWAP, FIB_HI: 0

    # Direction × level combos
    combo = (level, direction)
    if combo == ("FIB_EXT_LO_1.272", "down"):
        score += 2
    elif combo == ("IBL", "down"):
        score += 1
    elif combo == ("FIB_EXT_HI_1.272", "up"):
        score += 1
    elif combo == ("IBH", "up"):
        score -= 1
    elif combo == ("FIB_EXT_HI_1.272", "down"):
        score -= 1
    elif combo == ("VWAP", "down"):
        score -= 1

    # Entry count (test #)
    if entry_count == 2:
        score += 1
    elif entry_count == 3:
        score -= 1

    # Time of day
    if now_et is not None:
        mins = now_et.hour * 60 + now_et.minute
        if 10 * 60 + 31 <= mins < 11 * 60 + 30:
            score += 1  # post-IB: 73.3% (best bucket for bot)
        elif mins >= 15 * 60:
            score -= 2  # power hour: 67.3% (worst for bot)

    # Tick rate
    if tick_rate < 500:
        score -= 2
    elif tick_rate < 1000:
        score -= 1
    elif tick_rate >= 2500:
        score += 1

    # Session move (%-based)
    if -0.09 < session_move_pct <= -0.04:
        score += 1
    elif -0.05 < session_move_pct < 0:
        score += 1
    elif session_move_pct > 0.20:
        score -= 1

    # 30-min range volatility (%-based)
    if range_30m_pct is not None:
        if range_30m_pct < 0.15:
            score -= 4  # dead market: 61.8% WR
        elif range_30m_pct > 0.50:
            score += 1
        elif 0.35 <= range_30m_pct <= 0.50:
            score -= 1

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
        # Track which level has the active trade (for zone reset on close).
        self._active_trade_level: str | None = None

    def connect(self) -> bool:
        """Connect to IBKR. Returns True on success."""
        if not self._broker.connect():
            return False
        # Restore per-level daily caps from today's closed trades so a
        # restart can't hand each level a fresh BOT_MAX_ENTRIES_PER_LEVEL
        # allotment. Broker restores its own counters inside connect().
        try:
            # Match the system-local tz convention used when bot_trades
            # rows are written in broker._on_order_status.
            now = datetime.datetime.now(datetime.timezone.utc).astimezone()
            self._level_trade_counts = load_bot_daily_level_counts(
                now.strftime("%Y-%m-%d")
            )
            if self._level_trade_counts:
                summary = ", ".join(
                    f"{k}={v}" for k, v in sorted(self._level_trade_counts.items())
                )
                print(f"[bot] Restored per-level trade counts: {summary}")
        except Exception as exc:
            print(f"[bot] Failed to restore per-level trade counts: {exc}")
        return True

    @property
    def is_connected(self) -> bool:
        return self._broker.is_connected

    def process_events(self) -> None:
        """Pump ib_insync event loop so fill callbacks fire.

        Also checks if the open position (if any) has exceeded the per-trade
        timeout and closes it at market. This matches the 15-min window
        assumed by bot_risk_backtest.py.

        Always calls broker.process_events() (even when disconnected) so the
        broker's auto-reconnect logic can attempt to recover from initial
        connection failures or unexpected drops.
        """
        self._broker.process_events()
        if self._broker.is_connected:
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
        """Bulk update levels. Updates price on existing zones without resetting state."""
        levels = {"IBH": ibh}
        if BOT_INCLUDE_IBL:
            levels["IBL"] = ibl
        if BOT_INCLUDE_VWAP:
            levels["VWAP"] = vwap
        for name, price in levels.items():
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

    def on_tick(
        self,
        price: float,
        ib_range: float | None = None,
        tick_rate: float = 0.0,
        session_move_pct: float = 0.0,
        range_30m: float | None = None,
        now_et: datetime.time | None = None,
    ) -> None:
        """Check all bot zones and submit orders on fresh entries."""
        if not self._broker.is_connected:
            return

        # If a trade just closed, reset the zone on that level.
        if (
            self._active_trade_level is not None
            and not self._broker._position_open
        ):
            lv = self._active_trade_level
            if lv in self._zones:
                self._zones[lv].reset()
            self._active_trade_level = None

        # Update 60-min price window for trend calculation.
        now = datetime.datetime.now(_ET)
        self._price_window.append((now, price))
        cutoff = now - datetime.timedelta(minutes=BOT_TREND_LOOKBACK_MIN)
        while self._price_window and self._price_window[0][0] < cutoff:
            self._price_window.popleft()
        if len(self._price_window) >= 2:
            trend_60m = self._price_window[-1][1] - self._price_window[0][1]
        else:
            trend_60m = 0.0

        # 30m range as % of price (for scoring).
        range_30m_pct = (
            range_30m / price * 100 if range_30m is not None and price > 0 else None
        )

        for bz in self._zones.values():
            if bz.update(price):
                direction = "up" if price > bz.price else "down"

                # Per-level target/stop (falls back to default if not configured).
                target_pts, stop_pts = BOT_PER_LEVEL_TS.get(
                    bz.name, (BOT_TARGET_POINTS, BOT_STOP_POINTS)
                )
                entry_limit_buffer = round(target_pts / 2 * 4) / 4
                print(
                    f"[bot] Zone entry: {bz.name} test #{bz.entry_count} "
                    f"{direction} @ {price:.2f} (line {bz.price:.2f}, "
                    f"dist={abs(price - bz.price):.2f}, "
                    f"T{target_pts}/S{stop_pts})"
                )

                # Suppress entries during weak time windows.
                if now_et is not None:
                    et_mins = now_et.hour * 60 + now_et.minute
                    if any(ws <= et_mins < we for ws, we in SUPPRESSED_WINDOWS):
                        print(
                            f"[bot] Skipped {bz.name} (suppressed time window "
                            f"{now_et.strftime('%H:%M')} ET)"
                        )
                        bz.suppress()
                        continue

                # Per-level daily trade cap.
                level_trades = self._level_trade_counts.get(bz.name, 0)
                if level_trades >= BOT_MAX_ENTRIES_PER_LEVEL:
                    print(
                        f"[bot] Skipped {bz.name} (max {BOT_MAX_ENTRIES_PER_LEVEL} "
                        f"trades/level/day reached)"
                    )
                    bz.suppress()
                    continue

                # Volatility filter: skip dead markets.
                if (
                    range_30m_pct is not None
                    and range_30m_pct < BOT_VOL_FILTER_MIN_RANGE_PCT * 100
                ):
                    print(
                        f"[bot] Skipped {bz.name} (low vol: 30m range "
                        f"{range_30m_pct:.3f}% < {BOT_VOL_FILTER_MIN_RANGE_PCT*100:.2f}%)"
                    )
                    bz.suppress()
                    continue

                score = bot_entry_score(
                    bz.name,
                    direction,
                    bz.entry_count,
                    trend_60m,
                    tick_rate=tick_rate,
                    session_move_pct=session_move_pct,
                    range_30m_pct=range_30m_pct,
                    now_et=now_et,
                )
                if score < BOT_MIN_SCORE:
                    print(
                        f"[bot] Skipped {bz.name} (score {score} < {BOT_MIN_SCORE}) | "
                        f"test #{bz.entry_count}, {direction}, trend={trend_60m:+.0f}"
                    )
                    bz.suppress()
                    continue
                allowed, reason = self._broker.can_trade()
                if allowed:
                    result = self._broker.submit_bracket(
                        direction=direction,
                        current_price=price,
                        line_price=bz.price,
                        level_name=bz.name,
                        score=score,
                        trend_60m=trend_60m,
                        entry_count=bz.entry_count,
                        target_pts=target_pts,
                        stop_pts=stop_pts,
                        entry_limit_buffer=entry_limit_buffer,
                    )
                    if result.success:
                        self._level_trade_counts[bz.name] = level_trades + 1
                        self._active_trade_level = bz.name
                    else:
                        print(f"[broker] Trade failed: {result.error}")
                        bz.suppress()
                else:
                    print(f"[broker] Skipped {bz.name}: {reason}")
                    bz.suppress()

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
        self._active_trade_level = None

    def eod_flatten(self) -> None:
        """Flatten open position a few minutes before market close.

        Does not disconnect — close_session() still runs at 4pm for summary.
        Blocks any new trades after this is called.
        """
        self._broker.eod_flatten()
        # Clear zone state since no more trades will happen today.
        if self._active_trade_level and self._active_trade_level in self._zones:
            self._zones[self._active_trade_level].reset()
        self._active_trade_level = None

    def close_session(self) -> None:
        """Tracked close with failsafe verification, then disconnect."""
        if self._broker.is_connected:
            self._broker.session_close()
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
