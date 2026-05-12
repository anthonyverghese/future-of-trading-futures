"""
bot_trader.py — Bot trading logic, separate from human alert system.

Manages its own zone tracking (1pt entry) and delegates order execution
to IBKRBroker. Main.py calls into this module without needing to know
bot internals.

Bot parameters (updated 2026-05-02, validated over 336 days):
  - Entry: price within 1 pt of level (vs 7 pt for human alerts)
  - Zone: stays in_zone after entry until price leaves 1pt or trade closes
  - Target/Stop: per-level (MFE-based)
  - Risk: $200/day loss limit, 1 position at a time, 15-min timeout
  - Momentum filter: skip if 5-min price change > 5pts in trade direction
  - Levels: IBH (SELL only), FIB_EXT_HI, FIB_EXT_LO, FIB_0.236, FIB_0.618, FIB_0.764
    (IBL, VWAP, FIB_0.5 excluded — weak or deteriorating edge)
  - Per-level entry caps: data-driven from WR-by-entry-count analysis
  - Monday double caps
  - Scoring: unscored (scoring hurts OOS, validated 2026-04-26)
"""

from __future__ import annotations

import datetime
import time
from collections import deque

import pandas as pd
import pytz

from broker import IBKRBroker
from cache import load_bot_daily_level_counts
from config import (
    BOT_DIRECTION_FILTER,
    BOT_ENTRY_THRESHOLD,
    BOT_EXCLUDE_LEVELS,
    BOT_FAILED_FILL_COOLDOWN_SECS,
    BOT_GLOBAL_COOLDOWN_AFTER_LOSS_SECS,
    BOT_INCLUDE_IBH,
    BOT_INCLUDE_IBL,
    BOT_INCLUDE_INTERIOR_FIBS,
    BOT_INCLUDE_VWAP,
    BOT_MAX_ENTRIES_PER_LEVEL,
    BOT_ENTRY_LIMIT_BUFFER_PTS,
    BOT_MIN_SCORE,
    BOT_MOMENTUM_LOOKBACK_MIN,
    BOT_MOMENTUM_THRESHOLD,
    BOT_MONDAY_DOUBLE_CAPS,
    BOT_PER_LEVEL_MAX_ENTRIES,
    BOT_PER_LEVEL_TS,
    BOT_STOP_POINTS,
    BOT_TARGET_POINTS,
    BOT_TREND_LOOKBACK_MIN,
    BOT_VOL_FILTER_MIN_RANGE_PCT,
    BOT_COUNTER_TREND_VALLEY_FILTER,
    BOT_HYBRID_MIN_COMPOSITE_SCORE,
    BOT_MAX_SECS_SINCE_LAST_TRADE_THIS_DAY,
)
from scoring import composite_score as _human_composite_score
# SUPPRESSED_WINDOWS import removed — bot suppression disabled (2026-05-02).
# Human app still uses it via alert_manager.py.

_ET = pytz.timezone("America/New_York")
_TREND_LOOKBACK_TD = datetime.timedelta(minutes=BOT_TREND_LOOKBACK_MIN)
_MOMENTUM_LOOKBACK_TD = datetime.timedelta(minutes=BOT_MOMENTUM_LOOKBACK_MIN)


class BotZone:
    """Zone tracker for a single level.

    Zone lifecycle:
    - Zone enters when price within BOT_ENTRY_THRESHOLD (1 pt) of level
    - Zone stays in_zone (won't re-fire) until price leaves 1pt or
      reset() is called (trade close resets ALL zones)
    """

    def __init__(self, name: str, price: float, drifts: bool = False) -> None:
        self.name = name
        self.price = price
        self.in_zone = False
        self.entry_count = 0
        self.drifts = drifts  # True for VWAP

    def update(self, current_price: float) -> bool:
        """Returns True on fresh zone entry (price within BOT_ENTRY_THRESHOLD).

        Zone lifecycle:
        - Price enters 1pt threshold → fires (returns True), in_zone=True
        - Zone stays in_zone (won't re-fire) until either:
          a) Price leaves 1pt threshold → in_zone clears, can fire on
             next approach
          b) reset() called (trade closes) → in_zone clears immediately
        - Trade close resets ALL zones so any level can fire
        """
        if self.in_zone:
            if abs(current_price - self.price) > BOT_ENTRY_THRESHOLD:
                self.in_zone = False
            return False
        if abs(current_price - self.price) <= BOT_ENTRY_THRESHOLD:
            self.in_zone = True
            # entry_count is incremented in on_tick after filters pass,
            # not here — prevents inflation from oscillation noise.
            return True
        return False

    def reset(self) -> None:
        """Reset zone after trade closes. Allows re-entry on next approach."""
        self.in_zone = False


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
        # 5-min price window for momentum filter.
        self._price_window_5m: deque[tuple[datetime.datetime, float]] = deque()
        self._price_5m_ago: float | None = None
        # Per-level daily trade count (reset each day via reset_daily_state).
        self._level_trade_counts: dict[str, int] = {}
        # Track which level has the active trade (for zone reset on close).
        self._active_trade_level: str | None = None
        # Cooldown after failed entry (unfilled limit cancel).
        # Maps level name → monotonic timestamp when cooldown expires.
        # Cooldown deadlines stored as datetime (sim/real now + delta) so
        # the comparison works correctly under both live ticks (where
        # `now` is real wall time) and backtest replay (where `now` is
        # sim time advancing per-tick). Previously these used
        # time.monotonic() which is wall-clock seconds — fine in live
        # but pinned the cooldown effectively forever in backtests
        # (entire trading session simulates in seconds of wall time, so
        # `time.monotonic() + 60` is far past every sim tick of the day).
        self._level_cooldown_until: dict[str, datetime.datetime] = {}
        # Global cooldown after any stop loss — monotonic timestamp.
        self._global_cooldown_until: datetime.datetime | None = None
        # Adaptive caps: DISABLED (2026-05-02). Hurts P&L in all configs.
        # Fields kept for backtest compatibility (simulate_day uses them).
        self._adaptive_caps_restored: bool = True  # True = disabled
        # Throttled logging for the vol filter — log at most once per
        # 60s per level. Without throttling, sustained low-vol periods
        # would log on every fresh zone entry (hundreds per day per level
        # during chop). Without ANY logging, vol-filter skips were
        # invisible — we only learned they were happening today
        # (2026-05-05) by reverse-engineering from tick data after a
        # missed Elite alert.
        self._vol_filter_last_log: dict[str, datetime.datetime] = {}
        # Stale-approach filter: timestamp of the most recent trade ENTRY
        # attempt today. Used by BOT_MAX_SECS_SINCE_LAST_TRADE_THIS_DAY
        # to skip entries that come after a long quiet period (Phase C
        # finding: such trades have -$0.58/tr aggregate, Q4 -$2.93/tr).
        self._last_trade_entry_time: datetime.datetime | None = None

        # V_MULTI filter (only maintained when BOT_FILTER_ENABLED=True).
        # State here is dormant unless the filter is enabled — zero cost
        # when disabled.
        try:
            from config import BOT_FILTER_ENABLED
        except ImportError:
            BOT_FILTER_ENABLED = False
        self._filter_enabled: bool = bool(BOT_FILTER_ENABLED)
        # Rolling 15-min tick buffer for filter feature compute. Bounded
        # by time — trimmed in on_tick.
        self._filter_tick_buffer: list[tuple[datetime.datetime, float, int]] = []
        # First RTH tick price for session_move (set on first call to on_tick
        # while in RTH).
        self._filter_session_open: float | None = None
        # Per-(level, human_direction) count of prior 1pt-zone entries today.
        # Matches the test_count_7pt construction used during training.
        self._filter_touches_per_dir: dict[tuple[str, str], int] = {}

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
        levels = {}
        if BOT_INCLUDE_IBH:
            levels["IBH"] = ibh
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
            if name in BOT_EXCLUDE_LEVELS:
                continue
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
        _now_override: datetime.datetime | None = None,
        _momentum_threshold: float | None = None,
    ) -> None:
        """Check all bot zones and submit orders on fresh entries.

        _now_override: if provided, use this as the current datetime instead
        of wall clock. Used by backtesting to inject simulated time so that
        time-based logic (momentum window, trend window, Monday caps) works
        correctly. Production callers should never pass this.

        _momentum_threshold: if provided, override the hardcoded 5.0 momentum
        threshold. Use 0.0 to disable the filter. Production uses 5.0.
        """
        if not self._broker.is_connected:
            return

        now = _now_override if _now_override is not None else datetime.datetime.now(_ET)

        # Maintain rolling tick buffer for V_MULTI filter (only when enabled).
        # Zero-cost when disabled. Trims to last 15 min on each tick.
        if self._filter_enabled:
            self._filter_tick_buffer.append((now, price, 1))
            if self._filter_session_open is None:
                self._filter_session_open = price
            cutoff = now - datetime.timedelta(minutes=15)
            # Trim from front; buffer is append-only so trimming is cheap.
            i = 0
            while i < len(self._filter_tick_buffer) and self._filter_tick_buffer[i][0] < cutoff:
                i += 1
            if i > 0:
                del self._filter_tick_buffer[:i]

        # Detect trade close and reset zones.
        if (
            self._active_trade_level is not None
            and not self._broker._position_open
        ):
            self._on_position_closed()

        # Update price windows for trend and momentum (O(1) per tick).
        trend_60m = self._update_price_windows(now, price)

        # 30m range as % of price (for scoring).
        range_30m_pct = (
            range_30m / price * 100 if range_30m is not None and price > 0 else None
        )

        # Early exits: position open, daily limit hit, or global cooldown.
        if self._broker._position_open:
            return
        allowed, _ = self._broker.can_trade()
        if not allowed:
            return
        if self._global_cooldown_until is not None and now < self._global_cooldown_until:
            return

        # Check each zone for entry opportunities.
        self._process_zone_entries(
            price, now, trend_60m, tick_rate, session_move_pct,
            range_30m, range_30m_pct, now_et, _momentum_threshold,
        )

    def _on_position_closed(self) -> None:
        """Handle trade close: reset zones and set cooldown if loss."""
        for z in self._zones.values():
            z.reset()
        if (
            BOT_GLOBAL_COOLDOWN_AFTER_LOSS_SECS > 0
            and self._broker._consecutive_losses > 0
        ):
            self._global_cooldown_until = (
                now + datetime.timedelta(
                    seconds=BOT_GLOBAL_COOLDOWN_AFTER_LOSS_SECS
                )
            )
        self._active_trade_level = None

    def _update_price_windows(
        self, now: datetime.datetime, price: float
    ) -> float:
        """Update 60-min trend and 5-min momentum windows. Returns trend_60m."""
        # 60-min trend.
        self._price_window.append((now, price))
        cutoff = now - _TREND_LOOKBACK_TD
        while self._price_window and self._price_window[0][0] < cutoff:
            self._price_window.popleft()
        if len(self._price_window) >= 2:
            trend_60m = self._price_window[-1][1] - self._price_window[0][1]
        else:
            trend_60m = 0.0

        # 5-min momentum.
        self._price_window_5m.append((now, price))
        cutoff_5m = now - _MOMENTUM_LOOKBACK_TD
        while self._price_window_5m and self._price_window_5m[0][0] < cutoff_5m:
            self._price_5m_ago = self._price_window_5m.popleft()[1]

        return trend_60m

    def _process_zone_entries(
        self,
        price: float,
        now: datetime.datetime,
        trend_60m: float,
        tick_rate: float,
        session_move_pct: float,
        range_30m: float | None,
        range_30m_pct: float | None,
        now_et: datetime.time | None,
        momentum_threshold: float | None = None,
    ) -> None:
        """Check all zones for entry and submit orders."""
        for bz in self._zones.values():
            # Per-level cooldown after failed entry.
            cooldown = self._level_cooldown_until.get(bz.name)
            if cooldown is not None and now < cooldown:
                continue

            if not bz.update(price):
                continue

            direction = "up" if price > bz.price else "down"

            # Direction filter (e.g., IBH SELL only).
            allowed_dir = BOT_DIRECTION_FILTER.get(bz.name)
            if allowed_dir and allowed_dir != direction:
                continue

            # Momentum filter: skip if 5-min price change > threshold
            # in the trade direction. Default 5.0, overridable for backtesting.
            mom_thresh = momentum_threshold if momentum_threshold is not None else BOT_MOMENTUM_THRESHOLD
            if mom_thresh > 0 and self._price_5m_ago is not None:
                momentum = price - self._price_5m_ago
                if direction == "down":
                    momentum = -momentum
                if momentum > mom_thresh:
                    continue

            # Per-level daily trade cap.
            level_trades = self._level_trade_counts.get(bz.name, 0)
            level_cap = BOT_PER_LEVEL_MAX_ENTRIES.get(
                bz.name, BOT_MAX_ENTRIES_PER_LEVEL
            )
            if BOT_MONDAY_DOUBLE_CAPS and now.weekday() == 0:
                level_cap *= 2
            if level_trades >= level_cap:
                continue

            # Hybrid pre-filter: skip if the human-side composite_score
            # is below threshold. The human's scoring is the production-
            # validated alert quality system (5.6 alerts/day at 82.5% WR
            # with score>=5, 318-day walk-forward). Memory says human
            # weights don't transfer to bot at 1pt entry under no-slippage,
            # but never tested under slippage modeling. Disabled by
            # default — set BOT_HYBRID_MIN_COMPOSITE_SCORE to 5 (or
            # other) to enable.
            if BOT_HYBRID_MIN_COMPOSITE_SCORE is not None:
                # Human's score uses session_move in pts; bot has pct.
                session_move_pts = session_move_pct * price / 100.0
                h_score = _human_composite_score(
                    level_name=bz.name,
                    entry_count=bz.entry_count + 1,
                    now_et=now_et,
                    tick_rate=tick_rate,
                    session_move_pts=session_move_pts,
                    direction=direction,
                    consecutive_wins=self._broker._consecutive_wins,
                    consecutive_losses=self._broker._consecutive_losses,
                )
                if h_score < BOT_HYBRID_MIN_COMPOSITE_SCORE:
                    continue

            # Counter-trend valley filter: skip moderate-counter-trend
            # trades where with_trend ∈ [lo, hi]. Walk-forward 4-quarter
            # validation (2026-05-05) showed trades in this band have
            # negative $/tr in all 4 quarters. Strong counter-trend
            # (with_trend < lo) is profitable; this band is the
            # "fakeout zone." Disabled by default — set
            # BOT_COUNTER_TREND_VALLEY_FILTER = (lo, hi) to enable.
            if BOT_COUNTER_TREND_VALLEY_FILTER is not None:
                lo, hi = BOT_COUNTER_TREND_VALLEY_FILTER
                if direction == "down":
                    with_trend = -trend_60m
                else:
                    with_trend = trend_60m
                if lo <= with_trend < hi:
                    continue

            # Stale-approach filter: skip if too long has passed since the
            # last entry attempt today. Only applies after the first trade
            # of the day. Disabled by default (threshold=0).
            if (
                BOT_MAX_SECS_SINCE_LAST_TRADE_THIS_DAY > 0
                and self._last_trade_entry_time is not None
            ):
                gap_secs = (now - self._last_trade_entry_time).total_seconds()
                if gap_secs > BOT_MAX_SECS_SINCE_LAST_TRADE_THIS_DAY:
                    continue

            # Volatility filter: skip dead markets.
            if (
                range_30m_pct is not None
                and range_30m_pct < BOT_VOL_FILTER_MIN_RANGE_PCT * 100
            ):
                last_log = self._vol_filter_last_log.get(bz.name)
                if last_log is None or (now - last_log).total_seconds() >= 60:
                    threshold_pct = BOT_VOL_FILTER_MIN_RANGE_PCT * 100
                    print(
                        f"[bot] Skipped {bz.name} (vol filter: "
                        f"range_30m {range_30m_pct:.3f}% < "
                        f"{threshold_pct:.2f}%) | dist={abs(price - bz.price):.2f}"
                    )
                    self._vol_filter_last_log[bz.name] = now
                continue

            # All filters passed — attempt entry.
            bz.entry_count += 1
            target_pts, stop_pts = BOT_PER_LEVEL_TS.get(
                bz.name, (BOT_TARGET_POINTS, BOT_STOP_POINTS)
            )
            # Fixed buffer (was target_pts/2 — caused up to 6pt slippage on
            # FIB_0.618). 1.0pt is the slippage-vs-fill-rate optimum from
            # the buffer sweep. See config.py BOT_ENTRY_LIMIT_BUFFER_PTS.
            #
            # Forms supported:
            #   float > 0: use that value for all levels (current default).
            #   float <= 0: use legacy target_pts/2 (backtest comparison).
            #   dict[level_name → float]: per-level buffer, falls back to
            #     1.0pt if the level isn't in the dict.
            if isinstance(BOT_ENTRY_LIMIT_BUFFER_PTS, dict):
                entry_limit_buffer = BOT_ENTRY_LIMIT_BUFFER_PTS.get(bz.name, 1.0)
            elif BOT_ENTRY_LIMIT_BUFFER_PTS > 0:
                entry_limit_buffer = BOT_ENTRY_LIMIT_BUFFER_PTS
            else:
                entry_limit_buffer = round(target_pts / 2 * 4) / 4
            print(
                f"[bot] Zone entry: {bz.name} test #{bz.entry_count} "
                f"{direction} @ {price:.2f} (line {bz.price:.2f}, "
                f"dist={abs(price - bz.price):.2f}, "
                f"T{target_pts}/S{stop_pts})"
            )

            score = bot_entry_score(
                bz.name, direction, bz.entry_count, trend_60m,
                tick_rate=tick_rate, session_move_pct=session_move_pct,
                range_30m_pct=range_30m_pct, now_et=now_et,
            )
            if score < BOT_MIN_SCORE:
                print(
                    f"[bot] Skipped {bz.name} (score {score} < {BOT_MIN_SCORE}) | "
                    f"test #{bz.entry_count}, {direction}, trend={trend_60m:+.0f}"
                )
                continue

            # V_MULTI filter gate (only active when BOT_FILTER_ENABLED=True).
            # Decides whether to actually place the trade based on:
            # (a) human composite_score >= 5 at the 7pt-zone moment, and
            # (b) all 3 LightGBM models (8/20, 8/25, 10/20) agree
            #     expected_pnl > 0 at their respective TP/SL geometry.
            # See bot_filter.py for details. Toggle via config.BOT_FILTER_ENABLED.
            if self._filter_enabled:
                # Track per-(level, direction) entry counts for human's
                # test_count score component. Matches training's
                # cumcount + 1 construction. Increment AFTER counting so
                # entry_count_7pt is 1-indexed including this event.
                self._filter_touches_per_dir[(bz.name, direction)] = (
                    self._filter_touches_per_dir.get((bz.name, direction), 0) + 1
                )
                entry_count_7pt = self._filter_touches_per_dir[(bz.name, direction)]
                # Approach direction: +1 if approaching from below (SELL/down),
                # -1 if from above (BUY/up). price vs level_price tells us.
                approach_direction = 1 if price < bz.price else -1
                # Build tick buffer DataFrame for the filter.
                from bot_filter import FilterContext, get_filter
                if self._filter_tick_buffer:
                    tb = pd.DataFrame(
                        {"price": [t[1] for t in self._filter_tick_buffer],
                         "size":  [t[2] for t in self._filter_tick_buffer]},
                        index=pd.DatetimeIndex(
                            [t[0] for t in self._filter_tick_buffer]
                        ).tz_convert("UTC"),
                    )
                else:
                    tb = pd.DataFrame(columns=["price", "size"],
                                      index=pd.DatetimeIndex([], tz="UTC"))
                # Gather all level prices (zones + VWAP via market_data context
                # is not available here; pass current zones only — model trained
                # with distance_to_vwap so we approximate with 0.0 when VWAP
                # unavailable. Acceptable: feature has low importance).
                all_levels = {name: z.price for name, z in self._zones.items()}
                ctx = FilterContext(
                    tick_buffer=tb,
                    session_open_price=self._filter_session_open or price,
                    levels=all_levels,
                    bot_touches_today=bz.entry_count - 1,
                    resolution_order_cw=self._broker._consecutive_wins,
                    resolution_order_cl=self._broker._consecutive_losses,
                )
                event_ts_utc = pd.Timestamp(now).tz_convert("UTC")
                decision = get_filter().should_take(
                    level_name=bz.name, level_price=bz.price,
                    event_ts=event_ts_utc, event_price=price,
                    approach_direction=approach_direction,
                    entry_count_7pt=entry_count_7pt, context=ctx,
                )
                if not decision.take:
                    print(
                        f"[bot_filter] SKIP {bz.name} ({direction}) | "
                        f"reason={decision.reason} score={decision.human_score} "
                        f"votes={decision.votes}/3"
                    )
                    continue
                else:
                    probs_str = ", ".join(
                        f"{n}:{p['p_win']:.2f}" for n, p in decision.model_probs.items()
                    )
                    print(
                        f"[bot_filter] TAKE {bz.name} ({direction}) | "
                        f"score={decision.human_score} votes={decision.votes}/3 "
                        f"({probs_str})"
                    )

            allowed, reason = self._broker.can_trade()
            if allowed:
                result = self._broker.submit_bracket(
                    direction=direction, current_price=price,
                    line_price=bz.price, level_name=bz.name,
                    score=score, trend_60m=trend_60m,
                    entry_count=bz.entry_count,
                    target_pts=target_pts, stop_pts=stop_pts,
                    entry_limit_buffer=entry_limit_buffer,
                    range_30m=range_30m, tick_rate=tick_rate,
                    session_move_pct=session_move_pct,
                )
                if result.success:
                    self._level_trade_counts[bz.name] = level_trades + 1
                    self._active_trade_level = bz.name
                    self._last_trade_entry_time = now
                else:
                    print(f"[broker] Trade failed: {result.error}")
                    self._level_cooldown_until[bz.name] = (
                        now + datetime.timedelta(
                            seconds=BOT_FAILED_FILL_COOLDOWN_SECS
                        )
                    )
                    bz.reset()
            else:
                print(f"[broker] Skipped {bz.name}: {reason}")

    def advance_zones(self, price: float) -> None:
        """Update zone state without trading (used during replay)."""
        for bz in self._zones.values():
            bz.update(price)

    def reset_zones_for_live(self) -> None:
        """Reset all zones after replay catches up to live ticks.

        During replay, advance_zones() sets in_zone=True for levels
        that price touches, but no trade fires (on_tick not called).
        Without this reset, the first live approach to each level is
        missed because the zone is already in_zone from replay.
        """
        for bz in self._zones.values():
            bz.reset()
        print("[bot] Zones reset after replay transition")

    def reset_daily_state(self) -> None:
        """Reset risk counters and clear zones for a new session."""
        self._broker.reset_daily_state()
        self._zones.clear()
        self._price_window.clear()
        self._price_window_5m.clear()
        self._price_5m_ago = None
        self._level_trade_counts.clear()
        self._active_trade_level = None
        self._level_cooldown_until.clear()
        self._global_cooldown_until = None
        self._vol_filter_last_log.clear()
        self._last_trade_entry_time = None
        self._adaptive_caps_restored = True  # disabled

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
