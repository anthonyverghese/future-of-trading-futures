"""simulate_day_v2 — backtest using production BotTrader code.

Uses the real BotTrader with a BacktestBroker, ensuring the backtest
uses the exact same entry logic, filters, and zone behavior as
production. No more drift between live and backtest code.

The old simulate_day reimplemented zone/filter logic separately.
This version feeds ticks through BotTrader.on_tick() directly.
"""

from __future__ import annotations

import datetime
import numpy as np
import pytz

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot_trader import BotTrader, BotZone
from backtest.backtest_broker import BacktestBroker, BacktestTradeRecord
from backtest.data import DayArrays
from levels import calculate_fib_levels, calculate_interior_fibs
from walk_forward import _eod_cutoff_ns

# Import DayCache type
from targeted_backtest import DayCache

_ET = pytz.timezone("America/New_York")


def simulate_day_v2(
    dc: DayCache,
    arrays: DayArrays,
    *,
    per_level_ts: dict[str, tuple[float, float]],
    per_level_caps: dict[str, int],
    exclude_levels: set[str] | None = None,
    include_ibl: bool = False,
    include_vwap: bool = False,
    direction_filter: dict[str, str] | None = None,
    daily_loss: float = 200.0,
    timeout_secs: int = 900,
    momentum_max: float = 5.0,
) -> list[BacktestTradeRecord]:
    """Simulate one day using real BotTrader + BacktestBroker.

    Returns list of BacktestTradeRecord.
    """
    fp = dc.full_prices
    ft = dc.full_ts_ns
    eod = _eod_cutoff_ns(dc.date)
    start = dc.post_ib_start_idx
    n = len(dc.post_ib_prices)

    # Create BacktestBroker
    broker = BacktestBroker(
        full_prices=fp,
        full_ts_ns=ft,
        eod_cutoff_ns=eod,
        daily_loss=daily_loss,
        timeout_secs=timeout_secs,
    )

    # Create BotTrader with the backtest broker
    bot = BotTrader.__new__(BotTrader)
    bot._broker = broker
    bot._zones = {}
    bot._price_window = __import__("collections").deque()
    bot._price_window_5m = __import__("collections").deque()
    bot._price_5m_ago = None
    bot._level_trade_counts = {}
    bot._active_trade_level = None
    bot._level_cooldown_until = {}
    bot._global_cooldown_until = 0.0
    bot._adaptive_caps_restored = True  # disabled

    # Override config values for this simulation
    import config as cfg
    orig_direction_filter = cfg.BOT_DIRECTION_FILTER
    orig_per_level_max = cfg.BOT_PER_LEVEL_MAX_ENTRIES
    orig_per_level_ts = cfg.BOT_PER_LEVEL_TS
    orig_exclude = cfg.BOT_EXCLUDE_LEVELS
    orig_include_ibh = cfg.BOT_INCLUDE_IBH
    orig_include_ibl = cfg.BOT_INCLUDE_IBL
    orig_include_vwap = cfg.BOT_INCLUDE_VWAP

    try:
        cfg.BOT_DIRECTION_FILTER = direction_filter or {}
        cfg.BOT_PER_LEVEL_MAX_ENTRIES = per_level_caps
        cfg.BOT_PER_LEVEL_TS = per_level_ts
        cfg.BOT_EXCLUDE_LEVELS = exclude_levels or set()
        cfg.BOT_INCLUDE_IBH = "IBH" not in (exclude_levels or set())
        cfg.BOT_INCLUDE_IBL = include_ibl
        cfg.BOT_INCLUDE_VWAP = include_vwap

        # Register levels (same as production main.py)
        ib_range = dc.ibh - dc.ibl
        bot.update_levels(ibh=dc.ibh, ibl=dc.ibl if include_ibl else None)
        fib_levels = calculate_fib_levels(dc.ibh, dc.ibl)
        bot.update_fib_levels(fib_levels)
        interior_fibs = calculate_interior_fibs(dc.ibh, dc.ibl)
        bot.update_fib_levels(interior_fibs)

        # Set momentum threshold
        # BotTrader reads this from the price_5m_ago check hardcoded at 5.0
        # We need to handle momentum_max differently if it's not 5.0
        # For now, the production code hardcodes 5.0, which is our deployed value

        # Compute VWAP availability
        has_vwap = include_vwap and hasattr(dc, "post_ib_vwaps") and dc.post_ib_vwaps is not None

        # Feed ticks through the production code
        for j in range(n):
            gi = start + j
            if int(ft[gi]) >= eod:
                break

            broker._current_tick_idx = gi
            broker.process_events()  # close position if exit reached

            # Stop if daily loss hit
            if broker._stopped_for_day:
                break

            pj = float(dc.post_ib_prices[j])
            et_mins = int(arrays.et_mins[gi])

            # Compute factors matching what main.py passes to on_tick
            tick_rate = float(arrays.tick_rates[gi])
            session_move = float(arrays.session_move[gi])
            range_30m = float(arrays.range_30m_pts[gi])
            sm_pct = session_move / pj * 100 if pj > 0 else 0.0

            # Update VWAP level if active
            if has_vwap and j < len(dc.post_ib_vwaps):
                vwap_price = float(dc.post_ib_vwaps[j])
                if "VWAP" in bot._zones:
                    bot._zones["VWAP"].price = vwap_price

            # Convert et_mins to time object for on_tick
            h, m = divmod(et_mins, 60)
            if h < 24:
                now_et = datetime.time(h, m)
            else:
                now_et = datetime.time(23, 59)

            # Monday check for double caps
            is_monday = dc.date.weekday() == 0

            # Call production on_tick
            bot.on_tick(
                price=pj,
                ib_range=ib_range,
                tick_rate=tick_rate,
                session_move_pct=sm_pct,
                range_30m=range_30m,
                now_et=now_et,
            )

    finally:
        # Restore config
        cfg.BOT_DIRECTION_FILTER = orig_direction_filter
        cfg.BOT_PER_LEVEL_MAX_ENTRIES = orig_per_level_max
        cfg.BOT_PER_LEVEL_TS = orig_per_level_ts
        cfg.BOT_EXCLUDE_LEVELS = orig_exclude
        cfg.BOT_INCLUDE_IBH = orig_include_ibh
        cfg.BOT_INCLUDE_IBL = orig_include_ibl
        cfg.BOT_INCLUDE_VWAP = orig_include_vwap

    return broker.trades
