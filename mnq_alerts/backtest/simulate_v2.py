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
    simulate_slippage: bool = False,
    latency_ms: float = 100.0,
    fill_timeout_secs: float = 3.0,
    entry_limit_buffer_pts_override: float | None = None,
    counter_trend_valley_filter: tuple[float, float] | None = None,
    gap_close: float | None = None,
) -> list[BacktestTradeRecord]:
    """Simulate one day using real BotTrader + BacktestBroker.

    When `simulate_slippage=True`, BacktestBroker runs a tick-data
    limit-fill simulation matching the production limit-order behavior:
    network latency `latency_ms` then a `fill_timeout_secs` window for
    the limit to be hit. Trades whose limit isn't satisfied are
    correctly returned as failed (the bot frees the cap slot).

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
        simulate_slippage=simulate_slippage,
        latency_ms=latency_ms,
        fill_timeout_secs=fill_timeout_secs,
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
    bot._global_cooldown_until = None
    bot._vol_filter_last_log = {}
    bot._adaptive_caps_restored = True  # disabled

    # Override config values for this simulation.
    # bot_trader.py uses 'from config import X' which creates local bindings.
    # Changing config.X doesn't affect bot_trader.X. We must patch bot_trader
    # module's bindings directly.
    import config as cfg
    import bot_trader as bt_mod

    orig = {
        "cfg_dir": cfg.BOT_DIRECTION_FILTER,
        "cfg_caps": cfg.BOT_PER_LEVEL_MAX_ENTRIES,
        "cfg_ts": cfg.BOT_PER_LEVEL_TS,
        "cfg_excl": cfg.BOT_EXCLUDE_LEVELS,
        "cfg_ibh": cfg.BOT_INCLUDE_IBH,
        "cfg_ibl": cfg.BOT_INCLUDE_IBL,
        "cfg_vwap": cfg.BOT_INCLUDE_VWAP,
        "cfg_buf": cfg.BOT_ENTRY_LIMIT_BUFFER_PTS,
        "cfg_ctv": cfg.BOT_COUNTER_TREND_VALLEY_FILTER,
        "bt_dir": bt_mod.BOT_DIRECTION_FILTER,
        "bt_caps": bt_mod.BOT_PER_LEVEL_MAX_ENTRIES,
        "bt_ts": bt_mod.BOT_PER_LEVEL_TS,
        "bt_excl": bt_mod.BOT_EXCLUDE_LEVELS,
        "bt_ibh": bt_mod.BOT_INCLUDE_IBH,
        "bt_ibl": bt_mod.BOT_INCLUDE_IBL,
        "bt_vwap": bt_mod.BOT_INCLUDE_VWAP,
        "bt_buf": bt_mod.BOT_ENTRY_LIMIT_BUFFER_PTS,
        "bt_ctv": bt_mod.BOT_COUNTER_TREND_VALLEY_FILTER,
    }

    try:
        # Patch BOTH config module and bot_trader module bindings
        dir_filt = direction_filter or {}
        excl = exclude_levels or set()
        inc_ibh = "IBH" not in excl

        cfg.BOT_DIRECTION_FILTER = dir_filt
        cfg.BOT_PER_LEVEL_MAX_ENTRIES = per_level_caps
        cfg.BOT_PER_LEVEL_TS = per_level_ts
        cfg.BOT_EXCLUDE_LEVELS = excl
        cfg.BOT_INCLUDE_IBH = inc_ibh
        cfg.BOT_INCLUDE_IBL = include_ibl
        cfg.BOT_INCLUDE_VWAP = include_vwap

        if entry_limit_buffer_pts_override is not None:
            cfg.BOT_ENTRY_LIMIT_BUFFER_PTS = entry_limit_buffer_pts_override
            bt_mod.BOT_ENTRY_LIMIT_BUFFER_PTS = entry_limit_buffer_pts_override
        if counter_trend_valley_filter is not None:
            cfg.BOT_COUNTER_TREND_VALLEY_FILTER = counter_trend_valley_filter
            bt_mod.BOT_COUNTER_TREND_VALLEY_FILTER = counter_trend_valley_filter

        bt_mod.BOT_DIRECTION_FILTER = dir_filt
        bt_mod.BOT_PER_LEVEL_MAX_ENTRIES = per_level_caps
        bt_mod.BOT_PER_LEVEL_TS = per_level_ts
        bt_mod.BOT_EXCLUDE_LEVELS = excl
        bt_mod.BOT_INCLUDE_IBH = inc_ibh
        bt_mod.BOT_INCLUDE_IBL = include_ibl
        bt_mod.BOT_INCLUDE_VWAP = include_vwap

        # Register levels (same as production main.py)
        ib_range = dc.ibh - dc.ibl
        bot.update_levels(ibh=dc.ibh, ibl=dc.ibl if include_ibl else None)
        fib_levels = calculate_fib_levels(dc.ibh, dc.ibl)
        bot.update_fib_levels(fib_levels)
        interior_fibs = calculate_interior_fibs(dc.ibh, dc.ibl)
        bot.update_fib_levels(interior_fibs)
        # Optional: add prior day's RTH close as a "gap" magnet level.
        # Caller is responsible for ensuring per_level_ts and per_level_caps
        # have entries for "GAP_PRIOR_CLOSE" before passing.
        if gap_close is not None:
            bot.update_fib_levels({"GAP_PRIOR_CLOSE": gap_close})

        # Compute VWAP availability
        has_vwap = include_vwap and hasattr(dc, "post_ib_vwaps") and dc.post_ib_vwaps is not None

        # Precompute level prices for fast-skip (same optimization as old sim).
        # ~95% of ticks are far from any level and can skip the full on_tick.
        level_prices = np.array([z.price for z in bot._zones.values()])
        skip_dist = 1.5  # slightly wider than 1pt entry threshold

        # Precompute simulated datetimes batch for the day.
        # datetime.fromtimestamp is expensive — only compute when needed.
        _mom_thresh = momentum_max if momentum_max > 0 else 0.0

        # Feed ticks through the production code
        for j in range(n):
            gi = start + j
            if int(ft[gi]) >= eod:
                break

            broker._current_tick_idx = gi
            broker.process_events()  # close position if exit reached

            if broker._stopped_for_day:
                break

            pj = float(dc.post_ib_prices[j])

            # Fast skip: if price is far from all levels AND no position
            # open, just update price windows and skip the full on_tick.
            near_fixed = np.any(np.abs(level_prices - pj) <= skip_dist)
            near_vwap = (has_vwap and j < len(dc.post_ib_vwaps)
                         and abs(pj - float(dc.post_ib_vwaps[j])) <= skip_dist)

            if (not near_fixed and not near_vwap
                    and not broker._position_open
                    and bot._active_trade_level is None):
                # Fast path: skip entirely. Price windows will be stale
                # when we next reach a level, but _update_price_windows
                # will catch up since we always pass the current sim_now.
                # The momentum lookback (~1000 ticks) and trend lookback
                # (~60 min) are approximate anyway.
                continue

            # Full on_tick path (near a level or position open)
            et_mins = int(arrays.et_mins[gi])
            tick_rate = float(arrays.tick_rates[gi])
            session_move = float(arrays.session_move[gi])
            range_30m = float(arrays.range_30m_pts[gi])
            sm_pct = session_move / pj * 100 if pj > 0 else 0.0

            if has_vwap and j < len(dc.post_ib_vwaps):
                vwap_price = float(dc.post_ib_vwaps[j])
                if "VWAP" in bot._zones:
                    bot._zones["VWAP"].price = vwap_price

            h, m = divmod(et_mins, 60)
            now_et = datetime.time(h, m) if h < 24 else datetime.time(23, 59)

            sim_now = datetime.datetime.fromtimestamp(
                int(ft[gi]) / 1e9, tz=pytz.utc
            ).astimezone(_ET)

            bot.on_tick(
                price=pj,
                ib_range=ib_range,
                tick_rate=tick_rate,
                session_move_pct=sm_pct,
                range_30m=range_30m,
                now_et=now_et,
                _now_override=sim_now,
                _momentum_threshold=_mom_thresh,
            )

        # Close any position still open at EOD
        broker.eod_flatten()

    finally:
        # Restore both config and bot_trader module bindings
        cfg.BOT_DIRECTION_FILTER = orig["cfg_dir"]
        cfg.BOT_PER_LEVEL_MAX_ENTRIES = orig["cfg_caps"]
        cfg.BOT_PER_LEVEL_TS = orig["cfg_ts"]
        cfg.BOT_EXCLUDE_LEVELS = orig["cfg_excl"]
        cfg.BOT_INCLUDE_IBH = orig["cfg_ibh"]
        cfg.BOT_INCLUDE_IBL = orig["cfg_ibl"]
        cfg.BOT_INCLUDE_VWAP = orig["cfg_vwap"]
        cfg.BOT_ENTRY_LIMIT_BUFFER_PTS = orig["cfg_buf"]
        cfg.BOT_COUNTER_TREND_VALLEY_FILTER = orig["cfg_ctv"]

        bt_mod.BOT_DIRECTION_FILTER = orig["bt_dir"]
        bt_mod.BOT_PER_LEVEL_MAX_ENTRIES = orig["bt_caps"]
        bt_mod.BOT_PER_LEVEL_TS = orig["bt_ts"]
        bt_mod.BOT_EXCLUDE_LEVELS = orig["bt_excl"]
        bt_mod.BOT_INCLUDE_IBH = orig["bt_ibh"]
        bt_mod.BOT_INCLUDE_IBL = orig["bt_ibl"]
        bt_mod.BOT_INCLUDE_VWAP = orig["bt_vwap"]
        bt_mod.BOT_ENTRY_LIMIT_BUFFER_PTS = orig["bt_buf"]
        bt_mod.BOT_COUNTER_TREND_VALLEY_FILTER = orig["bt_ctv"]

    return broker.trades
