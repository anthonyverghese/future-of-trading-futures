"""Tick-by-tick simulation with integrated scoring and constraints.

Uses evaluate_bot_trade for trade resolution. Iterates all ticks but
skips quickly when no level is nearby.

Handles:
- 1-position constraint
- Scoring filter (skipped entries don't activate zone)
- Streak tracking across days
- Daily loss limit (default $100)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

from .data import DayArrays
from .scoring import EntryFactors, score_entry

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from targeted_backtest import DayCache
from bot_risk_backtest import evaluate_bot_trade, MULTIPLIER, FEE_PTS
from walk_forward import _eod_cutoff_ns


@dataclass
class TradeRecord:
    date: datetime.date
    level: str
    direction: str
    entry_count: int
    outcome: str    # win, loss, timeout
    pnl_usd: float
    factors: EntryFactors
    entry_idx: int = 0   # global tick index of entry
    exit_idx: int = 0    # global tick index of exit
    entry_ns: int = 0    # entry timestamp (ns since epoch)


def simulate_day(
    dc: DayCache,
    arrays: DayArrays,
    zone_factory,
    target_fn,
    stop_pts=None,
    max_per_level: int = 12,
    weights: dict | None = None,
    min_score: int = -99,
    streak_state: tuple[int, int] = (0, 0),
    daily_loss: float = 100.0,
    max_consec: int = 999,
    timeout_secs: int = 900,
    stop_fn=None,
    include_ibl: bool = True,
    include_vwap: bool = True,
    extra_suppressed: list[tuple[int, int]] | None = None,
    level_cooldown_secs: int = 0,
    no_repeat_loss_combo: bool = False,
    max_wins_per_level: int = 999,
    max_approach_speed: float = 0.0,
    max_per_level_map: dict[str, int] | None = None,
    exclude_levels: set[str] | None = None,
    vol_filter_pct: float = 0.0015,
) -> tuple[list[TradeRecord], tuple[int, int]]:
    """Simulate one day. Returns (trades, (cw, cl)).

    stop_pts: fixed stop for all levels (legacy).
    stop_fn: callable(level_name) -> stop_pts (per-level, takes priority).
    include_ibl: whether to include IBL level (live bot: False).
    include_vwap: whether to include VWAP level (live bot: False).
    extra_suppressed: additional (start_et_mins, end_et_mins) windows to suppress.
    level_cooldown_secs: after a trade on level X, skip X for this many seconds.
    no_repeat_loss_combo: if True, skip level+direction combo that just lost.
    max_wins_per_level: stop trading a level after this many wins (anticipate breakout).
    max_approach_speed: skip entries where approach_speed > this value (0=disabled).
    """
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    fp = dc.full_prices
    ft = dc.full_ts_ns
    eod = _eod_cutoff_ns(dc.date)

    # Level configs.
    ib_range = dc.ibh - dc.ibl
    fixed_levels = [
        ("IBH", dc.ibh, False),
        ("FIB_EXT_HI_1.272", dc.fib_hi, False),
        ("FIB_EXT_LO_1.272", dc.fib_lo, False),
        ("FIB_0.236", dc.ibl + 0.236 * ib_range, False),
        ("FIB_0.5", dc.ibl + 0.5 * ib_range, False),
        ("FIB_0.618", dc.ibl + 0.618 * ib_range, False),
        ("FIB_0.764", dc.ibl + 0.764 * ib_range, False),
    ]
    if include_ibl:
        fixed_levels.append(("IBL", dc.ibl, False))
    has_vwap = include_vwap and hasattr(dc, "post_ib_vwaps") and dc.post_ib_vwaps is not None

    # Create zones (skip excluded levels).
    zones = {
        name: zone_factory(name, price, drifts)
        for name, price, drifts in fixed_levels
        if not exclude_levels or name not in exclude_levels
    }
    if has_vwap and (not exclude_levels or "VWAP" not in exclude_levels):
        zones["VWAP"] = zone_factory("VWAP", float(dc.post_ib_vwaps[0]), True)
    ec = {name: 0 for name in zones}

    cw, cl = streak_state
    trades = []
    pos_exit_idx = -1
    dpnl = 0.0
    dcons = 0
    stopped = False
    # Per-level cooldown: track last trade exit timestamp (ns) per level.
    level_last_exit_ns: dict[str, int] = {}
    # Per-level win count (for max_wins_per_level).
    level_wins: dict[str, int] = {}
    # Lost combos: set of (level, direction) that just lost.
    lost_combos: set[tuple[str, str]] = set()

    # Precompute fixed level prices for fast distance check (only active levels).
    fixed_prices = np.array([lv for name, lv, _ in fixed_levels if name in zones])

    for j in range(n):
        gi = start + j
        ens = int(ft[gi])
        if ens >= eod:
            break
        if stopped:
            break
        if gi <= pos_exit_idx:
            continue

        pj = float(prices[j])

        # Fast skip: check if price is near ANY fixed level.
        near_fixed = np.any(np.abs(fixed_prices - pj) <= 1.0)
        near_vwap = has_vwap and abs(pj - float(dc.post_ib_vwaps[j])) <= 1.0
        if not near_fixed and not near_vwap:
            continue

        # Update VWAP zone price.
        if has_vwap:
            zones["VWAP"].price = float(dc.post_ib_vwaps[j])

        # Check each level for entry.
        for name, zone in zones.items():
            level_cap = (max_per_level_map or {}).get(name, max_per_level)
            if zone.in_zone or ec[name] >= level_cap:
                continue
            if not zone.update(pj):
                continue

            # Precompute values used by both filters and factors.
            et_mins = int(arrays.et_mins[gi])
            range_30m = float(arrays.range_30m_pts[gi])

            # Suppress entries during weak time windows (13:30-14:00 ET).
            suppressed = 810 <= et_mins < 840
            if not suppressed and extra_suppressed:
                suppressed = any(ws <= et_mins < we for ws, we in extra_suppressed)
            if suppressed:
                zone.reset()
                continue

            # Vol filter: skip dead markets (matches live bot).
            if pj > 0 and range_30m / pj < vol_filter_pct:
                zone.reset()
                continue

            # Same-level cooldown: skip if we traded this level recently.
            if level_cooldown_secs > 0 and name in level_last_exit_ns:
                cooldown_ns = np.int64(level_cooldown_secs) * 1_000_000_000
                if ens - level_last_exit_ns[name] < cooldown_ns:
                    zone.reset()
                    continue

            # Max wins per level: stop trading a level likely to break.
            # Reset win count if no trade on this level for 30+ minutes.
            if name in level_last_exit_ns:
                gap_ns = ens - level_last_exit_ns[name]
                if gap_ns > 1_800_000_000_000:  # 30 minutes
                    level_wins[name] = 0
            if level_wins.get(name, 0) >= max_wins_per_level:
                zone.reset()
                continue

            # Zone entry fired. Compute factors and score.
            d = "up" if pj > zone.price else "down"

            # Post-loss direction filter: skip if this combo just lost.
            if no_repeat_loss_combo and (name, d) in lost_combos:
                zone.reset()
                continue

            # Approach speed filter: skip fast approaches (momentum through level).
            if max_approach_speed > 0 and float(arrays.approach_speed[gi]) > max_approach_speed:
                zone.reset()
                continue
            fac = EntryFactors(
                level=name, direction=d, entry_count=ec[name] + 1,
                et_mins=et_mins,
                tick_rate=float(arrays.tick_rates[gi]),
                session_move=float(arrays.session_move[gi]),
                range_30m=range_30m,
                approach_speed=float(arrays.approach_speed[gi]),
                tick_density=float(arrays.tick_density[gi]),
            )

            if weights is not None:
                sc = score_entry(fac, weights, cw, cl)
                if sc < min_score:
                    zone.reset()
                    continue

            # Evaluate trade outcome.
            tgt = target_fn(name)
            stp = stop_fn(name) if stop_fn is not None else stop_pts
            out, exit_idx, pnl_pts = evaluate_bot_trade(
                gi, zone.price, d,
                ft, fp, tgt, stp, timeout_secs, eod,
            )
            pnl_usd = pnl_pts * MULTIPLIER

            # Record trade.
            ec[name] += 1
            trades.append(TradeRecord(
                dc.date, name, d, ec[name], out, pnl_usd, fac,
                entry_idx=gi, exit_idx=exit_idx, entry_ns=ens,
            ))

            # Update state.
            pos_exit_idx = exit_idx
            zone.reset()
            dpnl += pnl_usd
            level_last_exit_ns[name] = int(ft[exit_idx])
            if pnl_usd >= 0:
                cw += 1; cl = 0; dcons = 0
                level_wins[name] = level_wins.get(name, 0) + 1
            else:
                cw = 0; cl += 1; dcons += 1
                lost_combos.add((name, d))
                level_wins[name] = 0  # reset consecutive wins on loss
            if dpnl <= -daily_loss:
                stopped = True
            if dcons >= max_consec:
                stopped = True
            break  # 1 position at a time

    return trades, (cw, cl)
