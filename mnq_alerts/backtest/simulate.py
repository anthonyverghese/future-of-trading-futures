"""Tick-by-tick simulation with integrated scoring and constraints.

The ONE simulation function. Zone class is pluggable. Handles:
- 1-position constraint (only one trade open at a time)
- Scoring filter (skipped entries don't activate the zone)
- Streak tracking across days
- Risk limits ($150 daily loss, 3 consecutive losses)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

from .data import DayArrays
from .scoring import EntryFactors, score_entry
from .evaluate import FEE_PTS, MULTIPLIER

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from targeted_backtest import DayCache
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


def simulate_day(
    dc: DayCache,
    arrays: DayArrays,
    zone_factory,     # callable(name, price, drifts) → zone instance
    target_fn,        # callable(level_name) → target_pts
    stop_pts: float,
    max_per_level: int,
    weights: dict | None = None,
    min_score: int = -99,
    streak_state: tuple[int, int] = (0, 0),
    daily_loss: float = 150.0,
    max_consec: int = 3,
    timeout_secs: int = 900,
) -> tuple[list[TradeRecord], tuple[int, int]]:
    """Simulate one day. Returns (trades, (cw, cl))."""
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    fp = dc.full_prices
    ft = dc.full_ts_ns
    eod = _eod_cutoff_ns(dc.date)

    # Create zones for each level.
    lvls = [
        ("IBH", dc.ibh, False),
        ("IBL", dc.ibl, False),
        ("FIB_EXT_HI_1.272", dc.fib_hi, False),
        ("FIB_EXT_LO_1.272", dc.fib_lo, False),
    ]
    has_vwap = hasattr(dc, "post_ib_vwaps") and dc.post_ib_vwaps is not None

    zones = {name: zone_factory(name, price, drifts) for name, price, drifts in lvls}
    if has_vwap:
        zones["VWAP"] = zone_factory("VWAP", float(dc.post_ib_vwaps[0]), True)
    ec = {name: 0 for name in zones}

    cw, cl = streak_state
    trades = []
    in_trade = False
    t_lv = ""
    t_dir = ""
    t_lp = 0.0
    t_tp = 0.0
    t_sl = 0.0
    t_to = 0
    t_ec = 0
    t_fac = None
    dpnl = 0.0
    dcons = 0
    stopped = False

    for j in range(n):
        gi = start + j
        pj = float(prices[j])
        ens = int(ft[gi])

        # Update VWAP zone price (drifts).
        if has_vwap:
            zones["VWAP"].price = float(dc.post_ib_vwaps[j])

        if ens >= eod:
            if in_trade:
                pnl = (pj - t_lp - FEE_PTS if t_dir == "up" else t_lp - pj - FEE_PTS) * MULTIPLIER
                trades.append(TradeRecord(dc.date, t_lv, t_dir, t_ec, "timeout", pnl, t_fac))
                zones[t_lv].reset()
                in_trade = False
                if pnl >= 0: cw += 1; cl = 0
                else: cw = 0; cl += 1
            break

        # If in a trade, check target/stop/timeout.
        if in_trade:
            closed = False
            outcome = ""
            pnl = 0.0
            if ens > t_to:
                pnl = (pj - t_lp - FEE_PTS if t_dir == "up" else t_lp - pj - FEE_PTS) * MULTIPLIER
                outcome = "timeout"
                closed = True
            elif t_dir == "up":
                if pj >= t_tp:
                    pnl = (t_tp - t_lp - FEE_PTS) * MULTIPLIER
                    outcome = "win"
                    closed = True
                elif pj <= t_sl:
                    pnl = (-(t_lp - t_sl + FEE_PTS)) * MULTIPLIER
                    outcome = "loss"
                    closed = True
            else:
                if pj <= t_tp:
                    pnl = (t_lp - t_tp - FEE_PTS) * MULTIPLIER
                    outcome = "win"
                    closed = True
                elif pj >= t_sl:
                    pnl = (-(t_sl - t_lp + FEE_PTS)) * MULTIPLIER
                    outcome = "loss"
                    closed = True

            if closed:
                trades.append(TradeRecord(dc.date, t_lv, t_dir, t_ec, outcome, pnl, t_fac))
                zones[t_lv].reset()
                in_trade = False
                dpnl += pnl
                if pnl >= 0: cw += 1; cl = 0; dcons = 0
                else: cw = 0; cl += 1; dcons += 1
                if dpnl <= -daily_loss: stopped = True
                if dcons >= max_consec: stopped = True
            continue

        if stopped:
            continue

        # Check each level for entry.
        for name, zone in zones.items():
            if zone.in_zone or ec[name] >= max_per_level:
                continue
            if not zone.update(pj):
                continue

            # Zone entry fired. Compute factors and score.
            d = "up" if pj > zone.price else "down"
            fac = EntryFactors(
                level=name, direction=d, entry_count=ec[name] + 1,
                et_mins=int(arrays.et_mins[gi]),
                tick_rate=float(arrays.tick_rates[gi]),
                session_move=float(arrays.session_move[gi]),
                range_30m=float(arrays.range_30m_pts[gi]),
                approach_speed=float(arrays.approach_speed[gi]),
                tick_density=float(arrays.tick_density[gi]),
            )

            if weights is not None:
                sc = score_entry(fac, weights, cw, cl)
                if sc < min_score:
                    zone.reset()
                    continue

            # Open trade.
            ec[name] += 1
            tgt = target_fn(name)
            if d == "up":
                tp = zone.price + tgt
                sl = zone.price - stop_pts
            else:
                tp = zone.price - tgt
                sl = zone.price + stop_pts
            tp = round(tp * 4) / 4
            sl = round(sl * 4) / 4

            in_trade = True
            t_lv = name
            t_dir = d
            t_lp = zone.price
            t_tp = tp
            t_sl = sl
            t_to = ens + timeout_secs * 1_000_000_000
            t_ec = ec[name]
            t_fac = fac
            break  # 1 position at a time

    return trades, (cw, cl)
