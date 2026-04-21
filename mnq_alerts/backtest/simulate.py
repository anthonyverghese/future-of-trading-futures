"""Tick-by-tick simulation with integrated scoring and constraints.

Uses evaluate_bot_trade for trade resolution. Iterates all ticks but
skips quickly when no level is nearby.

Handles:
- 1-position constraint
- Scoring filter (skipped entries don't activate zone)
- Streak tracking across days
- Risk limits ($150 daily loss, 3 consecutive losses)
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

    # Level configs.
    fixed_levels = [
        ("IBH", dc.ibh, False),
        ("IBL", dc.ibl, False),
        ("FIB_EXT_HI_1.272", dc.fib_hi, False),
        ("FIB_EXT_LO_1.272", dc.fib_lo, False),
    ]
    has_vwap = hasattr(dc, "post_ib_vwaps") and dc.post_ib_vwaps is not None

    # Create zones.
    zones = {name: zone_factory(name, price, drifts) for name, price, drifts in fixed_levels}
    if has_vwap:
        zones["VWAP"] = zone_factory("VWAP", float(dc.post_ib_vwaps[0]), True)
    ec = {name: 0 for name in zones}

    cw, cl = streak_state
    trades = []
    pos_exit_idx = -1
    dpnl = 0.0
    dcons = 0
    stopped = False

    # Precompute fixed level prices for fast distance check.
    fixed_prices = np.array([lv for _, lv, _ in fixed_levels])

    for j in range(n):
        gi = start + j
        ens = int(ft[gi])
        if ens >= eod:
            break
        if stopped:
            continue
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

            # Evaluate trade outcome.
            tgt = target_fn(name)
            out, exit_idx, pnl_pts = evaluate_bot_trade(
                gi, zone.price, d,
                ft, fp, tgt, stop_pts, timeout_secs, eod,
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
            if pnl_usd >= 0:
                cw += 1; cl = 0; dcons = 0
            else:
                cw = 0; cl += 1; dcons += 1
            if dpnl <= -daily_loss:
                stopped = True
            if dcons >= max_consec:
                stopped = True
            break  # 1 position at a time

    return trades, (cw, cl)
