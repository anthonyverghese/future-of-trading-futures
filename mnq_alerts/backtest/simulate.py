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
    direction_filter: dict[str, str] | None = None,
    global_cooldown_after_loss_secs: int = 0,
    no_reverse_after_loss: bool = False,
    max_tick_rate: float = 0.0,
    score_fn=None,
    trend_filter: str | None = None,
    vwap_filter: str | None = None,
    confirmation_bounce_pts: float = 0.0,
    confirm_only_counter_trend: bool = False,
    split_budget: tuple[float, float] | None = None,
    split_budget_cutoff_mins: int = 780,
    momentum_max: float = 0.0,
    momentum_lookback_ticks: int = 1000,
    direction_caps: dict[tuple[str, str], int] | None = None,
    suppress_1330: bool = True,
    adaptive_caps: bool = False,
    adaptive_caps_duration_mins: int = 30,
    adaptive_caps_restore_wins: int = 3,
) -> tuple[list[TradeRecord], tuple[int, int]]:
    """Simulate one day. Returns (trades, (cw, cl)).

    trend_filter: 'block' (block counter-trend), 'ext_only' (counter-trend
      only on extensions), 'halve' (halve caps for counter-trend).
    vwap_filter: same options as trend_filter but based on price vs VWAP.
    confirmation_bounce_pts: require price to bounce this many pts from
      level before re-approaching. 0 = disabled.
    confirm_only_counter_trend: only require confirmation on counter-trend.
    split_budget: (morning_limit, afternoon_limit) in USD. Replaces daily_loss.
    split_budget_cutoff_mins: ET minutes for morning/afternoon split (default 780 = 13:00).

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
    ec_dir: dict[tuple[str, str], int] = {}  # (level, direction) → count

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
    # Global cooldown: ns timestamp when all-level cooldown expires.
    global_cooldown_until_ns: int = 0
    # Last loss per level: maps level → direction of last loss (for no-reverse filter).
    last_loss_direction: dict[str, str] = {}
    # Split budget tracking.
    morning_pnl = 0.0
    afternoon_pnl = 0.0
    # Adaptive caps: halve caps for first N min after IB, restore on
    # consecutive wins from start, extend on loss.
    ac_restored = not adaptive_caps  # True = full caps (disabled or restored)
    ac_until_et = 630 + adaptive_caps_duration_mins  # ET mins when half-caps expire
    ac_accepted = 0
    ac_any_loss = False
    morning_stopped = False
    afternoon_stopped = False
    # Confirmation bounce: track max distance from level after zone entry.
    # Maps level name → max distance seen since zone entered.
    level_bounce_dist: dict[str, float] = {}

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
        # Widen the distance check if any zone is observing a confirmation bounce.
        obs_dist = max(
            (getattr(z, '_bounce_pts', 0) + 1.0
             for z in zones.values() if getattr(z, '_observing', False)),
            default=0
        )
        skip_dist = max(1.0, obs_dist)
        near_fixed = np.any(np.abs(fixed_prices - pj) <= skip_dist)
        near_vwap = has_vwap and abs(pj - float(dc.post_ib_vwaps[j])) <= skip_dist
        if not near_fixed and not near_vwap:
            continue

        # Update VWAP zone price.
        if has_vwap:
            zones["VWAP"].price = float(dc.post_ib_vwaps[j])

        # Update any zones that are observing a confirmation bounce.
        for zone in zones.values():
            if getattr(zone, '_observing', False):
                zone.update(pj)

        # Check each level for entry.
        for name, zone in zones.items():
            level_cap = (max_per_level_map or {}).get(name, max_per_level)
            # Adaptive caps: halve caps until restored.
            if not ac_restored:
                et_now = int(arrays.et_mins[gi]) if gi < len(arrays.et_mins) else 960
                if et_now < ac_until_et:
                    level_cap = max(1, level_cap // 2)
                else:
                    ac_restored = True
            if zone.in_zone or ec[name] >= level_cap:
                continue
            if not zone.update(pj):
                continue

            # Precompute values used by both filters and factors.
            et_mins = int(arrays.et_mins[gi])
            range_30m = float(arrays.range_30m_pts[gi])

            # Suppress entries during weak time windows (13:30-14:00 ET).
            suppressed = suppress_1330 and 810 <= et_mins < 840
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

            # Per-level direction filter: only allow specified direction.
            if direction_filter and name in direction_filter:
                if direction_filter[name] != d:
                    zone.reset()
                    continue

            # Per-direction cap: limit entries per (level, direction).
            if direction_caps and (name, d) in direction_caps:
                dir_count = ec_dir.get((name, d), 0)
                if dir_count >= direction_caps[(name, d)]:
                    zone.reset()
                    continue

            # Trend filter: compare trade direction to session move direction.
            if trend_filter is not None:
                session_mv = float(arrays.session_move[gi])
                trend_dir = "down" if session_mv < 0 else "up"
                is_counter = (d != trend_dir)
                is_interior = name in ("FIB_0.236", "FIB_0.618", "FIB_0.764")
                if is_counter:
                    if trend_filter == "block":
                        zone.reset()
                        continue
                    elif trend_filter == "ext_only" and is_interior:
                        zone.reset()
                        continue
                    elif trend_filter == "halve":
                        # Halve the cap for counter-trend — check manually.
                        lv_cap = (max_per_level_map or {}).get(name, max_per_level)
                        if ec[name] >= max(1, lv_cap // 2):
                            zone.reset()
                            continue

            # VWAP filter: compare trade direction to price vs VWAP.
            if vwap_filter is not None and has_vwap:
                vwap_val = float(dc.post_ib_vwaps[j])
                vwap_dir = "up" if pj > vwap_val else "down"
                is_counter_vwap = (d != vwap_dir)
                is_interior = name in ("FIB_0.236", "FIB_0.618", "FIB_0.764")
                if is_counter_vwap:
                    if vwap_filter == "block":
                        zone.reset()
                        continue
                    elif vwap_filter == "ext_only" and is_interior:
                        zone.reset()
                        continue
                    elif vwap_filter == "halve":
                        lv_cap = (max_per_level_map or {}).get(name, max_per_level)
                        if ec[name] >= max(1, lv_cap // 2):
                            zone.reset()
                            continue

            # Split budget: separate morning/afternoon loss limits.
            if split_budget is not None:
                et_mins_now = int(arrays.et_mins[gi])
                if et_mins_now < split_budget_cutoff_mins:
                    if morning_stopped:
                        zone.reset()
                        continue
                else:
                    if afternoon_stopped:
                        zone.reset()
                        continue

            # Global cooldown: skip all entries for N seconds after any loss.
            if global_cooldown_after_loss_secs > 0 and ens < global_cooldown_until_ns:
                zone.reset()
                continue

            # No reverse after loss: if this level just lost in one direction,
            # don't trade the opposite direction (level is broken, not bouncing).
            if no_reverse_after_loss and name in last_loss_direction:
                if last_loss_direction[name] != d:
                    zone.reset()
                    continue

            # Tick rate filter: skip entries in very high tick rate (momentum) markets.
            if max_tick_rate > 0 and float(arrays.tick_rates[gi]) > max_tick_rate:
                zone.reset()
                continue

            # Post-loss direction filter: skip if this combo just lost.
            if no_repeat_loss_combo and (name, d) in lost_combos:
                zone.reset()
                continue

            # Approach speed filter: skip fast approaches (momentum through level).
            if max_approach_speed > 0 and float(arrays.approach_speed[gi]) > max_approach_speed:
                zone.reset()
                continue

            # Momentum filter: skip entries where price moved > N pts
            # in the trade direction over the last ~5 min (1000 ticks).
            # "With momentum" means price is blasting through the level,
            # not bouncing. 0 = disabled. Only applied when enough
            # history exists (matches live bot allowing trades at startup).
            if momentum_max > 0 and gi >= start + momentum_lookback_ticks:
                prev_idx = gi - momentum_lookback_ticks
                mom = float(fp[gi]) - float(fp[prev_idx])
                if d == "down":
                    mom = -mom
                if mom > momentum_max:
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

            # Custom score function (e.g., bot_entry_score).
            if score_fn is not None:
                sc = score_fn(fac)
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
            ec_dir[(name, d)] = ec_dir.get((name, d), 0) + 1
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
                # Clear no-reverse block if this level won (level is working).
                last_loss_direction.pop(name, None)
                # Adaptive caps: restore on N consecutive wins from start.
                if not ac_restored:
                    ac_accepted += 1
                    if ac_accepted >= adaptive_caps_restore_wins and not ac_any_loss:
                        ac_restored = True
            else:
                cw = 0; cl += 1; dcons += 1
                lost_combos.add((name, d))
                level_wins[name] = 0  # reset consecutive wins on loss
                # Adaptive caps: extend window on loss.
                if not ac_restored:
                    ac_any_loss = True
                    ac_accepted += 1
                    loss_et = int(arrays.et_mins[exit_idx]) if exit_idx < len(arrays.et_mins) else 960
                    ac_until_et = loss_et + adaptive_caps_duration_mins
                last_loss_direction[name] = d
                if global_cooldown_after_loss_secs > 0:
                    global_cooldown_until_ns = int(ft[exit_idx]) + \
                        global_cooldown_after_loss_secs * 1_000_000_000
            # Split budget: track morning/afternoon P&L separately.
            if split_budget is not None:
                exit_et_mins = int(arrays.et_mins[exit_idx]) if exit_idx < len(arrays.et_mins) else 960
                if exit_et_mins < split_budget_cutoff_mins:
                    morning_pnl += pnl_usd
                    if morning_pnl <= -split_budget[0]:
                        morning_stopped = True
                else:
                    afternoon_pnl += pnl_usd
                    if afternoon_pnl <= -split_budget[1]:
                        afternoon_stopped = True
                if morning_stopped and afternoon_stopped:
                    stopped = True
            else:
                if dpnl <= -daily_loss:
                    stopped = True
            if dcons >= max_consec:
                stopped = True
            break  # 1 position at a time

    return trades, (cw, cl)
