"""
bot_frequency_backtest.py — Maximize trade frequency within 1-position constraint.

The bot's $/day is limited by how many trades it can take (currently ~5/day).
The human app's theoretical $26-40/day comes from 6-12 signals/day. The gap
is the 1-position constraint + zone exit threshold (15 pts).

Tests:
- Exit threshold: how far price must move to reset the zone (9-20 pts)
- Entry threshold: how close to the line (1-3 pts)
- Max entries per level: how many re-entries per level per day
- Timeout: shorter timeout frees the slot faster
- All with T8/S25, scoring, $150/3 risk, 1-position constraint

Walk-forward validated. Optimizes for $/day.

Usage:
    python -u bot_frequency_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    _run_zone_numpy,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, evaluate_bot_trade
from config import BOT_EOD_FLATTEN_BUFFER_MIN
from bot_pct_backtest import (
    BotEntry,
    fit_bot_weights,
    score_bot_entry,
    precompute_pct_outcomes,
)
from score_optimizer import compute_tick_rate, EnrichedAlert, Weights
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS

import pandas as pd

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY COMPUTATION WITH VARIABLE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FlexEntry:
    idx: int  # index into the arrays for this day
    global_idx: int
    level: str
    direction: str
    entry_count: int
    entry_price: float
    line_price: float
    entry_ns: int
    now_et: datetime.time | None = None
    tick_rate: float | None = None
    session_move_pct: float | None = None
    range_30m: float | None = None


def compute_entries(
    dc: DayCache,
    entry_threshold: float,
    exit_threshold: float,
    include_vwap: bool = True,
) -> list[FlexEntry]:
    """Compute zone entries with flexible thresholds and enrich with scoring factors."""
    prices = dc.post_ib_prices
    n = len(prices)
    first_price = float(prices[0])

    all_entries: list[tuple[int, str, int, float]] = []
    levels = [
        ("IBH", np.full(n, dc.ibh)),
        ("IBL", np.full(n, dc.ibl)),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo)),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi)),
    ]
    if include_vwap:
        levels.append(("VWAP", dc.post_ib_vwaps))

    for level_name, level_arr in levels:
        use_current = level_name == "VWAP"
        entries = _run_zone_numpy(
            prices, level_arr, entry_threshold, exit_threshold,
            use_current_exit=use_current,
        )
        for local_idx, ec, rp in entries:
            all_entries.append(
                (dc.post_ib_start_idx + local_idx, level_name, ec, rp)
            )

    all_entries.sort(key=lambda x: x[0])

    result = []
    for i, (gidx, level_name, ec, rp) in enumerate(all_entries):
        entry_price = float(dc.full_prices[gidx])
        entry_ns = int(dc.full_ts_ns[gidx])
        direction = "up" if entry_price > rp else "down"

        ts_pd = pd.Timestamp(entry_ns, unit="ns", tz="UTC").tz_convert(_ET)
        now_et = ts_pd.time()
        tick_rate = compute_tick_rate(dc.full_df, ts_pd)
        session_move = entry_price - first_price
        session_move_pct = session_move / first_price * 100 if first_price > 0 else 0

        # 30-min range
        window_ns = np.int64(30 * 60 * 1_000_000_000)
        ws_ns = dc.full_ts_ns[gidx] - window_ns
        ws_idx = int(np.searchsorted(dc.full_ts_ns, ws_ns, side="left"))
        if ws_idx < gidx:
            wp = dc.full_prices[ws_idx : gidx + 1]
            range_30m = float(np.max(wp) - np.min(wp))
        else:
            range_30m = None

        result.append(FlexEntry(
            idx=i, global_idx=gidx, level=level_name, direction=direction,
            entry_count=ec, entry_price=entry_price, line_price=rp,
            entry_ns=entry_ns, now_et=now_et, tick_rate=tick_rate,
            session_move_pct=session_move_pct, range_30m=range_30m,
        ))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOME COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Outcomes:
    outcome: list[str]
    exit_ns: list[int]
    pnl_usd: list[float]


def compute_outcomes(
    entries: list[FlexEntry],
    dc: DayCache,
    target_pts: float,
    stop_pts: float,
    timeout_secs: int = 900,
) -> Outcomes:
    eod_ns = _eod_cutoff_ns(dc.date)
    outcomes = []
    exit_ns_list = []
    pnl_list = []
    for e in entries:
        out, eidx, pnl = evaluate_bot_trade(
            e.global_idx, e.line_price, e.direction,
            dc.full_ts_ns, dc.full_prices,
            target_pts, stop_pts, timeout_secs, eod_ns,
        )
        outcomes.append(out)
        exit_ns_list.append(int(dc.full_ts_ns[eidx]))
        pnl_list.append(pnl * MULTIPLIER)
    return Outcomes(outcomes, exit_ns_list, pnl_list)


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    direction: str
    outcome: str
    pnl_usd: float


def replay(
    days, entries_by_date, outcomes_by_date,
    weights, min_score, max_entries_per_level,
    daily_loss_usd, max_consec_losses,
):
    trades = []
    cw, cl = 0, 0
    for date in days:
        entries = entries_by_date.get(date)
        oc = outcomes_by_date.get(date)
        if not entries or not oc:
            continue
        eod_ns = _eod_cutoff_ns(date)
        pos_exit_ns = 0
        daily_pnl = 0.0
        daily_consec = 0
        stopped = False
        lc = {}
        for i, e in enumerate(entries):
            if stopped:
                break
            if e.entry_ns >= eod_ns:
                break
            if e.entry_ns < pos_exit_ns:
                continue
            lv = lc.get(e.level, 0)
            if lv >= max_entries_per_level:
                continue
            # Vol filter
            r30_pct = (
                e.range_30m / e.entry_price * 100
                if e.range_30m is not None and e.entry_price > 0
                else None
            )
            if r30_pct is not None and r30_pct < 0.15:
                continue
            score = score_bot_entry(e, weights, cw, cl)
            if score < min_score:
                continue

            outcome = oc.outcome[i]
            pnl = oc.pnl_usd[i]
            pos_exit_ns = oc.exit_ns[i]
            lc[e.level] = lv + 1
            trades.append(Trade(date, e.level, e.direction, outcome, pnl))
            daily_pnl += pnl
            if pnl < 0:
                daily_consec += 1; cw = 0; cl += 1
            else:
                daily_consec = 0; cl = 0; cw += 1
            if daily_loss_usd and daily_pnl <= -daily_loss_usd:
                stopped = True
            if max_consec_losses and daily_consec >= max_consec_losses:
                stopped = True
    return trades


def fmt(trades, n_days, label=""):
    if not trades:
        return f"  {label:>55s}  no trades"
    w = sum(1 for t in trades if t.outcome == "win")
    l = sum(1 for t in trades if t.outcome == "loss")
    o = len(trades) - w - l
    d = w + l
    wr = w / d * 100 if d else 0
    pnl = sum(t.pnl_usd for t in trades)
    ppd = pnl / n_days
    eq = STARTING_BALANCE
    peak = eq
    dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return (
        f"  {label:>55s}  {len(trades):>4} ({len(trades)/n_days:.1f}/d) "
        f"{w}W/{l}L/{o}O {wr:>5.1f}%  "
        f"PnL {pnl:>+8,.0f} ({ppd:>+5.1f}/d)  DD {dd:>5,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def _precompute_scoring_factors(dc: DayCache) -> dict:
    """Precompute tick_rate, session_move_pct, range_30m for every post-IB tick.

    Returns a dict with arrays indexed by global_idx, so any entry threshold
    combo can look up scoring factors in O(1) without recomputing.
    """
    first_price = float(dc.post_ib_prices[0])
    n = len(dc.full_prices)
    start = dc.post_ib_start_idx

    # Precompute tick counts in 3-min windows using a sliding approach.
    # tick_rate[i] = number of ticks in [ts[i] - 3min, ts[i]] / 3
    tick_rates = np.zeros(n, dtype=np.float64)
    left = start
    for right in range(start, n):
        window_start = dc.full_ts_ns[right] - np.int64(3 * 60 * 1_000_000_000)
        while left < right and dc.full_ts_ns[left] < window_start:
            left += 1
        tick_rates[right] = (right - left) / 3.0

    # Precompute 30-min range using sliding window.
    # This is approximate: we track min/max in a deque-like structure.
    range_30m = np.zeros(n, dtype=np.float64)
    from collections import deque
    win_ns = np.int64(30 * 60 * 1_000_000_000)
    for i in range(start, n):
        ws = dc.full_ts_ns[i] - win_ns
        ws_idx = int(np.searchsorted(dc.full_ts_ns, ws, side="left"))
        if ws_idx < i:
            wp = dc.full_prices[ws_idx : i + 1]
            range_30m[i] = float(np.max(wp) - np.min(wp))

    return {
        "tick_rates": tick_rates,
        "range_30m": range_30m,
        "first_price": first_price,
    }


def compute_entries_fast(
    dc: DayCache,
    entry_threshold: float,
    exit_threshold: float,
    factors: dict,
    include_vwap: bool = True,
) -> list[FlexEntry]:
    """Like compute_entries but uses precomputed scoring factors."""
    prices = dc.post_ib_prices
    n = len(prices)
    first_price = factors["first_price"]
    tick_rates = factors["tick_rates"]
    range_30m_arr = factors["range_30m"]

    all_raw: list[tuple[int, str, int, float]] = []
    levels = [
        ("IBH", np.full(n, dc.ibh)),
        ("IBL", np.full(n, dc.ibl)),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo)),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi)),
    ]
    if include_vwap:
        levels.append(("VWAP", dc.post_ib_vwaps))

    for level_name, level_arr in levels:
        use_current = level_name == "VWAP"
        entries = _run_zone_numpy(
            prices, level_arr, entry_threshold, exit_threshold,
            use_current_exit=use_current,
        )
        for local_idx, ec, rp in entries:
            all_raw.append(
                (dc.post_ib_start_idx + local_idx, level_name, ec, rp)
            )
    all_raw.sort(key=lambda x: x[0])

    result = []
    for i, (gidx, level_name, ec, rp) in enumerate(all_raw):
        ep = float(dc.full_prices[gidx])
        ens = int(dc.full_ts_ns[gidx])
        direction = "up" if ep > rp else "down"
        ts_pd = pd.Timestamp(ens, unit="ns", tz="UTC").tz_convert(_ET)
        sm = ep - first_price
        sm_pct = sm / first_price * 100 if first_price > 0 else 0

        result.append(FlexEntry(
            idx=i, global_idx=gidx, level=level_name, direction=direction,
            entry_count=ec, entry_price=ep, line_price=rp, entry_ns=ens,
            now_et=ts_pd.time(),
            tick_rate=float(tick_rates[gidx]),
            session_move_pct=sm_pct,
            range_30m=float(range_30m_arr[gidx]) if range_30m_arr[gidx] > 0 else None,
        ))
    return result


def main():
    t0 = time.time()
    print("=" * 100)
    print("  FREQUENCY BACKTEST — Maximize trades/day via zone thresholds")
    print("=" * 100)

    days = load_cached_days()
    day_caches = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except:
            pass
    valid_days = sorted(day_caches.keys())
    N = len(valid_days)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    # Precompute scoring factors once per day.
    print(f"  Precomputing scoring factors...", flush=True)
    t1 = time.time()
    factors_by_date = {}
    for i, date in enumerate(valid_days):
        factors_by_date[date] = _precompute_scoring_factors(day_caches[date])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N}...", flush=True)
    print(f"  Factors done in {time.time()-t1:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: In-sample sweep
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 1: Entry/exit threshold sweep (T8/S25, in-sample)")
    print("=" * 100)

    ENTRY_GRID = [1.0, 2.0, 3.0]
    EXIT_GRID = [9.0, 10.0, 12.0, 15.0, 20.0]
    MAX_ENTRY_GRID = [3, 5, 8]
    TIMEOUT_GRID = [300, 600, 900]

    configs = {}
    print(f"\n  Precomputing entries and outcomes...", flush=True)
    t2 = time.time()
    for et in ENTRY_GRID:
        for ex in EXIT_GRID:
            # Compute entries once per (entry, exit) combo.
            ebd_base = {}
            for date in valid_days:
                ebd_base[date] = compute_entries_fast(
                    day_caches[date], et, ex, factors_by_date[date]
                )
            for timeout in TIMEOUT_GRID:
                key = (et, ex, timeout)
                obd = {}
                for date in valid_days:
                    obd[date] = compute_outcomes(
                        ebd_base[date], day_caches[date], 8.0, 25.0, timeout
                    )
                configs[key] = (ebd_base, obd)
            print(f"    entry={et} exit={ex} done ({time.time()-t2:.0f}s)", flush=True)

    # Train weights.
    ref_entries = configs[(1.0, 15.0, 900)][0]
    ref_obd = configs[(1.0, 15.0, 900)][1]
    all_eo = []
    for date in valid_days:
        entries = ref_entries[date]
        oc = ref_obd[date]
        for i, e in enumerate(entries):
            all_eo.append((e, oc.outcome[i]))
    w_full = fit_bot_weights(all_eo)
    print(f"  Total precompute: {time.time()-t2:.0f}s\n")

    # Sweep.
    results = []
    for key, (ebd, obd) in configs.items():
        et, ex, timeout = key
        for min_score in [-1, 0]:
            for max_e in MAX_ENTRY_GRID:
                trades = replay(
                    valid_days, ebd, obd, w_full,
                    min_score, max_e, 150.0, 3,
                )
                pnl = sum(t.pnl_usd for t in trades)
                ppd = pnl / N
                label = f"ent={et:.0f} exit={ex:.0f} to={timeout}s score>={min_score} max={max_e}"
                results.append((label, trades, ppd, key, min_score, max_e))

    results.sort(key=lambda x: x[2], reverse=True)
    print(f"  Top 25 by $/day (in-sample):\n")
    for label, trades, ppd, *_ in results[:25]:
        print(fmt(trades, N, label))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward on top configs
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 2: Walk-forward validation")
    print("=" * 100)

    # Pick top 12 distinct (entry, exit, timeout) combos.
    seen = set()
    wf_keys = []
    for _, _, _, key, _, _ in results:
        if key not in seen:
            seen.add(key)
            wf_keys.append(key)
        if len(wf_keys) >= 12:
            break
    # Always include current config.
    current = (1.0, 15.0, 900)
    if current not in seen:
        wf_keys.append(current)

    print(f"\n  Validating {len(wf_keys)} configs OOS...", flush=True)

    oos_by_config: dict[str, list[Trade]] = {}
    oos_days = 0
    k = INITIAL_TRAIN_DAYS
    windows = 0

    while k < N:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days:
            break
        windows += 1
        oos_days += len(test_days)

        # Train weights on training data.
        train_eo = []
        for d in train_days:
            entries = ref_entries.get(d, [])
            oc = ref_obd.get(d)
            if oc:
                for i, e in enumerate(entries):
                    if i < len(oc.outcome):
                        train_eo.append((e, oc.outcome[i]))
        wt = fit_bot_weights(train_eo)

        for key in wf_keys:
            ebd, obd = configs[key]
            et, ex, timeout = key
            for min_score in [-1, 0]:
                for max_e in [3, 5, 8]:
                    label = f"ent={et:.0f} exit={ex:.0f} to={timeout}s score>={min_score} max={max_e}"
                    trades = replay(
                        test_days, ebd, obd, wt,
                        min_score, max_e, 150.0, 3,
                    )
                    oos_by_config.setdefault(label, []).extend(trades)

        k += STEP_DAYS

    print(f"  {windows} windows, {oos_days} OOS days\n")

    oos_results = []
    for cfg, trades in oos_by_config.items():
        pnl = sum(t.pnl_usd for t in trades)
        ppd = pnl / oos_days if oos_days else 0
        oos_results.append((cfg, trades, ppd))
    oos_results.sort(key=lambda x: x[2], reverse=True)

    print(f"  Top 20 by $/day (OOS walk-forward):\n")
    for cfg, trades, ppd in oos_results[:20]:
        print(fmt(trades, oos_days, cfg))

    # Current config for comparison.
    current_label = "ent=1 exit=15 to=900s score>=-1 max=3"
    if current_label in oos_by_config:
        trades = oos_by_config[current_label]
        print(f"\n  Current live config:")
        print(fmt(trades, oos_days, current_label))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 3: Recent 60 days (top 10)")
    print("=" * 100)

    recent = valid_days[-60:]
    rn = len(recent)
    pre = [d for d in valid_days if d < recent[0]]
    pre_eo = []
    for d in pre:
        entries = ref_entries.get(d, [])
        oc = ref_obd.get(d)
        if oc:
            for i, e in enumerate(entries):
                if i < len(oc.outcome):
                    pre_eo.append((e, oc.outcome[i]))
    w_recent = fit_bot_weights(pre_eo)

    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")

    for cfg, _, _ in oos_results[:10]:
        # Parse config.
        parts = cfg.split()
        et = float(parts[0].split("=")[1])
        ex = float(parts[1].split("=")[1])
        timeout = int(parts[2].split("=")[1].rstrip("s"))
        min_score = int(parts[3].split(">=")[1])
        max_e = int(parts[4].split("=")[1])
        key = (et, ex, timeout)

        if key not in configs:
            continue
        ebd, obd = configs[key]
        trades = replay(
            recent, ebd, obd, w_recent,
            min_score, max_e, 150.0, 3,
        )
        print(fmt(trades, rn, f"RECENT: {cfg}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
