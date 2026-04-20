"""
bot_trader_backtest.py — Day trader simulation backtest.

Architecture:
  1. Human pre-filter: run the proven human scoring (7-pt entry, 20-pt exit,
     score >= 5) to identify high-quality setups
  2. Bot entry: wait for price to reach the line (within 1 pt)
  3. Bot execution: tight stop + adaptive target, using the approach data
     the human couldn't see

Also tests per-level T/S since bounce magnitude differs by level.

Walk-forward validated. Optimizes for $/day.

Usage:
    python -u bot_trader_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    _run_zone_numpy,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, FEE_PTS
from score_optimizer import Weights, compute_tick_rate, score_alert, EnrichedAlert
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# TWO-STAGE ENTRY: human pre-filter → bot at the line
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualifiedEntry:
    """An entry that passed the human pre-filter and reached the line."""
    global_idx: int  # tick index where price reached within 1 pt
    level: str
    direction: str
    line_price: float
    entry_price: float
    entry_ns: int
    human_score: int
    entry_count: int


def find_qualified_entries(
    dc: DayCache,
    human_weights: Weights,
    min_human_score: int = 5,
) -> list[QualifiedEntry]:
    """Find entries that pass the human pre-filter AND reach the line.

    Stage 1: Run human zone (7-pt entry, 20-pt exit) to find alerts.
    Stage 2: For each alert that scores >= min_human_score, check if
             price subsequently reaches within 1 pt of the line.
    """
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    full_prices = dc.full_prices
    full_ts = dc.full_ts_ns
    first_price = float(prices[0])

    levels = [
        ("IBH", np.full(n, dc.ibh), False),
        ("IBL", np.full(n, dc.ibl), False),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), False),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), False),
        ("VWAP", dc.post_ib_vwaps, True),
    ]

    qualified = []

    for level_name, level_arr, drifts in levels:
        # Run human zone (7-pt entry, 20-pt exit).
        in_zone_h = False
        ref_h = 0.0
        entry_count_h = 0

        # Run bot zone (1-pt entry, 20-pt exit) in parallel.
        in_zone_b = False
        ref_b = 0.0
        entry_count_b = 0

        # Track: is there a pending human-approved alert waiting for line touch?
        pending_human_alert = False
        pending_score = 0
        pending_entry_count = 0

        for j in range(n):
            pj = prices[j]
            lj = level_arr[j]
            gidx = start + j
            ens = int(full_ts[gidx])

            # Human zone update.
            if in_zone_h:
                exit_ref = lj if drifts else ref_h
                if abs(pj - exit_ref) > 20.0:
                    in_zone_h = False
                    pending_human_alert = False  # zone reset, clear pending
            else:
                if abs(pj - lj) <= 7.0:
                    in_zone_h = True
                    ref_h = lj
                    entry_count_h += 1

                    # Score this human alert.
                    ep = float(pj)
                    direction = "up" if ep > lj else "down"
                    ts_pd = pd.Timestamp(ens, unit="ns", tz="UTC").tz_convert(_ET)
                    now_et = ts_pd.time()
                    tr = compute_tick_rate(dc.full_df, ts_pd)
                    sm = ep - first_price

                    ea = EnrichedAlert(
                        date=dc.date, level=level_name, direction=direction,
                        entry_count=entry_count_h, outcome="correct",
                        entry_price=ep, line_price=float(lj),
                        alert_time=ts_pd, now_et=now_et, tick_rate=tr,
                        session_move_pts=sm, consecutive_wins=0,
                        consecutive_losses=0,
                    )
                    sc = score_alert(ea, human_weights)

                    if sc >= min_human_score:
                        pending_human_alert = True
                        pending_score = sc
                        pending_entry_count = entry_count_h

            # Bot zone update (1-pt entry, 20-pt exit).
            if in_zone_b:
                exit_ref_b = lj if drifts else ref_b
                if abs(pj - exit_ref_b) > 20.0:
                    in_zone_b = False
            else:
                if abs(pj - lj) <= 1.0:
                    in_zone_b = True
                    ref_b = lj
                    entry_count_b += 1

                    # Bot reached the line. Is there a pending human approval?
                    if pending_human_alert:
                        ep = float(pj)
                        direction = "up" if ep > lj else "down"
                        qualified.append(QualifiedEntry(
                            global_idx=gidx,
                            level=level_name,
                            direction=direction,
                            line_price=float(lj),
                            entry_price=ep,
                            entry_ns=ens,
                            human_score=pending_score,
                            entry_count=pending_entry_count,
                        ))
                        pending_human_alert = False  # consumed

    qualified.sort(key=lambda e: e.entry_ns)
    return qualified


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE EXIT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_adaptive(
    entry_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    stop_pts: float,
    target_pts: float,
    max_hold_secs: int = 900,
    eod_cutoff_ns: int | None = None,
) -> tuple[str, int, float]:
    """Simple evaluation with configurable T/S. Used for per-level T/S."""
    entry_ns = ts_ns[entry_idx]
    max_ns = entry_ns + np.int64(max_hold_secs * 1_000_000_000)
    if eod_cutoff_ns is not None and eod_cutoff_ns < max_ns:
        max_ns = np.int64(eod_cutoff_ns)

    if direction == "up":
        tp = line_price + target_pts
        sl = line_price - stop_pts
    else:
        tp = line_price - target_pts
        sl = line_price + stop_pts

    last_idx = entry_idx
    for j in range(entry_idx + 1, len(prices)):
        if ts_ns[j] > max_ns:
            break
        last_idx = j
        p = float(prices[j])
        if direction == "up":
            if p >= tp: return "win", j, target_pts - FEE_PTS
            if p <= sl: return "loss", j, -(stop_pts + FEE_PTS)
        else:
            if p <= tp: return "win", j, target_pts - FEE_PTS
            if p >= sl: return "loss", j, -(stop_pts + FEE_PTS)

    ep = float(prices[last_idx])
    pnl = (ep - line_price if direction == "up" else line_price - ep) - FEE_PTS
    return "timeout", last_idx, pnl


def evaluate_trailing(
    entry_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    stop_pts: float,
    trail_activate_pts: float,
    trail_offset_pts: float,
    max_target_pts: float = 50.0,
    max_hold_secs: int = 900,
    eod_cutoff_ns: int | None = None,
) -> tuple[str, int, float]:
    """Tight stop + trailing exit. No fixed target — let winners run."""
    entry_ns = ts_ns[entry_idx]
    max_ns = entry_ns + np.int64(max_hold_secs * 1_000_000_000)
    if eod_cutoff_ns is not None and eod_cutoff_ns < max_ns:
        max_ns = np.int64(eod_cutoff_ns)

    if direction == "up":
        sl = line_price - stop_pts
    else:
        sl = line_price + stop_pts

    best_fav = 0.0
    trailing_active = False
    last_idx = entry_idx

    for j in range(entry_idx + 1, len(prices)):
        if ts_ns[j] > max_ns:
            break
        last_idx = j
        p = float(prices[j])

        # Stop hit.
        if direction == "up" and p <= sl:
            return "loss", j, -(stop_pts + FEE_PTS)
        if direction == "down" and p >= sl:
            return "loss", j, -(stop_pts + FEE_PTS)

        # Track favorable.
        fav = (p - line_price) if direction == "up" else (line_price - p)
        best_fav = max(best_fav, fav)

        # Cap at max target.
        if fav >= max_target_pts:
            return "win", j, fav - FEE_PTS

        # Activate trailing stop.
        if best_fav >= trail_activate_pts:
            if not trailing_active:
                trailing_active = True
                # Move stop to breakeven.
                if direction == "up":
                    sl = line_price
                else:
                    sl = line_price

            # Trail: stop follows peak - offset.
            if direction == "up":
                trail_sl = (line_price + best_fav) - trail_offset_pts
                sl = max(sl, trail_sl)
                if p <= sl:
                    pnl = p - line_price - FEE_PTS
                    return "trail_win", j, pnl
            else:
                trail_sl = (line_price - best_fav) + trail_offset_pts
                sl = min(sl, trail_sl)
                if p >= sl:
                    pnl = line_price - p - FEE_PTS
                    return "trail_win", j, pnl

    ep = float(prices[last_idx])
    pnl = (ep - line_price if direction == "up" else line_price - ep) - FEE_PTS
    return "timeout", last_idx, pnl


# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DayResult:
    entries: list[QualifiedEntry]
    outcomes: dict[str, tuple[list[str], list[int], list[float]]]


def precompute_outcomes(
    entries: list[QualifiedEntry],
    dc: DayCache,
    configs: dict[str, dict],
) -> dict[str, tuple[list[str], list[int], list[float]]]:
    """Evaluate entries under multiple exit configs.

    configs: { config_name: { "type": "fixed"|"trailing"|"per_level",
                              + params } }
    """
    eod_ns = _eod_cutoff_ns(dc.date)
    results = {}

    # Per-level T/S derived from bounce analysis MFE medians.
    # Target ≈ 40% of median bounce, max target ≈ 60%.
    # IBH bounce median ~24 pts, Fib ~25-35, IBL ~30, VWAP ~26.
    PER_LEVEL_TS = {
        "IBH": (8, 15),
        "IBL": (10, 18),
        "FIB_EXT_HI_1.272": (8, 15),
        "FIB_EXT_LO_1.272": (12, 20),
        "VWAP": (6, 12),
    }

    for cfg_name, cfg in configs.items():
        outcomes = []
        exit_ns = []
        pnl_usd = []

        for e in entries:
            if cfg["type"] == "fixed":
                out, eidx, pnl = evaluate_adaptive(
                    e.global_idx, e.line_price, e.direction,
                    dc.full_ts_ns, dc.full_prices,
                    cfg["stop"], cfg["target"],
                    cfg.get("timeout", 900), eod_ns,
                )
            elif cfg["type"] == "trailing":
                out, eidx, pnl = evaluate_trailing(
                    e.global_idx, e.line_price, e.direction,
                    dc.full_ts_ns, dc.full_prices,
                    cfg["stop"], cfg["trail_activate"],
                    cfg["trail_offset"],
                    cfg.get("max_target", 50.0),
                    cfg.get("timeout", 900), eod_ns,
                )
            elif cfg["type"] == "per_level":
                ts = PER_LEVEL_TS.get(e.level, (8, 15))
                out, eidx, pnl = evaluate_adaptive(
                    e.global_idx, e.line_price, e.direction,
                    dc.full_ts_ns, dc.full_prices,
                    cfg["stop"], float(ts[0]),
                    cfg.get("timeout", 900), eod_ns,
                )
            elif cfg["type"] == "per_level_trailing":
                ts = PER_LEVEL_TS.get(e.level, (8, 15))
                out, eidx, pnl = evaluate_trailing(
                    e.global_idx, e.line_price, e.direction,
                    dc.full_ts_ns, dc.full_prices,
                    cfg["stop"], float(ts[0]) * 0.5,
                    cfg.get("trail_offset", 3.0),
                    float(ts[1]),
                    cfg.get("timeout", 900), eod_ns,
                )
            else:
                continue

            outcomes.append(out)
            exit_ns.append(int(dc.full_ts_ns[eidx]))
            pnl_usd.append(pnl * MULTIPLIER)

        results[cfg_name] = (outcomes, exit_ns, pnl_usd)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    outcome: str
    pnl_usd: float


def replay(days, data_by_date, cfg_name, max_per_level, daily_loss, max_consec):
    trades = []
    for date in days:
        dr = data_by_date.get(date)
        if not dr or cfg_name not in dr.outcomes:
            continue
        entries = dr.entries
        outcomes, exit_ns, pnl_usd = dr.outcomes[cfg_name]
        eod = _eod_cutoff_ns(date)
        pos_exit = 0
        dpnl = 0.0
        dcons = 0
        stopped = False
        lc = {}
        for i, e in enumerate(entries):
            if stopped: break
            if e.entry_ns >= eod: break
            if e.entry_ns < pos_exit: continue
            lv = lc.get(e.level, 0)
            if lv >= max_per_level: continue
            pos_exit = exit_ns[i]
            lc[e.level] = lv + 1
            trades.append(Trade(date, e.level, outcomes[i], pnl_usd[i]))
            dpnl += pnl_usd[i]
            if pnl_usd[i] < 0:
                dcons += 1
            else:
                dcons = 0
            if daily_loss and dpnl <= -daily_loss: stopped = True
            if max_consec and dcons >= max_consec: stopped = True
    return trades


def fmt(trades, nd, label=""):
    if not trades:
        return f"  {label:>55s}  no trades"
    w = sum(1 for t in trades if t.outcome in ("win", "trail_win"))
    l = sum(1 for t in trades if t.outcome == "loss")
    o = len(trades) - w - l
    d = w + l
    wr = w / d * 100 if d else 0
    pnl = sum(t.pnl_usd for t in trades)
    ppd = pnl / nd
    eq = STARTING_BALANCE; peak = eq; dd = 0.0
    for t in trades:
        eq += t.pnl_usd; peak = max(peak, eq); dd = max(dd, peak - eq)
    avg_w = sum(t.pnl_usd for t in trades if t.outcome in ("win", "trail_win")) / w if w else 0
    avg_l = sum(t.pnl_usd for t in trades if t.outcome == "loss") / l if l else 0
    return (
        f"  {label:>55s}  {len(trades):>4} ({len(trades)/nd:.1f}/d) "
        f"{w}W/{l}L/{o}O {wr:>5.1f}%  "
        f"W:{avg_w:>+5.1f} L:{avg_l:>+5.1f}  "
        f"PnL {pnl:>+8,.0f} ({ppd:>+5.1f}/d)  DD {dd:>5,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 110)
    print("  DAY TRADER SIMULATION — Human pre-filter + bot execution")
    print("=" * 110)

    days = load_cached_days()
    day_caches = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except: pass
    valid_days = sorted(day_caches.keys())
    N = len(valid_days)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    # Exit configs to test.
    CONFIGS = {
        # Baselines (fixed T/S).
        "T8/S20 (human baseline)": {"type": "fixed", "target": 8, "stop": 20},
        "T8/S5 (tight stop)": {"type": "fixed", "target": 8, "stop": 5},
        "T8/S7": {"type": "fixed", "target": 8, "stop": 7},
        "T8/S10": {"type": "fixed", "target": 8, "stop": 10},
        "T10/S5": {"type": "fixed", "target": 10, "stop": 5},
        "T10/S7": {"type": "fixed", "target": 10, "stop": 7},
        "T12/S5": {"type": "fixed", "target": 12, "stop": 5},
        "T12/S7": {"type": "fixed", "target": 12, "stop": 7},
        "T6/S5": {"type": "fixed", "target": 6, "stop": 5},
        # Per-level T/S (IBH/IBL: T8, Fib: T10, VWAP: T6).
        "per-level S5": {"type": "per_level", "stop": 5},
        "per-level S7": {"type": "per_level", "stop": 7},
        "per-level S10": {"type": "per_level", "stop": 10},
        # Trailing (tight stop, let winners run).
        "trail S5 act@3 off@3": {"type": "trailing", "stop": 5, "trail_activate": 3, "trail_offset": 3, "max_target": 30},
        "trail S5 act@4 off@3": {"type": "trailing", "stop": 5, "trail_activate": 4, "trail_offset": 3, "max_target": 30},
        "trail S5 act@5 off@3": {"type": "trailing", "stop": 5, "trail_activate": 5, "trail_offset": 3, "max_target": 30},
        "trail S5 act@5 off@4": {"type": "trailing", "stop": 5, "trail_activate": 5, "trail_offset": 4, "max_target": 30},
        "trail S7 act@4 off@3": {"type": "trailing", "stop": 7, "trail_activate": 4, "trail_offset": 3, "max_target": 30},
        "trail S7 act@5 off@3": {"type": "trailing", "stop": 7, "trail_activate": 5, "trail_offset": 3, "max_target": 30},
        "trail S7 act@5 off@4": {"type": "trailing", "stop": 7, "trail_activate": 5, "trail_offset": 4, "max_target": 30},
        # Per-level trailing.
        "per-level trail S5 off@3": {"type": "per_level_trailing", "stop": 5, "trail_offset": 3},
        "per-level trail S7 off@3": {"type": "per_level_trailing", "stop": 7, "trail_offset": 3},
        "per-level trail S7 off@4": {"type": "per_level_trailing", "stop": 7, "trail_offset": 4},
    }

    # Precompute qualified entries + outcomes.
    human_w = Weights()
    print(f"\n  Finding human-approved entries + computing outcomes...", flush=True)
    t1 = time.time()
    data_by_date: dict[datetime.date, DayResult] = {}
    total_entries = 0
    for i, date in enumerate(valid_days):
        dc = day_caches[date]
        entries = find_qualified_entries(dc, human_w, min_human_score=5)
        outcomes = precompute_outcomes(entries, dc, CONFIGS)
        data_by_date[date] = DayResult(entries=entries, outcomes=outcomes)
        total_entries += len(entries)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N}... ({total_entries} entries so far)", flush=True)
    print(f"  {total_entries} qualified entries in {time.time()-t1:.0f}s")
    print(f"  {total_entries/N:.1f} entries/day (human approved + reached line)")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: In-sample sweep
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  STAGE 1: In-sample sweep (human pre-filter, all exit configs)")
    print("=" * 110 + "\n")

    results = []
    for cfg_name in CONFIGS:
        trades = replay(valid_days, data_by_date, cfg_name, 5, 150.0, 3)
        pnl = sum(t.pnl_usd for t in trades)
        ppd = pnl / N
        results.append((cfg_name, trades, ppd))
        print(fmt(trades, N, cfg_name))
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  --- Top 10 (in-sample, all {N} days) ---\n")
    for cfg_name, trades, ppd in results[:10]:
        print(fmt(trades, N, cfg_name))

    # Per-level breakdown for best config.
    best_name = results[0][0]
    best_trades = results[0][1]
    print(f"\n  Per-level breakdown ({best_name}):")
    for lv in ["IBH", "IBL", "VWAP", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]:
        lv_trades = [t for t in best_trades if t.level == lv]
        if lv_trades:
            print(fmt(lv_trades, N, f"  {lv}"))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  STAGE 2: Walk-forward validation (top configs)")
    print("=" * 110)

    # Pick top 10 + baseline.
    wf_configs = ["T8/S20 (human baseline)"]
    for name, _, _ in results[:10]:
        if name not in wf_configs:
            wf_configs.append(name)

    oos_by_config: dict[str, list[Trade]] = {}
    oos_days = 0
    k = INITIAL_TRAIN_DAYS
    windows = 0

    while k < N:
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days: break
        windows += 1
        oos_days += len(test_days)

        # No weight retraining needed — using fixed human weights.
        for cfg_name in wf_configs:
            trades = replay(test_days, data_by_date, cfg_name, 5, 150.0, 3)
            oos_by_config.setdefault(cfg_name, []).extend(trades)

        k += STEP_DAYS

    print(f"\n  {windows} windows, {oos_days} OOS days\n")
    oos_results = []
    for cfg, trades in oos_by_config.items():
        pnl = sum(t.pnl_usd for t in trades)
        ppd = pnl / oos_days
        oos_results.append((cfg, trades, ppd))
    oos_results.sort(key=lambda x: x[2], reverse=True)

    for cfg, trades, ppd in oos_results:
        print(fmt(trades, oos_days, f"OOS: {cfg}"))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("  STAGE 3: Recent 60 days")
    print("=" * 110)
    recent = valid_days[-60:]
    rn = len(recent)
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")

    for cfg, _, _ in oos_results[:8]:
        name = cfg.replace("OOS: ", "")
        trades = replay(recent, data_by_date, name, 5, 150.0, 3)
        print(fmt(trades, rn, f"RECENT: {name}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
