"""
bot_optimal_wf.py — Definitive walk-forward for bot parameter optimization.

Sweeps score threshold + T/S + risk params together in a single walk-forward.
At each training window, picks the best combo by P&L/day, then tests on the
next unseen 20 days. Reports honest OOS results.

Also tests a static "Score >= 1" filter (the most robust single improvement
from the individual tests) for comparison.

Usage:
    python -u bot_optimal_wf.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time

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
from walk_forward import (
    DayEntries,
    DayOutcomes,
    Trade,
    precompute_day_entries,
    precompute_outcomes,
    trade_stats,
    _eod_cutoff_ns,
)
from bot_backtest import BOT_ENTRY_THRESHOLD, BOT_EXIT_THRESHOLD, FEE_PTS

_ET = pytz.timezone("America/New_York")

INITIAL_TRAIN_DAYS = 60
STEP_DAYS = 20

# Grids to sweep
TS_GRID = [
    (12.0, 25.0),  # current
    (10.0, 20.0),
    (10.0, 15.0),
    (8.0, 16.0),
]

RISK_GRID = [
    (100.0, 3),
    (150.0, 3),
    (150.0, 4),
    (200.0, 4),
    (None, None),
]

SCORE_GRID = [None, 0, 1, 2, 3]  # None = no filter


def _simple_score(level: str, direction: str, entry_count: int, hour: float) -> int:
    """Simplified scoring for bot entries."""
    score = 0
    if level == "IBL":
        score += 2
    elif level in ("FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"):
        score += 1
    elif level == "IBH":
        score -= 1

    if level == "IBL" and direction == "up":
        score += 1
    if level == "IBH" and direction == "down":
        score += 1
    if level == "IBH" and direction == "up":
        score -= 1

    if entry_count == 1:
        score -= 2
    elif entry_count >= 3:
        score += 1

    if hour >= 15.0:
        score += 2
    elif 13.0 <= hour < 15.0:
        score += 1
    elif 10.5 <= hour < 11.5:
        score -= 1

    return score


def enrich_entries(de: DayEntries, dc: DayCache) -> dict:
    """Compute per-entry features."""
    n = len(de.global_idx)
    features = {
        "time_hour": np.zeros(n, dtype=np.float64),
        "entry_count": de.entry_count.copy(),
    }
    for i in range(n):
        ts_sec = int(de.entry_ns[i]) / 1e9
        dt = datetime.datetime.fromtimestamp(ts_sec, tz=_ET)
        features["time_hour"][i] = dt.hour + dt.minute / 60.0
    return features


def replay_filtered(
    days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    outcomes_by_date: dict[datetime.date, DayOutcomes],
    features_by_date: dict[datetime.date, dict],
    daily_loss_usd: float | None,
    max_consec_losses: int | None,
    min_score: int | None = None,
) -> list[Trade]:
    """Replay with optional score filter + risk gates."""
    trades: list[Trade] = []
    for date in days:
        de = entries_by_date.get(date)
        do = outcomes_by_date.get(date)
        feat = features_by_date.get(date)
        if de is None or do is None or feat is None or len(de.global_idx) == 0:
            continue
        eod_ns = _eod_cutoff_ns(date)
        position_exit_ns = 0
        daily_pnl = 0.0
        consec = 0
        stopped = False
        for i in range(len(de.global_idx)):
            if stopped:
                break
            if int(de.entry_ns[i]) >= eod_ns:
                break
            if int(de.entry_ns[i]) < position_exit_ns:
                continue
            if min_score is not None:
                score = _simple_score(
                    de.level[i],
                    de.direction[i],
                    int(feat["entry_count"][i]),
                    float(feat["time_hour"][i]),
                )
                if score < min_score:
                    continue

            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            position_exit_ns = int(do.exit_ns[i])
            trades.append(
                Trade(date=date, level=de.level[i], outcome=outcome, pnl_usd=pnl)
            )
            daily_pnl += pnl
            if pnl < 0:
                consec += 1
            else:
                consec = 0
            if daily_loss_usd is not None and daily_pnl <= -daily_loss_usd:
                stopped = True
            if max_consec_losses is not None and consec >= max_consec_losses:
                stopped = True
    return trades


def print_stats(label: str, trades: list[Trade], test_days: int) -> None:
    """Print formatted stats for a set of trades."""
    if not trades:
        print(f"  {label}: NO TRADES")
        return
    total_pnl, wins, losses, wr, max_dd = trade_stats(trades)
    timeouts = sum(1 for t in trades if t.outcome == "timeout")
    tpd = len(trades) / max(1, test_days)
    ev = total_pnl / len(trades)
    # P&L per day
    pnl_per_day = total_pnl / max(1, test_days)
    # Risk-adjusted: P&L / MaxDD
    risk_adj = total_pnl / max_dd if max_dd > 0 else float("inf")
    print(
        f"  {label:<50} | "
        f"{len(trades):>4} trades ({tpd:.1f}/d) "
        f"[{wins}W/{losses}L/{timeouts}T] | "
        f"WR {wr:5.1f}% | "
        f"P&L ${total_pnl:>+8.2f} (${pnl_per_day:>+.2f}/d) | "
        f"EV ${ev:>+5.2f} | "
        f"MaxDD ${max_dd:>7.2f} | "
        f"P&L/DD {risk_adj:.2f}x"
    )


def main():
    t0 = time.time()

    # Load all cached data
    print("Step 1: Loading data...")
    all_days = sorted(load_cached_days())
    print(f"  {len(all_days)} cached trading days")

    print("Step 2: Preprocessing...")
    day_caches: dict[datetime.date, DayCache] = {}
    for d in all_days:
        try:
            df = load_day(d)
            dc = preprocess_day(df, d)
            if dc is not None and dc.ibh > dc.ibl:
                day_caches[d] = dc
        except Exception:
            pass
    valid_days = sorted(day_caches.keys())
    print(f"  {len(valid_days)} valid days")

    print("Step 3: Precomputing entries + features...")
    entries_by_date: dict[datetime.date, DayEntries] = {}
    features_by_date: dict[datetime.date, dict] = {}
    for d in valid_days:
        de = precompute_day_entries(day_caches[d])
        entries_by_date[d] = de
        features_by_date[d] = enrich_entries(de, day_caches[d])

    print("Step 4: Precomputing outcomes for T/S grid...")
    outcomes_by_ts: dict[tuple[float, float], dict[datetime.date, DayOutcomes]] = {}
    for target, stop in TS_GRID:
        outcomes_by_ts[(target, stop)] = {}
        for d in valid_days:
            outcomes_by_ts[(target, stop)][d] = precompute_outcomes(
                entries_by_date[d], day_caches[d], target, stop
            )
        print(f"  T{target}/S{stop} done")

    n = len(valid_days)
    test_day_count = n - INITIAL_TRAIN_DAYS

    # ──────────────────────────────────────────────────────────────────────────
    # WALK-FORWARD 1: Baseline (no score filter, sweep T/S + risk only)
    # This matches the current live bot behavior.
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(
        f"  WALK-FORWARD COMPARISON — {len(valid_days)} days, {INITIAL_TRAIN_DAYS}+{STEP_DAYS}"
    )
    print(f"{'='*120}")

    baseline_trades: list[Trade] = []
    adaptive_trades: list[Trade] = []
    score1_trades: list[Trade] = []
    window_log: list[str] = []

    k = INITIAL_TRAIN_DAYS
    window_num = 0
    while k < n:
        train_days = valid_days[:k]
        test_days_list = valid_days[k : k + STEP_DAYS]
        if not test_days_list:
            break
        window_num += 1

        # --- Baseline: sweep T/S + risk, no score filter ---
        best_baseline_pnl = -float("inf")
        best_baseline_cfg = ((12.0, 25.0), (150.0, 4))
        for ts in TS_GRID:
            outs = outcomes_by_ts[ts]
            for risk in RISK_GRID:
                trades = replay_filtered(
                    train_days,
                    entries_by_date,
                    outs,
                    features_by_date,
                    risk[0],
                    risk[1],
                    min_score=None,
                )
                total = sum(t.pnl_usd for t in trades)
                per_day = total / len(train_days)
                if per_day > best_baseline_pnl:
                    best_baseline_pnl = per_day
                    best_baseline_cfg = (ts, risk)

        bts, brisk = best_baseline_cfg
        baseline_test = replay_filtered(
            test_days_list,
            entries_by_date,
            outcomes_by_ts[bts],
            features_by_date,
            brisk[0],
            brisk[1],
            min_score=None,
        )
        baseline_trades.extend(baseline_test)

        # --- Adaptive: sweep T/S + risk + score threshold ---
        best_adaptive_pnl = -float("inf")
        best_adaptive_cfg = ((12.0, 25.0), (150.0, 4), None)
        for ts in TS_GRID:
            outs = outcomes_by_ts[ts]
            for risk in RISK_GRID:
                for sc in SCORE_GRID:
                    trades = replay_filtered(
                        train_days,
                        entries_by_date,
                        outs,
                        features_by_date,
                        risk[0],
                        risk[1],
                        min_score=sc,
                    )
                    total = sum(t.pnl_usd for t in trades)
                    per_day = total / len(train_days)
                    if per_day > best_adaptive_pnl:
                        best_adaptive_pnl = per_day
                        best_adaptive_cfg = (ts, risk, sc)

        ats, arisk, asc = best_adaptive_cfg
        adaptive_test = replay_filtered(
            test_days_list,
            entries_by_date,
            outcomes_by_ts[ats],
            features_by_date,
            arisk[0],
            arisk[1],
            min_score=asc,
        )
        adaptive_trades.extend(adaptive_test)

        # --- Static Score >= 1: fixed score filter, sweep T/S + risk ---
        best_s1_pnl = -float("inf")
        best_s1_cfg = ((12.0, 25.0), (150.0, 4))
        for ts in TS_GRID:
            outs = outcomes_by_ts[ts]
            for risk in RISK_GRID:
                trades = replay_filtered(
                    train_days,
                    entries_by_date,
                    outs,
                    features_by_date,
                    risk[0],
                    risk[1],
                    min_score=1,
                )
                total = sum(t.pnl_usd for t in trades)
                per_day = total / len(train_days)
                if per_day > best_s1_pnl:
                    best_s1_pnl = per_day
                    best_s1_cfg = (ts, risk)

        s1ts, s1risk = best_s1_cfg
        s1_test = replay_filtered(
            test_days_list,
            entries_by_date,
            outcomes_by_ts[s1ts],
            features_by_date,
            s1risk[0],
            s1risk[1],
            min_score=1,
        )
        score1_trades.extend(s1_test)

        # Log window choices
        window_log.append(
            f"  Window {window_num:>2}: "
            f"test {test_days_list[0]}→{test_days_list[-1]} ({len(test_days_list)}d) | "
            f"baseline T{bts[0]:.0f}/S{bts[1]:.0f} risk={brisk} | "
            f"adaptive T{ats[0]:.0f}/S{ats[1]:.0f} risk={arisk} score>={asc} | "
            f"score1 T{s1ts[0]:.0f}/S{s1ts[1]:.0f} risk={s1risk}"
        )

        k += STEP_DAYS

    # Print window details
    print(f"\n  Window-by-window config choices:")
    for line in window_log:
        print(line)

    # Print final OOS results
    print(f"\n  {'─'*120}")
    print(f"  OUT-OF-SAMPLE RESULTS ({test_day_count} test days):")
    print(f"  {'─'*120}")
    print_stats(
        "Baseline (no score, adaptive T/S+risk)", baseline_trades, test_day_count
    )
    print_stats("Score >= 1 (fixed, adaptive T/S+risk)", score1_trades, test_day_count)
    print_stats("Fully adaptive (T/S+risk+score)", adaptive_trades, test_day_count)

    # Per-level breakdown for the best config
    print(f"\n  {'─'*120}")
    print(f"  PER-LEVEL BREAKDOWN — Fully adaptive OOS:")
    print(f"  {'─'*120}")
    for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        level_trades = [t for t in adaptive_trades if t.level == level]
        if level_trades:
            print_stats(level, level_trades, test_day_count)

    # Daily P&L distribution
    print(f"\n  {'─'*120}")
    print(f"  DAILY P&L DISTRIBUTION — Fully adaptive OOS:")
    print(f"  {'─'*120}")
    for label, trades in [
        ("Baseline", baseline_trades),
        ("Score >= 1", score1_trades),
        ("Fully adaptive", adaptive_trades),
    ]:
        daily_pnls: dict[datetime.date, float] = {}
        for t in trades:
            daily_pnls[t.date] = daily_pnls.get(t.date, 0.0) + t.pnl_usd
        if daily_pnls:
            vals = sorted(daily_pnls.values())
            pos_days = sum(1 for v in vals if v > 0)
            neg_days = sum(1 for v in vals if v < 0)
            zero_days = sum(1 for v in vals if v == 0)
            print(
                f"  {label:<25} | "
                f"pos {pos_days} / neg {neg_days} / flat {zero_days} days | "
                f"worst ${vals[0]:>+.2f} | "
                f"median ${vals[len(vals)//2]:>+.2f} | "
                f"best ${vals[-1]:>+.2f} | "
                f"avg ${sum(vals)/len(vals):>+.2f}/day"
            )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
