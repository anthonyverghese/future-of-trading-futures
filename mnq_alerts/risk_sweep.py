"""
risk_sweep.py — Sweep risk gates for specific T/S configs on OOS period.

Reuses walk_forward.py's precompute infrastructure but only for the T/S
configs we care about, and sweeps a wider range of risk gates to find the
best combo per T/S.

Usage:
    python -u risk_sweep.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from walk_forward import (
    INITIAL_TRAIN_DAYS,
    STEP_DAYS,
    precompute_day_entries,
    precompute_outcomes,
    replay_with_risk,
    trade_stats,
    timeout_stats,
)
from targeted_backtest import load_cached_days, load_day, preprocess_day

# Configs to sweep.
TS_CONFIGS = [
    (4.0, 8.0),
    (8.0, 20.0),
    (12.0, 25.0),  # for reference vs current
]

RISK_CONFIGS: list[tuple[float | None, int | None]] = [
    (75.0, 3),
    (100.0, 3),
    (100.0, 4),
    (150.0, 3),
    (150.0, 4),
    (150.0, 5),
    (200.0, 3),
    (200.0, 4),
    (200.0, 5),
    (300.0, 4),
    (None, None),  # unrestricted
]


def main() -> None:
    t0 = time.time()
    print("=" * 78, flush=True)
    print("  RISK-GATE SWEEP — 4/8, 8/20, 12/25 × many risk gates", flush=True)
    print("  OOS = days after INITIAL_TRAIN_DAYS (true OOS period)", flush=True)
    print("=" * 78, flush=True)

    # Load days.
    days = load_cached_days()
    print(f"\n  Loading {len(days)} cached days...", flush=True)
    day_caches = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except Exception:
            pass
    valid_days = sorted(day_caches.keys())
    print(f"  Loaded {len(valid_days)} valid days in {time.time()-t0:.1f}s", flush=True)

    # OOS days = everything after the initial train window (no retraining; fixed
    # configs are evaluated on the true OOS period for fair comparison with
    # walk_forward.py's adaptive run).
    oos_days = valid_days[INITIAL_TRAIN_DAYS:]
    print(
        f"  OOS window: {oos_days[0]} → {oos_days[-1]} ({len(oos_days)} days)",
        flush=True,
    )

    # Stage 1: entries per day (T/S-independent).
    t1 = time.time()
    print(f"\n  Precomputing entries for {len(valid_days)} days...", flush=True)
    entries_by_date = {d: precompute_day_entries(day_caches[d]) for d in valid_days}
    print(f"  Done in {time.time()-t1:.1f}s", flush=True)

    # Stage 2: outcomes per T/S config.
    t2 = time.time()
    print(f"\n  Precomputing outcomes for {len(TS_CONFIGS)} T/S configs...", flush=True)
    outcomes_by_ts = {}
    for ts in TS_CONFIGS:
        outcomes_by_ts[ts] = {
            d: precompute_outcomes(entries_by_date[d], day_caches[d], ts[0], ts[1])
            for d in valid_days
        }
        print(f"    T/S={ts[0]:.0f}/{ts[1]:.0f} done", flush=True)
    print(f"  Stage 2 done in {time.time()-t2:.1f}s", flush=True)

    # Sweep risk per T/S.
    for ts in TS_CONFIGS:
        print("\n" + "=" * 78, flush=True)
        print(f"  T/S = {ts[0]:.0f}/{ts[1]:.0f}", flush=True)
        print("=" * 78, flush=True)
        print(
            f"  {'Risk':>12} {'Trades':>7} {'/day':>6} {'WR%':>6} "
            f"{'$/day':>7} {'MaxDD':>8} {'TOs':>5} {'TO avg':>8} {'TO total':>10}",
            flush=True,
        )
        print(
            f"  {'-'*12} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*5} {'-'*8} {'-'*10}",
            flush=True,
        )
        obd = outcomes_by_ts[ts]
        results = []
        for risk in RISK_CONFIGS:
            tr = replay_with_risk(oos_days, entries_by_date, obd, risk[0], risk[1])
            if not tr:
                continue
            total, wins, losses, wr, dd = trade_stats(tr)
            per_day = total / len(oos_days)
            to_n, to_total, to_avg = timeout_stats(tr)
            label = f"${int(risk[0])}/{risk[1]}" if risk[0] is not None else "unrestr"
            results.append(
                (
                    label,
                    tr,
                    total,
                    wins,
                    losses,
                    wr,
                    dd,
                    per_day,
                    to_n,
                    to_avg,
                    to_total,
                )
            )
            print(
                f"  {label:>12} {len(tr):>7} {len(tr)/len(oos_days):>5.1f} "
                f"{wr:>5.1f}% ${per_day:>+5.1f} ${dd:>6,.0f} "
                f"{to_n:>5} ${to_avg:>+6.2f} ${to_total:>+8,.0f}",
                flush=True,
            )

        # Highlight best by $/day and best by $/day per $ of DD.
        if results:
            best_pnl = max(results, key=lambda r: r[7])
            best_sharpe = max(
                results, key=lambda r: r[7] / max(r[6], 1.0)
            )  # $/day per $ of DD
            print(
                f"\n  Best $/day:            {best_pnl[0]}  (${best_pnl[7]:+.1f}/day, DD ${best_pnl[6]:,.0f})",
                flush=True,
            )
            print(
                f"  Best $/day per $ DD:  {best_sharpe[0]}  "
                f"(${best_sharpe[7]:+.1f}/day, DD ${best_sharpe[6]:,.0f}, "
                f"ratio {best_sharpe[7]/max(best_sharpe[6],1):.4f})",
                flush=True,
            )

    print(f"\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
