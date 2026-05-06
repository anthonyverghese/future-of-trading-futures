"""Walk-forward validation of the Thursday signal.

Split the 333 days into 4 quarters (chronological). For each quarter,
compute per-weekday $/day. If Thursday is consistently the worst (or
substantially below others) in 3 of 4 quarters, the signal is robust
enough to deploy. If it's only worst in 1 quarter, it's noise.

Same C-config (buffer=1.0pt, drop FIB_EXT_LO_1.272) slippage-modeled.
"""

from __future__ import annotations

import datetime
import os
import pickle
import sys
import time
from collections import defaultdict

import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import (
    simulate_fill, adjusted_pnl,
)

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    print("\nApplying slippage at C config (buffer=1.0pt)...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        dc = caches[r["date"]]
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=1.0, latency_ms=100.0,
        )
        if fill_idx is None:
            continue
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        enriched.append({
            "date": r["date"],
            "weekday": r["date"].weekday(),
            "level": r["level"],
            "outcome": r["outcome"],
            "pnl": new_pnl,
        })

    # Build chronological list of unique dates and split into 4 quarters
    sorted_dates = sorted({t["date"] for t in enriched})
    n = len(sorted_dates)
    q_size = n // 4
    quarters = [
        ("Q1 (oldest)", sorted_dates[:q_size]),
        ("Q2",          sorted_dates[q_size:2*q_size]),
        ("Q3",          sorted_dates[2*q_size:3*q_size]),
        ("Q4 (newest)", sorted_dates[3*q_size:]),
    ]

    print(f"\n{'='*82}\nQUARTERLY $/day BY WEEKDAY\n{'='*82}")
    print(f"  {'Quarter':<14} {'Dates':<25} ", end="")
    for wd in range(5):
        print(f"{WEEKDAYS[wd]:>9}", end="")
    print(f"  {'baseline':>10}")

    thu_consistently_bad = 0
    thu_worst_count = 0
    quarter_results = []
    for label, qdates in quarters:
        qdate_set = set(qdates)
        # Days per weekday in this quarter
        days_by_wd = defaultdict(int)
        for d in qdates:
            days_by_wd[d.weekday()] += 1
        # Pnl per weekday
        pnl_by_wd = defaultdict(float)
        for t in enriched:
            if t["date"] in qdate_set:
                pnl_by_wd[t["weekday"]] += t["pnl"]
        per_day_by_wd = {
            wd: (pnl_by_wd[wd] / days_by_wd[wd] if days_by_wd[wd] else 0)
            for wd in range(5)
        }
        baseline = sum(pnl_by_wd.values()) / len(qdates)
        date_range = f"{qdates[0]} → {qdates[-1]}"
        print(f"  {label:<14} {date_range:<25} ", end="")
        for wd in range(5):
            print(f"{per_day_by_wd[wd]:>+9.2f}", end="")
        print(f"  {baseline:>+10.2f}")
        quarter_results.append((label, per_day_by_wd, baseline))
        if per_day_by_wd[3] < 0:  # Thu is index 3
            thu_consistently_bad += 1
        worst_wd = min(per_day_by_wd, key=per_day_by_wd.get)
        if worst_wd == 3:
            thu_worst_count += 1

    print(f"\n{'='*82}\nWALK-FORWARD VERDICT\n{'='*82}")
    print(f"  Quarters where Thursday is the worst weekday: {thu_worst_count}/4")
    print(f"  Quarters where Thursday $/day is negative:    {thu_consistently_bad}/4")
    print()
    if thu_worst_count >= 3:
        print(f"  ROBUST signal: Thursday is consistently the worst weekday.")
        print(f"  Real-sim of 'drop Thursday' is justified — likely +$3-6/day improvement.")
    elif thu_worst_count == 2:
        print(f"  WEAK signal: Thursday is worst in 2/4 quarters. Could be sample bias.")
        print(f"  Recommend: don't deploy a Thursday filter without more data.")
    else:
        print(f"  NOT ROBUST: Thursday only worst in {thu_worst_count}/4 quarters.")
        print(f"  The full-sample Thursday signal is likely overfitting to one or two")
        print(f"  bad sub-periods. Don't deploy a filter on this.")

    # Also print pure ranking per quarter
    print(f"\n  Worst weekday per quarter (sanity):")
    for label, per_day, _ in quarter_results:
        ranked = sorted(per_day.items(), key=lambda x: x[1])
        print(f"    {label}: ranking worst→best → "
              f"{', '.join(f'{WEEKDAYS[wd]} (${pd:+.2f})' for wd, pd in ranked)}")


if __name__ == "__main__":
    main()
