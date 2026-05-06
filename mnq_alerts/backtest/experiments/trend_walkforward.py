"""4-quarter walk-forward of the with_trend [-30, -15] filter candidate.

Tier-1 trio analysis surfaced a "valley" in the with_trend distribution:
moderate counter-trend trades (with_trend ∈ [-30, -15]) had $/tr -$0.88
on n=431, vs surrounding buckets being positive. Filter would save
~+$1.16/day full-sample. Need walk-forward to confirm it's not
overfitting to one period.

Same C-config slippage assumptions: per-level buffer (IBH=0.75, others
1.0pt), 100ms latency, drop FIB_EXT_LO.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import (
    simulate_fill, adjusted_pnl,
)

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
PER_LEVEL_BUFFER = {"IBH": 0.75}

# The bucket we're validating
FILTER_LO = -30.0
FILTER_HI = -15.0


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    print("\nApplying slippage + computing with_trend per trade...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        dc = caches[r["date"]]
        buf = PER_LEVEL_BUFFER.get(r["level"], 1.0)
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=buf, latency_ms=100.0,
        )
        if fill_idx is None:
            continue
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        entry_idx = int(np.searchsorted(dc.full_ts_ns, r["entry_ns"]))
        sixty_m_ago_ns = r["entry_ns"] - 60 * 60 * int(1e9)
        old_idx = int(np.searchsorted(dc.full_ts_ns, sixty_m_ago_ns))
        trend = float(dc.full_prices[entry_idx]) - float(dc.full_prices[old_idx])
        if r["direction"] == "down":
            with_trend = -trend
        else:
            with_trend = trend
        enriched.append({
            "date": r["date"],
            "pnl": new_pnl,
            "with_trend": with_trend,
        })

    sorted_dates = sorted({t["date"] for t in enriched})
    n = len(sorted_dates)
    q_size = n // 4
    quarters = [
        ("Q1 (oldest)", sorted_dates[:q_size]),
        ("Q2",          sorted_dates[q_size:2*q_size]),
        ("Q3",          sorted_dates[2*q_size:3*q_size]),
        ("Q4 (newest)", sorted_dates[3*q_size:]),
    ]

    print(f"\n{'='*86}\nWALK-FORWARD: with_trend ∈ [{FILTER_LO}, {FILTER_HI}] bucket "
          f"(filter candidate)\n{'='*86}")
    print(f"  {'Quarter':<14} {'#days':>6} {'in_bucket':>10} {'in_$/tr':>9} "
          f"{'in_$/day':>10} | {'rest_$/day':>12} {'baseline':>10} {'Δ if drop':>11}")

    bucket_consistently_negative = 0
    bucket_consistently_below_avg = 0
    for label, qdates in quarters:
        qdate_set = set(qdates)
        in_bucket = [t for t in enriched if t["date"] in qdate_set
                     and FILTER_LO <= t["with_trend"] < FILTER_HI]
        all_in_q = [t for t in enriched if t["date"] in qdate_set]
        if not all_in_q:
            continue
        in_n = len(in_bucket)
        in_pnl = sum(t["pnl"] for t in in_bucket)
        rest = [t for t in all_in_q if not (FILTER_LO <= t["with_trend"] < FILTER_HI)]
        rest_pnl = sum(t["pnl"] for t in rest)
        all_pnl = sum(t["pnl"] for t in all_in_q)
        days = len(qdates)
        in_ptr = in_pnl / in_n if in_n else 0
        baseline = all_pnl / days
        rest_per_day = rest_pnl / days
        delta = rest_per_day - baseline
        print(f"  {label:<14} {days:>6} {in_n:>10} {in_ptr:>+9.2f} "
              f"{in_pnl/days:>+10.2f} | {rest_per_day:>+12.2f} {baseline:>+10.2f} "
              f"{delta:>+11.2f}")
        if in_pnl < 0:
            bucket_consistently_negative += 1
        if in_ptr < (all_pnl / len(all_in_q) - 0.5):  # bucket $/tr more than $0.50 below mean
            bucket_consistently_below_avg += 1

    print(f"\n{'='*86}\nVERDICT\n{'='*86}")
    print(f"  Quarters where in-bucket pnl is negative:           "
          f"{bucket_consistently_negative}/4")
    print(f"  Quarters where in-bucket $/tr is >$0.50 below mean: "
          f"{bucket_consistently_below_avg}/4")
    if bucket_consistently_negative >= 3:
        print(f"\n  ROBUST: bucket is reliably bad across quarters. Worth real-sim.")
    elif bucket_consistently_negative == 2:
        print(f"\n  WEAK: consistent in 2/4 — could go either way. Recommend skip.")
    else:
        print(f"\n  NOT ROBUST: bucket only bad in {bucket_consistently_negative}/4 "
              f"quarters. Full-sample signal is overfitting.")


if __name__ == "__main__":
    main()
