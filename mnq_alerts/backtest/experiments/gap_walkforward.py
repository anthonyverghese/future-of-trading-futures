"""4-quarter walk-forward of contrarian gap signal.

Full-sample finding: on gap-down days (open < prev_close), SELL trades
on FIB_0.236 (-$1.69/tr, n=375) and FIB_0.764 (-$0.67/tr, n=153) are
the weakest buckets. Other levels' SELLs are positive on gap-down.

Filter candidate: skip SELL trades on FIB_0.236 / FIB_0.764 when gap
is negative. Pre-sim potential ~+$2.27/day.

This walk-forward checks whether the per-quarter $/tr in those
specific (level, direction, gap-down) buckets is consistently
negative across regimes, the same way Thursday/counter-trend/VIX
walk-forwards were applied.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict

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

# Levels we're considering filtering SELLs on for gap-down days
TARGET_LEVELS = ["FIB_0.236", "FIB_0.764"]
# Gap-down threshold (any negative gap, since the "small DOWN" bucket has
# only 3 days, almost all gap-down days are in <=-5pt range anyway)
GAP_THRESHOLD = 0.0


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    print("\nComputing gap per day...", flush=True)
    gaps = {}
    for i, d in enumerate(dates_all):
        if i == 0:
            continue
        prior = dates_all[i - 1]
        prior_dc = caches[prior]
        today_dc = caches[d]
        if len(prior_dc.full_prices) == 0 or len(today_dc.full_prices) == 0:
            continue
        gaps[d] = float(today_dc.full_prices[0]) - float(prior_dc.full_prices[-1])

    print("\nApplying slippage + tagging trades...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        if r["date"] not in gaps:
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
        enriched.append({
            "date": r["date"],
            "level": r["level"],
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pnl": new_pnl,
            "gap": gaps[r["date"]],
        })

    sorted_dates = sorted({e["date"] for e in enriched})
    n_days = len(sorted_dates)
    q_size = n_days // 4
    quarters = [
        ("Q1 (oldest)", sorted_dates[:q_size]),
        ("Q2",          sorted_dates[q_size:2*q_size]),
        ("Q3",          sorted_dates[2*q_size:3*q_size]),
        ("Q4 (newest)", sorted_dates[3*q_size:]),
    ]

    # ============ Per-quarter check on the candidate filter buckets ============
    print(f"\n{'='*94}\nWALK-FORWARD: SELL on {TARGET_LEVELS} when gap < {GAP_THRESHOLD}\n{'='*94}")

    for lv in TARGET_LEVELS:
        print(f"\n  Level: {lv}")
        print(f"  {'Quarter':<14} {'#days':>5} {'gap-down days':>14} | "
              f"{'SELL n':>6} {'SELL $/tr':>9} {'SELL $tot':>10} | "
              f"{'(BUY n':>6} {'BUY $/tr)':>10}")
        bucket_neg_count = 0
        for label, qdates in quarters:
            qd_set = set(qdates)
            gap_down_dates = [d for d in qdates if gaps.get(d, 0) < GAP_THRESHOLD]
            in_bucket = [
                e for e in enriched
                if e["date"] in qd_set
                and e["level"] == lv
                and e["gap"] < GAP_THRESHOLD
            ]
            sells = [e for e in in_bucket if e["direction"] == "down"]
            buys = [e for e in in_bucket if e["direction"] == "up"]
            sn = len(sells); bn = len(buys)
            s_pnl = sum(e["pnl"] for e in sells)
            s_ptr = s_pnl / sn if sn else 0
            b_ptr = sum(e["pnl"] for e in buys) / bn if bn else 0
            print(f"  {label:<14} {len(qdates):>5} {len(gap_down_dates):>14} | "
                  f"{sn:>6} {s_ptr:>+9.2f} {s_pnl:>+10.0f} | "
                  f"{bn:>6} {b_ptr:>+10.2f}")
            if s_pnl < 0:
                bucket_neg_count += 1
        print(f"\n  Quarters with NEGATIVE SELL pnl: {bucket_neg_count}/4")

    # ============ Combined filter: $/day delta if we drop both per-quarter ===========
    print(f"\n{'='*94}\nCOMBINED FILTER: drop SELL on FIB_0.236 + FIB_0.764 on gap-down\n{'='*94}")
    print(f"  {'Quarter':<14} {'baseline_$/day':>14} {'kept_$/day':>11} {'delta':>8}")
    quarter_deltas = []
    for label, qdates in quarters:
        qd_set = set(qdates)
        all_in_q = [e for e in enriched if e["date"] in qd_set]
        days = len(qdates)
        baseline_pnl = sum(e["pnl"] for e in all_in_q)
        # Apply filter
        kept = [
            e for e in all_in_q
            if not (
                e["level"] in TARGET_LEVELS
                and e["direction"] == "down"
                and e["gap"] < GAP_THRESHOLD
            )
        ]
        kept_pnl = sum(e["pnl"] for e in kept)
        baseline_pday = baseline_pnl / days
        kept_pday = kept_pnl / days
        delta = kept_pday - baseline_pday
        print(f"  {label:<14} ${baseline_pday:>+13.2f} ${kept_pday:>+10.2f} ${delta:>+7.2f}")
        quarter_deltas.append(delta)

    print(f"\n  Quarters with positive delta: {sum(1 for d in quarter_deltas if d > 0)}/4")
    if all(d > 0 for d in quarter_deltas):
        print(f"\n  ROBUST: filter improves $/day in all 4 quarters. Worth real-sim.")
    elif sum(1 for d in quarter_deltas if d > 0) == 3:
        print(f"\n  GOOD: filter improves in 3/4 quarters. Worth real-sim with caveat.")
    elif sum(1 for d in quarter_deltas if d > 0) == 2:
        print(f"\n  WEAK: filter improves in only 2/4. Same overfitting risk pattern.")
    else:
        print(f"\n  NOT ROBUST: filter improves in <=1/4. Don't deploy.")


if __name__ == "__main__":
    main()
