"""4-quarter walk-forward validation of VIX directional finding.

Full-sample showed: in the 22-27 VIX bucket, BUYs averaged +$2.50/tr
while SELLs averaged -$0.56/tr (Δ = -$3.06 SELL−BUY). Walk-forward
checks whether this asymmetry holds across regimes or comes from one
specific market period.

Specifically for the prev-close-VIX (live-usable timestamp), the
22-27 bucket showed Δ = -$2.41. We test that.

Verdict criteria:
  4/4 quarters with SELL < BUY: ROBUST — worth real-sim of filter
  3/4: WORTH TESTING with caveats
  ≤2/4: NOT robust — full-sample signal is regime-specific
"""

from __future__ import annotations

import csv
import datetime
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
VIX_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data_cache", "vix_daily.csv"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
PER_LEVEL_BUFFER = {"IBH": 0.75}

# The bucket and asymmetry we're validating
VIX_LO, VIX_HI = 22.0, 27.0


def load_vix() -> dict:
    """Returns prev_close_vix dict (live-usable timestamp)."""
    close_vix = {}
    with open(VIX_CSV_PATH) as f:
        for row in csv.DictReader(f):
            d = datetime.date.fromisoformat(row["date"])
            close_vix[d] = float(row["close"])
    prev_close = {}
    sorted_dates = sorted(close_vix.keys())
    for i in range(1, len(sorted_dates)):
        prev_close[sorted_dates[i]] = close_vix[sorted_dates[i - 1]]
    return prev_close


def main():
    print("Loading pickle + VIX + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    prev_close_vix = load_vix()
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    print("\nApplying slippage + tagging with prev-close VIX...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        d = r["date"]
        if d not in prev_close_vix:
            continue
        dc = caches[d]
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
            "date": d,
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pnl": new_pnl,
            "vix": prev_close_vix[d],
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

    print(f"\n{'='*94}\nWALK-FORWARD: BUY vs SELL in prev-close VIX [{VIX_LO}, {VIX_HI})\n{'='*94}")
    print(f"  {'Quarter':<14} {'#days':>5} {'in_bucket':>10} | "
          f"{'BUY n':>5} {'BUY $/tr':>9} {'BUY WR':>7} | "
          f"{'SELL n':>6} {'SELL $/tr':>9} {'SELL WR':>8} | "
          f"{'Δ (S-B)':>8}")

    sell_under_buy_count = 0
    quarter_results = []
    for label, qdates in quarters:
        qdate_set = set(qdates)
        in_bucket = [
            e for e in enriched
            if e["date"] in qdate_set and VIX_LO <= e["vix"] < VIX_HI
        ]
        buys = [e for e in in_bucket if e["direction"] == "up"]
        sells = [e for e in in_bucket if e["direction"] == "down"]
        n_total = len(in_bucket)
        bn = len(buys); sn = len(sells)
        b_pnl = sum(e["pnl"] for e in buys)
        s_pnl = sum(e["pnl"] for e in sells)
        b_ptr = b_pnl / bn if bn else 0
        s_ptr = s_pnl / sn if sn else 0
        b_wr = sum(1 for e in buys if e["outcome"] == "win") / bn * 100 if bn else 0
        s_wr = sum(1 for e in sells if e["outcome"] == "win") / sn * 100 if sn else 0
        delta = s_ptr - b_ptr
        print(f"  {label:<14} {len(qdates):>5} {n_total:>10} | "
              f"{bn:>5} {b_ptr:>+9.2f} {b_wr:>7.1f} | "
              f"{sn:>6} {s_ptr:>+9.2f} {s_wr:>8.1f} | "
              f"{delta:>+8.2f}")
        if delta < 0:
            sell_under_buy_count += 1
        quarter_results.append((label, b_ptr, s_ptr, bn, sn, delta))

    print(f"\n{'='*94}\nVERDICT\n{'='*94}")
    print(f"  Quarters where SELL $/tr < BUY $/tr in 22-27 VIX bucket: "
          f"{sell_under_buy_count}/4")
    if sell_under_buy_count >= 3:
        print(f"\n  ROBUST: BUY consistently outperforms SELL in 22-27 VIX. "
              f"Worth real-sim of:")
        print(f"    - 'skip SELL when prev-close VIX in [22, 27)' filter")
        print(f"    - or 'SELL-cap reduction when prev-close VIX in [22, 27)'")
    elif sell_under_buy_count == 2:
        print(f"\n  WEAK: 2/4 quarters confirm. Could be sample noise. "
              f"Recommend skip — overfitting risk same as Thursday filter.")
    else:
        print(f"\n  NOT ROBUST: Full-sample signal driven by "
              f"{sell_under_buy_count}/4 quarters. Don't deploy.")

    # Also show $/day contribution per bucket per quarter, since the
    # in-bucket counts may be small per quarter (45 days × 25% = ~11 days)
    print(f"\n  Sample sizes per quarter (note: 22-27 bucket only fires on "
          f"~14% of all days):")
    for label, b_ptr, s_ptr, bn, sn, delta in quarter_results:
        print(f"    {label}:  BUY n={bn:>3}  SELL n={sn:>3}  "
              f"(small samples → high variance)")


if __name__ == "__main__":
    main()
