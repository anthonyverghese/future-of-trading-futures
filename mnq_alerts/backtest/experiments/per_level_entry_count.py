"""Per-level entry-count bucket analysis under slippage modeling.

Cap optimization only helps if late-entries (high entry_count) on a
given level have systematically worse $/tr than early-entries. If
$/tr is roughly flat across entry_count, the cap is a pure risk-
tolerance lever, not an edge-finder, and lowering it just removes
positive-EV trades.

Same filters as the deployed C config:
  - buffer=1.0pt slippage modeled
  - exclude FIB_EXT_LO_1.272
  - drop trades that wouldn't fill
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


def main() -> None:
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    days = len({r["date"] for r in rows})
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(rows)} trades, caches loaded in {time.time()-t0:.0f}s",
          flush=True)

    print("\nApplying slippage + collecting (level, entry_count, pnl)...",
          flush=True)
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
        enriched.append((r["level"], r["entry_count"], new_pnl, r["outcome"]))

    levels = sorted({t[0] for t in enriched})
    print(f"\n{'='*78}")
    print("$/tr BY ENTRY_COUNT (slippage-modeled, buffer=1.0pt)")
    print('='*78)
    # Per-level current caps for context
    caps = {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "IBH": 7,
    }

    for lv in levels:
        cap = caps.get(lv, "?")
        lv_trades = [t for t in enriched if t[0] == lv]
        print(f"\n  {lv}  (current cap={cap}, total n={len(lv_trades)})")
        # Group by entry_count
        by_ec = defaultdict(list)
        for _, ec, pnl, _ in lv_trades:
            by_ec[ec].append(pnl)
        max_ec = max(by_ec.keys()) if by_ec else 0
        # Cumulative — what's $/tr if we cap at K?
        print(f"    {'EC':>4} {'n':>4} {'$/tr':>7} {'cum_n':>6} {'cum_$tot':>10} {'cum_$/tr':>9}")
        cum_n = 0
        cum_pnl = 0.0
        for ec in sorted(by_ec.keys()):
            pnls = by_ec[ec]
            n = len(pnls)
            ptr = sum(pnls) / n
            cum_n += n
            cum_pnl += sum(pnls)
            cum_ptr = cum_pnl / cum_n
            print(f"    {ec:>4d} {n:>4d} {ptr:>+7.2f} {cum_n:>6d} {cum_pnl:>+10.0f} {cum_ptr:>+9.2f}")

        # Marginal $/tr in tail buckets (e.g., trades with ec>5 vs ec≤5)
        # This is the "if we cap at K, what's the lost $/tr from dropped tail trades"
        thresholds = [2, 3, 5, 8, 12]
        print(f"    {'IF cap K':>9} {'kept_n':>7} {'kept_$/day':>12} "
              f"{'dropped_$/tr':>14} {'delta':>7}")
        baseline_pnl_per_day = sum(p for ps in by_ec.values() for p in ps) / days
        for k in thresholds:
            if k >= max_ec:
                continue
            kept = [(ec, pnls) for ec, pnls in by_ec.items() if ec <= k]
            dropped = [(ec, pnls) for ec, pnls in by_ec.items() if ec > k]
            kept_n = sum(len(p) for _, p in kept)
            kept_pnl = sum(sum(p) for _, p in kept)
            dropped_n = sum(len(p) for _, p in dropped)
            dropped_pnl = sum(sum(p) for _, p in dropped)
            kept_per_day = kept_pnl / days
            dropped_per_tr = dropped_pnl / dropped_n if dropped_n else 0
            delta = kept_per_day - baseline_pnl_per_day
            print(f"    {k:>9d} {kept_n:>7d} {kept_per_day:>+12.2f} "
                  f"{dropped_per_tr:>+14.2f} {delta:>+7.2f}")


if __name__ == "__main__":
    main()
