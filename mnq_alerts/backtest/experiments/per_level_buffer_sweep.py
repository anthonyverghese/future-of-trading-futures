"""Per-level entry_limit_buffer sweep under slippage modeling.

Currently buffer = 1.0pt globally. This script sweeps buffer values
0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0 *per level* on the
existing pickle's trades, finds each level's individual optimum, and
identifies whether per-level customization meaningfully beats the
global 1.0pt.

Caveats:
- Post-process (fast, <2 min after caches load).
- Doesn't model cap-budget interactions across levels — but those are
  small for buffer changes within a single level.
- Still subject to in-sample overfitting; quarterly walk-forward
  noted in memory shows recent edge is much smaller than full-sample
  so any per-level tuning needs to be modest (avoid overfitting to
  Q1's fat tail).

Same C-config filters: drop FIB_EXT_LO_1.272.
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
BUFFERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    rows = [r for r in rows if r["level"] not in EXCLUDED_LEVELS]
    days = len({r["date"] for r in rows})
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s, {len(rows)} trades, {days} days",
          flush=True)

    # For each level x buffer, compute slippage-adjusted total pnl
    # (post-process: drop trades that wouldn't fill, recompute pnl from fill)
    results = defaultdict(lambda: defaultdict(lambda: {"n": 0, "pnl": 0.0}))
    print("\nSweeping buffers...", flush=True)
    for buf_i, buf in enumerate(BUFFERS):
        t1 = time.time()
        for r in rows:
            dc = caches[r["date"]]
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
            s = results[r["level"]][buf]
            s["n"] += 1
            s["pnl"] += new_pnl
        print(f"  buffer={buf:.2f}: {time.time()-t1:.1f}s", flush=True)

    # Per-level table + identify optimum
    levels = sorted(results.keys())
    print(f"\n{'='*94}\nPER-LEVEL BUFFER SWEEP — $/day contribution (slippage-modeled)\n{'='*94}")
    print(f"  {'Level':<22}", end="")
    for buf in BUFFERS:
        print(f"{buf:>8.2f}", end="")
    print()
    optima = {}
    for lv in levels:
        print(f"  {lv:<22}", end="")
        best_buf, best_pday = None, -1e9
        for buf in BUFFERS:
            s = results[lv][buf]
            pday = s["pnl"] / days
            print(f"{pday:>+8.2f}", end="")
            if pday > best_pday:
                best_pday = pday
                best_buf = buf
        optima[lv] = (best_buf, best_pday)
        print(f"  ← opt @ {best_buf:.2f} = {best_pday:+.2f}")

    # Same table but $/trade (to spot quality-vs-volume tradeoff)
    print(f"\n  $/trade @ each buffer (pure trade quality, not weighted by volume):")
    print(f"  {'Level':<22}", end="")
    for buf in BUFFERS:
        print(f"{buf:>8.2f}", end="")
    print()
    for lv in levels:
        print(f"  {lv:<22}", end="")
        for buf in BUFFERS:
            s = results[lv][buf]
            ptr = s["pnl"] / s["n"] if s["n"] else 0
            print(f"{ptr:>+8.2f}", end="")
        print()

    # Trade count per buffer
    print(f"\n  Trade count @ each buffer (fill rate proxy):")
    print(f"  {'Level':<22}", end="")
    for buf in BUFFERS:
        print(f"{buf:>8}", end="")
    print()
    for lv in levels:
        print(f"  {lv:<22}", end="")
        for buf in BUFFERS:
            s = results[lv][buf]
            print(f"{s['n']:>8d}", end="")
        print()

    # Summary: global vs per-level optimum
    print(f"\n{'='*94}\nSUMMARY: GLOBAL vs PER-LEVEL OPTIMUM\n{'='*94}")
    global_at_1 = sum(results[lv][1.0]["pnl"] for lv in levels) / days
    optimum_per_level = sum(p for _, p in optima.values())
    print(f"  Global buffer = 1.0pt:                 ${global_at_1:+.2f}/day")
    print(f"  Per-level optimum (sum of best):       ${optimum_per_level:+.2f}/day")
    print(f"  Δ (post-process upper bound):          ${optimum_per_level - global_at_1:+.2f}/day")
    print(f"\n  Per-level optimal buffers:")
    for lv, (buf, pday) in optima.items():
        print(f"    {lv:<22} buffer={buf:.2f}  $/day=${pday:+.2f}")
    print(f"\n  Caveat: this is post-process (no cap-budget effects). Real-sim")
    print(f"  validation needed before deploying any per-level buffer change.")


if __name__ == "__main__":
    main()
