"""Real-sim test: IBH buffer = 0.5pt (tightest) vs current C config (1.0pt).

The per-level buffer sweep (post-process) showed IBH's $/trade is the
only level with a clean monotonic decline as buffer widens:
  buffer:   0.50   0.75   1.00   1.25   1.50   1.75   2.00
  IBH $/tr: 1.31   1.25   0.98   0.86   0.71   0.76   0.70

Mechanically motivated: IBH is T6/S20, the smallest target. Slippage
of 1pt eats 17% of the target. Tighter buffer keeps wins closer to
target_pts. Other levels (FIB_0.764 T10, FIB_0.618 T12) are flatter
across buffers because larger targets absorb slippage.

This sim validates whether the post-process signal (~+$0.69/day from
1.0→0.75, ~+$0.48/day from 1.0→0.5) survives real-sim. Post-process
has been wrong twice today (cap-opt, buffer 1.0 vs target/2), so
confirmation is essential before deploying.

Three configs:
  C (current production):  buffer = 1.0pt globally
  IBH=0.75 only:           buffer = {"IBH": 0.75}, others 1.0pt
  IBH=0.50 only:           buffer = {"IBH": 0.50}, others 1.0pt
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.run_slippage_aware_v1 import run, fmt_result


def main():
    print("Loading day caches...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    results = {}

    print("\n" + "=" * 70)
    print("BASELINE: C deployed (buffer=1.0pt globally)")
    print("=" * 70)
    results["C"] = run(
        "C_baseline", dates, caches,
        exclude_levels={"FIB_EXT_LO_1.272"},
        simulate_slippage=True,
        entry_limit_buffer_pts_override=1.0,
    )
    fmt_result(results["C"])

    for ibh_buf in [0.75, 0.5]:
        label = f"IBH_buf_{ibh_buf}"
        print("\n" + "=" * 70)
        print(f"VARIANT: IBH buffer = {ibh_buf}pt, others = 1.0pt")
        print("=" * 70)
        # Pass a dict via the override mechanism. simulate_v2 already
        # monkey-patches BOT_ENTRY_LIMIT_BUFFER_PTS to whatever value is
        # passed; bot_trader now supports dict form.
        per_level = {
            "IBH": ibh_buf,
            "FIB_0.236": 1.0,
            "FIB_0.618": 1.0,
            "FIB_0.764": 1.0,
            "FIB_EXT_HI_1.272": 1.0,
        }
        results[label] = run(
            label, dates, caches,
            exclude_levels={"FIB_EXT_LO_1.272"},
            simulate_slippage=True,
            entry_limit_buffer_pts_override=per_level,
        )
        fmt_result(results[label])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Config                         $/day      MaxDD     Trades")
    print(f"  C (buffer=1.0 global):        ${results['C']['pnl_per_day']:+.2f}     "
          f"${results['C']['max_dd']:.0f}      {results['C']['trades']}")
    for ibh_buf in [0.75, 0.5]:
        r = results[f"IBH_buf_{ibh_buf}"]
        delta = r['pnl_per_day'] - results['C']['pnl_per_day']
        print(f"  IBH={ibh_buf}, others=1.0:        "
              f"${r['pnl_per_day']:+.2f}     "
              f"${r['max_dd']:.0f}      {r['trades']}     "
              f"(Δ vs C: ${delta:+.2f}/day)")
    print()
    # Just IBH numbers from each config
    print("  IBH-only contribution:")
    for label, r in results.items():
        ibh_stats = r["by_level"].get("IBH", (0, 0, 0.0))
        n, w, p = ibh_stats
        if n > 0:
            wr = w / n * 100
            print(f"    {label:<22} IBH n={n:<5} WR={wr:.1f}%  $/tr=${p/n:+.2f}  "
                  f"$/day=${p/r['days']:+.2f}")


if __name__ == "__main__":
    main()
