"""Single sim: target/2 buffer + drop FIB_EXT_LO_1.272 (slippage-aware).

Fills the gap in the slippage-aware sweep — the per-level math from
run_slippage_aware_v1.py predicted ~+$16.62/day from this config, but
that's an estimate. This run measures it directly.

Comparison points (from run_slippage_aware_v1.py final results):
  A: target/2 all levels        $+14.68/day  MaxDD $1,599
  C: buffer=1 drop FIB_EXT_LO   $+15.66/day  MaxDD $980
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.run_slippage_aware_v1 import run, fmt_result


def main():
    print("Loading day caches (~3 min)...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 70)
    print("D: target/2 buffer + drop FIB_EXT_LO_1.272, SLIPPAGE-AWARE")
    print("=" * 70)
    d = run(
        "D_target2_drop_fib_ext_lo", dates, caches,
        exclude_levels={"FIB_EXT_LO_1.272"},
        simulate_slippage=True,
        entry_limit_buffer_pts_override=0.0,  # 0 = legacy target/2 path
    )
    fmt_result(d)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("Reference (from previous run):")
    print("  A: target/2 all levels              $+14.68/day  MaxDD $1,599")
    print("  B: buffer=1 only                    $+13.29/day  MaxDD $1,110")
    print("  C: buffer=1 + drop FIB_EXT_LO       $+15.66/day  MaxDD $980")
    print(f"  D: target/2 + drop FIB_EXT_LO       ${d['pnl_per_day']:+.2f}/day"
          f"  MaxDD ${d['max_dd']:.0f}")
    print()
    print(f"D vs A (drop FIB_EXT_LO at target/2): ${d['pnl_per_day'] - 14.68:+.2f}/day")
    print(f"D vs C (target/2 vs buffer=1 — both with drop):")
    print(f"  P&L: ${d['pnl_per_day'] - 15.66:+.2f}/day")
    print(f"  MaxDD: ${d['max_dd'] - 980:+.0f}")


if __name__ == "__main__":
    main()
