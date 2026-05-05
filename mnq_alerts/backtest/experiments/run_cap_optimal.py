"""Real-sim of the cap-optimization variant from per_level_entry_count.py.

Combined cap changes (relative to deployed C config):
  FIB_0.236: 18 → 16  (Monday 32 → 32)
  FIB_0.618:  3 →  6  (Monday  6 → 12)
  FIB_0.764:  5 →  8  (Monday 10 → 16)
  FIB_EXT_HI: 6 → 10  (Monday 12 → 20)
  IBH:        7        (unchanged — already optimal in cum table)
  FIB_EXT_LO: excluded (deployed C)

Reference (from previous slippage-aware runs):
  Slippage-blind control:               $+49.16/day  MaxDD $586
  A: target/2 all levels:                $+14.68/day  MaxDD $1,599
  C: buffer=1 + drop FIB_EXT_LO:         $+15.66/day  MaxDD $980 ← deployed
  D: target/2 + drop FIB_EXT_LO:         $+16.32/day  MaxDD $1,544

Post-process upper bound for this variant: ~+$19.66/day (estimate,
likely optimistic — doesn't model cap-budget interactions).
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

    # Override BASE_CAPS for this run via monkey-patch on the run module
    from mnq_alerts.backtest.experiments import run_slippage_aware_v1 as r
    orig_caps = r.BASE_CAPS
    r.BASE_CAPS = {
        "FIB_0.236": 16,           # was 18
        "FIB_0.618": 6,            # was 3
        "FIB_0.764": 8,            # was 5
        "FIB_EXT_HI_1.272": 10,    # was 6
        "FIB_EXT_LO_1.272": 6,     # excluded anyway, kept for safety
        "IBH": 7,                  # unchanged
    }

    print("\n" + "=" * 70)
    print("CAP-OPT: combined cap optimization, slippage-aware, drop FIB_EXT_LO")
    print("=" * 70)
    print(f"  Caps: {r.BASE_CAPS}", flush=True)
    try:
        result = run(
            "CAP_OPT_v1", dates, caches,
            exclude_levels={"FIB_EXT_LO_1.272"},
            simulate_slippage=True,
            entry_limit_buffer_pts_override=1.0,
        )
        fmt_result(result)
    finally:
        r.BASE_CAPS = orig_caps  # restore

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  C (deployed): $+15.66/day  MaxDD $980")
    print(f"  CAP_OPT:      ${result['pnl_per_day']:+.2f}/day  MaxDD ${result['max_dd']:.0f}")
    print(f"  Δ vs C:       ${result['pnl_per_day'] - 15.66:+.2f}/day  "
          f"MaxDD ${result['max_dd'] - 980:+.0f}")
    print(f"  Post-process estimate was ~+$19.66/day (likely optimistic).")


if __name__ == "__main__":
    main()
