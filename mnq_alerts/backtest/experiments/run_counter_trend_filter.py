"""Real-sim: counter-trend valley filter (with_trend ∈ [-30, -15] skipped).

Walk-forward 4-quarter validation showed this bucket is reliably bad
(negative $/tr in all 4 quarters, $/tr > $0.50 below mean in all 4).
Real-sim confirms whether the filter survives cap-budget effects.

Configs:
  Baseline: C + IBH=0.75 (current deployed)        — buffer dict, no filter
  V1:       C + IBH=0.75 + valley filter (-30, -15) — adds the filter
  V2:       C + IBH=0.75 + valley filter (-25, -15) — slightly tighter
  V3:       C + IBH=0.75 + valley filter (-30, -10) — slightly wider
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.run_slippage_aware_v1 import run, fmt_result


PER_LEVEL_BUFFER = {"IBH": 0.75}


def main():
    print("Loading day caches...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    results = {}

    for label, valley in [
        ("baseline_no_filter",   None),
        ("V1_valley_-30_-15",    (-30.0, -15.0)),
        ("V2_valley_-25_-15",    (-25.0, -15.0)),
        ("V3_valley_-30_-10",    (-30.0, -10.0)),
    ]:
        print(f"\n{'='*70}\n{label}: filter={valley}\n{'='*70}")
        # We need to pass counter_trend_valley_filter to simulate_day_v2.
        # The shared run() helper doesn't take it directly, so we patch
        # cfg/bt_mod for the duration of this run.
        if valley is not None:
            import config as cfg
            import bot_trader as bt_mod
            cfg.BOT_COUNTER_TREND_VALLEY_FILTER = valley
            bt_mod.BOT_COUNTER_TREND_VALLEY_FILTER = valley
        try:
            r = run(
                label, dates, caches,
                exclude_levels={"FIB_EXT_LO_1.272"},
                simulate_slippage=True,
                entry_limit_buffer_pts_override=PER_LEVEL_BUFFER,
            )
            fmt_result(r)
            results[label] = r
        finally:
            if valley is not None:
                cfg.BOT_COUNTER_TREND_VALLEY_FILTER = None
                bt_mod.BOT_COUNTER_TREND_VALLEY_FILTER = None

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    base = results["baseline_no_filter"]
    print(f"  baseline (no filter):     ${base['pnl_per_day']:+.2f}/day  "
          f"MaxDD ${base['max_dd']:.0f}  trades {base['trades']}")
    for label in ["V1_valley_-30_-15", "V2_valley_-25_-15", "V3_valley_-30_-10"]:
        r = results.get(label)
        if r is None:
            continue
        delta = r['pnl_per_day'] - base['pnl_per_day']
        delta_dd = r['max_dd'] - base['max_dd']
        print(f"  {label:<24} ${r['pnl_per_day']:+.2f}/day  "
              f"MaxDD ${r['max_dd']:.0f}  trades {r['trades']}  "
              f"(Δ ${delta:+.2f}/day, MaxDD ${delta_dd:+.0f})")


if __name__ == "__main__":
    main()
