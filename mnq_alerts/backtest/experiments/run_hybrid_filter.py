"""Real-sim: hybrid pre-filter using human's composite_score >= threshold.

The human-side composite_score (scoring.py) is the production-validated
alert-quality scoring system (82.5% WR at 5.6 alerts/day with >=5,
318-day walk-forward). Memory says human weights don't transfer to bot
at 1pt entry under no-slippage; this re-tests that under slippage.

Configs (all on top of C+IBH=0.75):
  baseline_no_filter
  hybrid_min_3   — lenient (catches a lot)
  hybrid_min_5   — human's deployed threshold
  hybrid_min_7   — strict (Elite tier)
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

    for label, threshold in [
        ("baseline_no_filter", None),
        ("hybrid_min_3",       3),
        ("hybrid_min_5",       5),
        ("hybrid_min_7",       7),
    ]:
        print(f"\n{'='*70}\n{label}: BOT_HYBRID_MIN_COMPOSITE_SCORE={threshold}\n{'='*70}")
        if threshold is not None:
            import config as cfg
            import bot_trader as bt_mod
            cfg.BOT_HYBRID_MIN_COMPOSITE_SCORE = threshold
            bt_mod.BOT_HYBRID_MIN_COMPOSITE_SCORE = threshold
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
            if threshold is not None:
                cfg.BOT_HYBRID_MIN_COMPOSITE_SCORE = None
                bt_mod.BOT_HYBRID_MIN_COMPOSITE_SCORE = None

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    base = results["baseline_no_filter"]
    print(f"  baseline:                ${base['pnl_per_day']:+.2f}/day  "
          f"MaxDD ${base['max_dd']:.0f}  trades {base['trades']}")
    for thr in [3, 5, 7]:
        r = results[f"hybrid_min_{thr}"]
        delta = r['pnl_per_day'] - base['pnl_per_day']
        delta_dd = r['max_dd'] - base['max_dd']
        print(f"  hybrid >= {thr}:           ${r['pnl_per_day']:+.2f}/day  "
              f"MaxDD ${r['max_dd']:.0f}  trades {r['trades']}  "
              f"(Δ ${delta:+.2f}/day, MaxDD ${delta_dd:+.0f})")


if __name__ == "__main__":
    main()
