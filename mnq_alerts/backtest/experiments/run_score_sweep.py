"""Score-threshold sweep at deployed C config under slippage-aware sim.

The bot's `bot_entry_score()` (in bot_trader.py) computes an integer
score per zone-entry candidate from (level, direction, entry_count,
trend_60m, tick_rate, session_move_pct, range_30m_pct, now_et).
BOT_MIN_SCORE = -99 currently disables the filter.

Memory says under no-slippage every threshold hurts P&L by $4-26/day
(see feedback_scoring_doesnt_work.md). This re-tests under slippage
modeling — the factors might differentiate $/trade where they didn't
differentiate WR.

Variants:
  S-1: BOT_MIN_SCORE = -1  (very lenient)
  S0:  BOT_MIN_SCORE =  0
  S1:  BOT_MIN_SCORE =  1
  S2:  BOT_MIN_SCORE =  2  (strict)

Reference (C, BOT_MIN_SCORE = -99):
  $+15.66/day  MaxDD $980
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.run_slippage_aware_v1 import run, fmt_result


def run_with_score(label, dates, caches, min_score: int):
    """Run C config with the given BOT_MIN_SCORE override."""
    import config as cfg
    import bot_trader as bt_mod
    orig_cfg = cfg.BOT_MIN_SCORE
    orig_bt = bt_mod.BOT_MIN_SCORE
    cfg.BOT_MIN_SCORE = min_score
    bt_mod.BOT_MIN_SCORE = min_score
    try:
        return run(
            label, dates, caches,
            exclude_levels={"FIB_EXT_LO_1.272"},
            simulate_slippage=True,
            entry_limit_buffer_pts_override=1.0,
        )
    finally:
        cfg.BOT_MIN_SCORE = orig_cfg
        bt_mod.BOT_MIN_SCORE = orig_bt


def main():
    print("Loading day caches...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for thr in [-1, 0, 1, 2]:
        print(f"\n{'='*70}\nS{thr}: BOT_MIN_SCORE = {thr}\n{'='*70}", flush=True)
        r = run_with_score(f"S{thr}", dates, caches, thr)
        fmt_result(r)
        results[thr] = r

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  C baseline (no score filter): $+15.66/day  MaxDD $980  trades 5,662")
    for thr, r in results.items():
        delta = r["pnl_per_day"] - 15.66
        print(f"  S{thr:>+}  ($/day, MaxDD, trades):  ${r['pnl_per_day']:+.2f}  "
              f"${r['max_dd']:.0f}  {r['trades']}  (Δ vs C: ${delta:+.2f}/day)")


if __name__ == "__main__":
    main()
