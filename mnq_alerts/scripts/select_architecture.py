"""CLI: run architecture selection on the labeled dataset.

Usage: python -m mnq_alerts.scripts.select_architecture
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_validation import build_walk_forward_folds, run_architecture_selection


# V6 per-quarter mean daily P&L in dollars (CALENDAR quarters).
# APPROXIMATIONS from saved /tmp/filter_audit/V6_no_monday_caps.json sorted-day
# quartiles + partial-run data + memory snapshots. The proper calendar-quarter
# V6 backtest didn't complete (system thrashed on swap with 339 caches in RAM).
# These estimates are within ~±$5/day; if a model's edge over V6 is in that
# range the result is borderline and we should re-run V6 with calendar-quarter
# aggregation. Sources:
#   - Saved V6 sorted-quartiles: Q1=$33.36, Q2=$18.39, Q3=$8.81, Q4=$10.91 (336d)
#   - Partial calendar V6: day-50 $/day=$38.36, day-100=$29.39, day-150=$28.10
#   - Memory: recent-60d ~ +$1.83, Q1'26 ~ +$6.78
V6_PER_QUARTER: dict = {
    (2025, 2): 22.0,   # Q2'25 - partial V6 + saved-quartile blend
    (2025, 3): 15.0,   # Q3'25 - saved Q2 quartile aligning with calendar Q3
    (2025, 4): 9.0,    # Q4'25 - saved Q3 quartile
    (2026, 1): 7.0,    # Q1'26 - memory + saved Q4 quartile
}


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = pd.read_parquet(os.path.join(here, "_level_events_labeled.parquet"))
    days = pd.DatetimeIndex(sorted(dataset["event_ts"].dt.date.unique()))
    folds = build_walk_forward_folds(days)
    print(f"Built {len(folds)} folds")
    results = run_architecture_selection(dataset, folds, V6_PER_QUARTER)
    print(json.dumps({arch: {k: v for k, v in r.items() if k != "folds"} for arch, r in results.items()}, indent=2))
    print("--- Per-fold ---")
    for arch in ("A", "B"):
        for f in results[arch]["folds"]:
            g = f["gates"]
            print(f"{arch} fold {f['fold'].test_start.date()}-{f['fold'].test_end.date()}: "
                  f"lift={g.top_decile_lift:.4f} epnl={g.top_decile_expected_pnl_per_trade:.2f} daily=${g.simulated_mean_daily_pnl_dollars:.2f}")


if __name__ == "__main__":
    main()
