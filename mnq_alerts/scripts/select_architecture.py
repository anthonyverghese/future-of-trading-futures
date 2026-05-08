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


# V6 per-quarter mean daily P&L in dollars. These MUST be obtained by running the
# existing V6 slippage-aware backtest split by calendar quarter before architecture
# selection — they are the comparison floor for the architecture-selection P&L gate.
# Memory contains rough numbers (Q1'25 ~+$37.63, Q4'25 ~+$10.09, Q1'26 ~+$6.78,
# recent-60d ~+$1.83) but these mix V0/V6 contexts; do not trust them — re-run.
# Replace this placeholder with: {(year, q): mean_daily_pnl_dollars}
V6_PER_QUARTER: dict = {}  # filled in before running select_architecture


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
