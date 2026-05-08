"""CLI: run the final out-of-time test on the chosen architecture.

ONLY run this once architecture selection has chosen A or B. Touching this
script during development is the cardinal validation sin.

Usage: python -m mnq_alerts.scripts.run_final_test --arch A
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_validation import FINAL_TEST_TRADING_DAYS
from _level_final_test import evaluate_final_test


# V6 mean daily P&L over the final 30-day window — set when ready.
V6_FINAL_MEAN_DAILY = 1.83  # placeholder; replace with actual 30-day backtest result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["A", "B"], required=True)
    args = parser.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = pd.read_parquet(os.path.join(here, "_level_events_labeled.parquet"))
    days = pd.DatetimeIndex(sorted(dataset["event_ts"].dt.date.unique()))
    final_test_start = days[-FINAL_TEST_TRADING_DAYS]
    train = dataset[dataset["event_ts"].dt.date < final_test_start.date()]
    test = dataset[dataset["event_ts"].dt.date >= final_test_start.date()]

    val_cutoff = sorted(train["event_ts"].dt.date.unique())[-max(1, len(train["event_ts"].dt.date.unique()) // 20)]
    val = train[train["event_ts"].dt.date >= val_cutoff]
    train_for_fit = train[train["event_ts"].dt.date < val_cutoff]

    if args.arch == "A":
        from _level_model import train_architecture_a, predict_architecture_a
        models = train_architecture_a(train_for_fit, val)
        preds = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds[ev["event_ts"]] = predict_architecture_a(models, ev.to_dict())
    else:
        from _level_model import train_architecture_b, predict_architecture_b
        model = train_architecture_b(train_for_fit, val)
        preds = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds[ev["event_ts"]] = predict_architecture_b(model, ev.to_dict())

    result = evaluate_final_test(
        events=test.to_dict("records"), preds=preds, threshold=0.0,
        v6_mean_daily_pnl_dollars=V6_FINAL_MEAN_DAILY,
    )
    print(json.dumps({
        "mean_daily_pnl": result.mean_daily_pnl_dollars,
        "top_decile_lift": result.top_decile_lift,
        "n_positive_weeks": result.n_positive_weeks,
        "passes_v6_gate": result.passes_v6_gate,
        "passes_lift_gate": result.passes_lift_gate,
        "passes_per_week_gate": result.passes_per_week_gate,
        "all_pass": result.all_pass,
        "weekly_pnl": result.weekly_pnl_dollars,
    }, indent=2))


if __name__ == "__main__":
    main()
