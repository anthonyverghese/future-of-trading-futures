"""Final out-of-time test — touched ONCE, by the architecture chosen on dev."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from _level_simulation import simulate_strategy
from _level_validation import score_fold


@dataclass
class FinalTestResult:
    mean_daily_pnl_dollars: float
    top_decile_lift: float
    weekly_pnl_dollars: dict
    n_positive_weeks: int
    n_total_weeks: int
    passes_v6_gate: bool
    passes_lift_gate: bool
    passes_per_week_gate: bool
    all_pass: bool


def evaluate_final_test(
    events: list[dict], preds: dict, threshold: float,
    v6_mean_daily_pnl_dollars: float,
) -> FinalTestResult:
    """Evaluate the 3 final-test gates."""
    sim = simulate_strategy(events, preds, threshold=threshold)
    daily = sim["daily_pnl_dollars"]
    mean_daily = float(np.mean(list(daily.values()))) if daily else 0.0

    gates = score_fold(events, preds, threshold)

    if daily:
        weekly: dict = {}
        for d, v in daily.items():
            week = pd.Timestamp(d).to_period("W")
            weekly[week] = weekly.get(week, 0.0) + v
        weeks_sorted = sorted(weekly.keys())
        last_4 = weeks_sorted[-4:]
        positives = sum(1 for w in last_4 if weekly[w] > 0)
    else:
        weekly = {}
        last_4 = []
        positives = 0

    passes_v6 = mean_daily > v6_mean_daily_pnl_dollars
    passes_lift = gates.top_decile_lift > 0.05
    passes_weeks = positives >= 3 and len(last_4) >= 4

    return FinalTestResult(
        mean_daily_pnl_dollars=mean_daily,
        top_decile_lift=gates.top_decile_lift,
        weekly_pnl_dollars={str(w): v for w, v in weekly.items()},
        n_positive_weeks=positives,
        n_total_weeks=len(last_4),
        passes_v6_gate=passes_v6,
        passes_lift_gate=passes_lift,
        passes_per_week_gate=passes_weeks,
        all_pass=passes_v6 and passes_lift and passes_weeks,
    )
