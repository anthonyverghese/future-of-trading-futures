"""Tests for walk-forward fold construction."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_validation import build_walk_forward_folds, FoldDef, score_fold, GateResults


def test_five_quarterly_folds_for_full_history():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    # 5 quarters in dev set (Q1-Q4 2025 + partial Q1-2026); first quarter is training-only,
    # so 4 walk-forward test folds (Q2-25, Q3-25, Q4-25, Q1-26).
    assert len(folds) == 4


def test_train_set_grows_each_fold():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    assert all(folds[i].train_end >= folds[i - 1].train_end for i in range(1, len(folds)))


def test_first_day_of_test_dropped_for_embargo():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    fold = folds[0]
    assert fold.test_start > fold.train_end


def test_dev_set_excludes_final_test_window():
    days = pd.date_range("2025-01-02", "2026-05-06", freq="B")
    folds = build_walk_forward_folds(days)
    last_fold = folds[-1]
    final_test_start = days[-30]
    assert last_fold.test_end < final_test_start


def test_score_fold_returns_metrics():
    base = pd.Timestamp("2025-06-02", tz="UTC")
    events = []
    preds = {}
    for i in range(50):
        ts = base + pd.Timedelta(seconds=60 * i + 60)
        for direction in ("bounce", "breakthrough"):
            for tp, sl in [(8, 25), (8, 20), (10, 25), (10, 20)]:
                events.append({
                    "event_ts": ts, "level_name": "FIB_0.618",
                    "direction": direction, "tp": tp, "sl": sl,
                    "event_price": 18000.0, "approach_direction": 1,
                    "label": (i + (direction == "bounce")) % 2,
                    "time_to_resolution_sec": 60.0,
                })
        preds[ts] = {("bounce", 8, 25): 0.6, ("bounce", 8, 20): 0.55, ("bounce", 10, 25): 0.5,
                     ("bounce", 10, 20): 0.5, ("breakthrough", 8, 25): 0.5, ("breakthrough", 8, 20): 0.5,
                     ("breakthrough", 10, 25): 0.5, ("breakthrough", 10, 20): 0.5}
    result = score_fold(events=events, preds=preds, threshold=0.0)
    assert hasattr(result, "top_decile_lift")
    assert hasattr(result, "top_decile_expected_pnl_per_trade")
    assert hasattr(result, "simulated_mean_daily_pnl_dollars")
