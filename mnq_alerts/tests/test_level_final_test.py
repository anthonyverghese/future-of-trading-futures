"""Tests for final out-of-time test runner."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_final_test import evaluate_final_test, FinalTestResult


def _synth_test_data():
    base = pd.Timestamp("2026-04-09", tz="UTC")
    events = []
    preds = {}
    for d in range(30):
        for i in range(3):
            ts = base + pd.Timedelta(days=d, seconds=3600 * (i + 1))
            for direction in ("bounce", "breakthrough"):
                for tp, sl in [(8, 25), (8, 20), (10, 25), (10, 20)]:
                    events.append({
                        "event_ts": ts, "level_name": "FIB_0.618",
                        "direction": direction, "tp": tp, "sl": sl,
                        "event_price": 18000.0, "approach_direction": 1,
                        "label": 1, "time_to_resolution_sec": 60.0,
                    })
            preds[ts] = {("bounce", 8, 25): 0.7, **{k: 0.5 for k in [
                ("bounce", 8, 20), ("bounce", 10, 25), ("bounce", 10, 20),
                ("breakthrough", 8, 25), ("breakthrough", 8, 20),
                ("breakthrough", 10, 25), ("breakthrough", 10, 20)]}}
    return events, preds


def test_evaluate_final_test_returns_pass_fail_per_gate():
    events, preds = _synth_test_data()
    result = evaluate_final_test(events, preds, threshold=0.0, v6_mean_daily_pnl_dollars=0.0)
    assert isinstance(result, FinalTestResult)
    assert hasattr(result, "passes_v6_gate")
    assert hasattr(result, "passes_lift_gate")
    assert hasattr(result, "passes_per_week_gate")
    assert hasattr(result, "all_pass")
