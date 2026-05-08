"""Tests for _level_simulation."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_simulation import simulate_strategy


def _ev(ts, level, direction, tp, sl, label, ttr, day=0):
    base = pd.Timestamp("2025-06-02", tz="UTC")
    return {
        "event_ts": base + pd.Timedelta(days=day) + pd.Timedelta(seconds=ts),
        "level_name": level, "direction": direction, "tp": tp, "sl": sl,
        "label": label, "time_to_resolution_sec": ttr,
        "event_price": 18000.0, "approach_direction": 1,
    }


def test_skip_when_max_expected_pnl_below_threshold():
    events = [_ev(60, "FIB_0.618", "bounce", 8, 25, 1, 30)]
    # P(win) low for all 8 variants; expected P&L negative for all.
    preds = {(events[0]["event_ts"]): {("bounce", 8, 25): 0.5, ("bounce", 8, 20): 0.5,
                                       ("bounce", 10, 25): 0.5, ("bounce", 10, 20): 0.5,
                                       ("breakthrough", 8, 25): 0.5, ("breakthrough", 8, 20): 0.5,
                                       ("breakthrough", 10, 25): 0.5, ("breakthrough", 10, 20): 0.5}}
    result = simulate_strategy(events, preds, threshold=2.0)
    assert result["trades"] == 0


def test_picks_variant_with_highest_expected_pnl():
    events = [_ev(60, "FIB_0.618", "bounce", 8, 25, 1, 30)]
    # bounce_8_25: 8*0.95 - 25*0.05 = 7.6 - 1.25 = 6.35 — highest
    preds = {(events[0]["event_ts"]): {("bounce", 8, 25): 0.95, ("bounce", 8, 20): 0.5,
                                       ("bounce", 10, 25): 0.5, ("bounce", 10, 20): 0.5,
                                       ("breakthrough", 8, 25): 0.5, ("breakthrough", 8, 20): 0.5,
                                       ("breakthrough", 10, 25): 0.5, ("breakthrough", 10, 20): 0.5}}
    result = simulate_strategy(events, preds, threshold=0.0)
    assert result["trades"] == 1
    assert result["chosen_variants"][0] == ("bounce", 8, 25)


def test_one_position_at_a_time_blocks_overlapping_events():
    # Trade enters at t=60s, resolves at t=60+1800=1860s (30 min). Second event at t=300s should be blocked.
    events = [
        _ev(60, "FIB_0.618", "bounce", 8, 25, 1, 1800),
        _ev(300, "FIB_0.236", "bounce", 8, 25, 1, 30),  # would-be-second trade, blocked.
    ]
    preds = {
        events[0]["event_ts"]: {("bounce", 8, 25): 0.95, **{k: 0.5 for k in [("bounce", 8, 20),
            ("bounce", 10, 25), ("bounce", 10, 20), ("breakthrough", 8, 25), ("breakthrough", 8, 20),
            ("breakthrough", 10, 25), ("breakthrough", 10, 20)]}},
        events[1]["event_ts"]: {("bounce", 8, 25): 0.95, **{k: 0.5 for k in [("bounce", 8, 20),
            ("bounce", 10, 25), ("bounce", 10, 20), ("breakthrough", 8, 25), ("breakthrough", 8, 20),
            ("breakthrough", 10, 25), ("breakthrough", 10, 20)]}},
    }
    result = simulate_strategy(events, preds, threshold=0.0)
    assert result["trades"] == 1
