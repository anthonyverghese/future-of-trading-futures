"""Tests for _level_features."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_features import compute_kinematics


def _ticks(rows):
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s, _ in rows])
    return pd.DataFrame({"price": [p for _, p in rows], "size": [1] * len(rows)}, index=idx)


def test_velocity_5s_is_signed_points_per_second():
    # Price at 100 at t=-5s, 110 at t=0s → velocity = (110-100)/5 = 2.0
    ticks = _ticks([(0, 100.0), (5, 110.0)])
    event_ts = ticks.index[1]
    feats = compute_kinematics(ticks, event_ts)
    assert abs(feats["velocity_5s"] - 2.0) < 1e-9


def test_velocity_negative_when_price_falls():
    ticks = _ticks([(0, 110.0), (5, 100.0)])
    event_ts = ticks.index[1]
    feats = compute_kinematics(ticks, event_ts)
    assert feats["velocity_5s"] < 0


def test_path_efficiency_one_for_straight_line():
    # Monotonic increase: displacement == sum(|moves|), efficiency = 1.0
    ticks = _ticks([(0, 100.0), (60, 102.0), (120, 104.0), (180, 106.0), (240, 108.0), (300, 110.0)])
    event_ts = ticks.index[-1]
    feats = compute_kinematics(ticks, event_ts)
    assert abs(feats["path_efficiency_5min"] - 1.0) < 1e-6


def test_path_efficiency_low_for_choppy_move():
    # Up-down-up-down — small displacement, large total moves
    ticks = _ticks([(0, 100.0), (60, 105.0), (120, 100.0), (180, 105.0), (240, 100.0), (300, 100.5)])
    event_ts = ticks.index[-1]
    feats = compute_kinematics(ticks, event_ts)
    assert feats["path_efficiency_5min"] < 0.1


def test_no_future_data_used():
    # Future ticks after event_ts must NOT influence kinematics.
    ticks = _ticks([(0, 100.0), (5, 110.0), (10, 200.0)])  # last tick is "future"
    event_ts = ticks.index[1]
    feats = compute_kinematics(ticks, event_ts)
    # If we leaked the 10s tick, velocity_5s would skew toward 200.
    assert abs(feats["velocity_5s"] - 2.0) < 1e-9
