"""Tests for _level_dataset.build_day."""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_dataset import build_day, LEVELS_IN_SCOPE


def _make_synthetic_day():
    """Synthesize one trading day's tick data with a clean approach to FIB_0.618."""
    base = pd.Timestamp("2025-06-02 13:30:00", tz="UTC")  # 9:30 ET
    times = [base + pd.Timedelta(seconds=s) for s in range(0, 23400, 30)]  # 6.5 hr at 30s
    # Constant price 18000 → IB locks at 10:31 with H=L=18000 → FIB levels collapse.
    # Use a slow ramp.
    prices = [18000 + (s / 60.0) * 0.5 for s in range(0, 23400, 30)]
    df = pd.DataFrame({"price": prices, "size": [1] * len(times)}, index=pd.DatetimeIndex(times))
    return df


def test_build_day_returns_dataset_with_expected_schema():
    ticks = _make_synthetic_day()
    out = build_day(ticks)
    expected_cols = {"event_ts", "level_name", "level_price", "event_price",
                     "approach_direction", "direction", "tp", "sl", "label",
                     "time_to_resolution_sec"}
    assert expected_cols.issubset(set(out.columns))


def test_build_day_excludes_vwap():
    ticks = _make_synthetic_day()
    out = build_day(ticks)
    assert "VWAP" not in out["level_name"].unique()


def test_levels_in_scope_excludes_vwap():
    assert "VWAP" not in LEVELS_IN_SCOPE
    assert "FIB_0.618" in LEVELS_IN_SCOPE
    assert "IBH" in LEVELS_IN_SCOPE
    assert "IBL" in LEVELS_IN_SCOPE
