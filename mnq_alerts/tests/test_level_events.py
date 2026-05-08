"""Tests for _level_events.extract_events."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_events import extract_events


def _make_ticks(rows):
    """rows = list of (seconds_from_open, price)."""
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")  # 10:31 ET
    idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s, _ in rows])
    return pd.DataFrame({"price": [p for _, p in rows], "size": [1] * len(rows)}, index=idx)


def test_emits_event_when_price_within_1pt():
    levels = {"FIB_0.618": 100.0}
    ticks = _make_ticks([(0, 105), (1, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 1
    assert events.iloc[0]["level_name"] == "FIB_0.618"
    assert abs(events.iloc[0]["event_price"] - 100.5) < 1e-9


def test_arming_disarms_after_event():
    levels = {"FIB_0.618": 100.0}
    # Two ticks within 1pt — should fire only the first; second is suppressed.
    ticks = _make_ticks([(0, 105), (1, 100.5), (2, 100.7)])
    events = extract_events(ticks, levels)
    assert len(events) == 1


def test_rearming_requires_3pt_exit():
    levels = {"FIB_0.618": 100.0}
    # Enter zone, drift up to 102 (still within 3pt), back to level — should NOT re-fire.
    ticks = _make_ticks([(0, 105), (1, 100.5), (2, 102.0), (3, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 1


def test_rearms_after_exiting_3pt_zone():
    levels = {"FIB_0.618": 100.0}
    # Enter, exit beyond 3pt, re-enter → second event fires.
    ticks = _make_ticks([(0, 105), (1, 100.5), (2, 104.0), (3, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 2


def test_levels_armed_only_at_or_after_10_31_et():
    levels = {"FIB_0.618": 100.0}
    base = pd.Timestamp("2025-06-01 14:30:30", tz="UTC")  # 10:30:30 ET, before lock
    idx = pd.DatetimeIndex([base, base + pd.Timedelta(seconds=60)])
    ticks = pd.DataFrame({"price": [100.5, 100.5], "size": [1, 1]}, index=idx)
    events = extract_events(ticks, levels)
    # First tick is before 10:31 ET, suppressed. Second is at 10:31:30, armed.
    assert len(events) == 1
    assert events.iloc[0]["event_ts"] == base + pd.Timedelta(seconds=60)


def test_approach_direction_from_below():
    levels = {"FIB_0.618": 100.0}
    # 60s before event price was 95 → approach from below → +1
    ticks = _make_ticks([(0, 95.0), (60, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 1
    assert events.iloc[0]["approach_direction"] == 1


def test_approach_direction_from_above():
    levels = {"FIB_0.618": 100.0}
    ticks = _make_ticks([(0, 105.0), (60, 100.5)])
    events = extract_events(ticks, levels)
    assert events.iloc[0]["approach_direction"] == -1
