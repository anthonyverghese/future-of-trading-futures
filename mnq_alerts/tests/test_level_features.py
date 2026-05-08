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


from _level_features import compute_aggressor


def test_aggressor_balance_buy_dominant():
    # All upticks → buy_aggressor; balance = +1
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.5, 101.0, 101.5], "size": [1, 1, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2, 3)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    assert feats["aggressor_balance_5s"] == pytest.approx(1.0)


def test_aggressor_balance_sell_dominant():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [101.5, 101.0, 100.5, 100.0], "size": [1, 1, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2, 3)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    assert feats["aggressor_balance_5s"] == pytest.approx(-1.0)


def test_aggressor_zero_tick_inherits_prior_side():
    # First uptick (buy), then zero-tick (inherits buy), then zero-tick (inherits buy)
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.5, 100.5, 100.5], "size": [1, 1, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2, 3)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    # 3 buys (uptick + 2 inherited), 0 sells, first tick is neutral by convention
    assert feats["aggressor_balance_5s"] > 0.5


def test_net_dollar_flow_5min_signed():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.5, 101.0], "size": [10, 10, 10]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 60, 120)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    # 2 upticks of 10 contracts each at price 100.5 and 101.0 → positive net flow
    assert feats["net_dollar_flow_5min"] > 0


from _level_features import compute_volume_profile


def test_volume_5s_sums_size():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.0, 100.0], "size": [3, 5, 7]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    assert feats["volume_5s"] == 15


def test_max_print_size_30s():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0] * 4, "size": [1, 50, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 5, 10, 15)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    assert feats["max_print_size_30s"] == 50


def test_volume_concentration_high_when_one_big_print():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    # 99 of size 1, 1 of size 100. Herfindahl ≈ (100^2) / (199^2) ≈ 0.252
    sizes = [1] * 99 + [100]
    ticks = pd.DataFrame(
        {"price": [100.0] * 100, "size": sizes},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s * 0.1) for s in range(100)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    assert feats["volume_concentration_30s"] > 0.2


def test_volume_concentration_low_when_uniform():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0] * 30, "size": [1] * 30},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in range(30)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    # 30 trades of size 1 → Herfindahl = 30 / 900 = 0.0333
    assert feats["volume_concentration_30s"] < 0.05


from _level_features import compute_level_context


def test_touches_today_zero_for_first():
    feats = compute_level_context(
        prior_touches=[],
        all_levels={"FIB_0.618": 100.0, "VWAP": 99.0, "FIB_0.236": 95.0},
        event_ts=pd.Timestamp("2025-06-01 14:31:00", tz="UTC"),
        event_price=100.0,
        level_name="FIB_0.618",
    )
    assert feats["touches_today"] == 0
    assert feats["prior_touch_outcome"] == "none"


def test_prior_touch_outcome_only_uses_resolved():
    """Critical leakage protection — prior touch with resolution_ts > event_ts
    must NOT contribute to prior_touch_outcome."""
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    prior_touches = [
        # Touch at 14:31:00, resolved at 14:46:00 (15min later) — resolved BEFORE current event.
        {"event_ts": base, "resolution_ts": base + pd.Timedelta(minutes=15), "outcome": "bounce_held"},
        # Touch at 14:50:00, resolution at 15:05:00 — UNRESOLVED relative to current event at 14:55.
        {"event_ts": base + pd.Timedelta(minutes=19), "resolution_ts": base + pd.Timedelta(minutes=34), "outcome": "breakthrough_held"},
    ]
    event_ts = base + pd.Timedelta(minutes=24)  # 14:55
    feats = compute_level_context(
        prior_touches=prior_touches,
        all_levels={"FIB_0.618": 100.0, "VWAP": 99.0},
        event_ts=event_ts,
        event_price=100.0,
        level_name="FIB_0.618",
    )
    # The unresolved second touch must NOT influence prior_touch_outcome.
    assert feats["prior_touch_outcome"] == "bounce_held"
    # touches_today counts touches whose event_ts < current event_ts (regardless of resolution).
    assert feats["touches_today"] == 2


def test_distance_to_vwap_excludes_vwap_from_other_levels():
    feats = compute_level_context(
        prior_touches=[],
        all_levels={"FIB_0.618": 100.0, "VWAP": 99.0, "FIB_0.236": 95.0},
        event_ts=pd.Timestamp("2025-06-01 14:31:00", tz="UTC"),
        event_price=100.0,
        level_name="FIB_0.618",
    )
    assert feats["distance_to_vwap"] == pytest.approx(1.0)
    # nearest other level (excluding self & VWAP) is FIB_0.236 at 95
    assert feats["distance_to_nearest_other_level"] == pytest.approx(5.0)
