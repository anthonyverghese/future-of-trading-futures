"""Tests for _level_labels.label_events."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_labels import label_events, TP_SL_VARIANTS


def _make_ticks(rows):
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s, _ in rows])
    return pd.DataFrame({"price": [p for _, p in rows], "size": [1] * len(rows)}, index=idx)


def _make_event(ts_offset_sec, level_price, event_price, approach):
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    return pd.DataFrame(
        [{"event_ts": base + pd.Timedelta(seconds=ts_offset_sec), "level_name": "L", "level_price": level_price, "event_price": event_price, "approach_direction": approach}]
    )


def test_bounce_win_when_price_moves_away_from_approach_first():
    # Approach from below (+1). Bounce trade target = price moves DOWN by TP. Stop = UP by SL.
    ticks = _make_ticks([(0, 100.0), (5, 92.0)])  # event at 100, then drops to 92 (move -8 from 100, TP=8 hits)
    events = _make_event(0, 100.0, 100.0, 1)
    labels = label_events(events, ticks)
    bounce_8_25 = labels[(labels["direction"] == "bounce") & (labels["tp"] == 8) & (labels["sl"] == 25)]
    assert bounce_8_25.iloc[0]["label"] == 1


def test_bounce_loss_when_sl_hit_first():
    ticks = _make_ticks([(0, 100.0), (3, 125.5)])  # approach from below; price moves UP 25.5 → SL=25 hits before TP=8
    events = _make_event(0, 100.0, 100.0, 1)
    labels = label_events(events, ticks)
    bounce_8_25 = labels[(labels["direction"] == "bounce") & (labels["tp"] == 8) & (labels["sl"] == 25)]
    assert bounce_8_25.iloc[0]["label"] == 0


def test_breakthrough_win_when_price_moves_with_approach():
    ticks = _make_ticks([(0, 100.0), (5, 110.0)])  # approach from below, price rises 10
    events = _make_event(0, 100.0, 100.0, 1)
    labels = label_events(events, ticks)
    bt_10_25 = labels[(labels["direction"] == "breakthrough") & (labels["tp"] == 10) & (labels["sl"] == 25)]
    assert bt_10_25.iloc[0]["label"] == 1


def test_timeout_15min_counts_as_loss():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    idx = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=20)])  # only 2 ticks, 20 min apart, no movement
    ticks = pd.DataFrame({"price": [100.0, 100.5], "size": [1, 1]}, index=idx)
    events = _make_event(0, 100.0, 100.0, 1)
    labels = label_events(events, ticks)
    for _, row in labels.iterrows():
        assert row["label"] == 0  # all variants timeout-loss


def test_market_close_counts_as_loss():
    # Event at 3:55 PM ET (5 min before close); no TP/SL hit by 4:00 PM.
    base = pd.Timestamp("2025-06-01 19:55:00", tz="UTC")  # 15:55 ET
    idx = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=4)])
    ticks = pd.DataFrame({"price": [100.0, 100.3], "size": [1, 1]}, index=idx)
    event = pd.DataFrame(
        [{"event_ts": base, "level_name": "L", "level_price": 100.0, "event_price": 100.0, "approach_direction": 1}]
    )
    labels = label_events(event, ticks)
    for _, row in labels.iterrows():
        assert row["label"] == 0


def test_eight_label_rows_per_event():
    ticks = _make_ticks([(0, 100.0), (5, 110.0)])
    events = _make_event(0, 100.0, 100.0, 1)
    labels = label_events(events, ticks)
    assert len(labels) == 8  # 2 directions × 4 TP/SL variants


def test_variants_are_exactly_8_25_8_20_10_25_10_20():
    expected = {(8, 25), (8, 20), (10, 25), (10, 20)}
    assert set(TP_SL_VARIANTS) == expected
