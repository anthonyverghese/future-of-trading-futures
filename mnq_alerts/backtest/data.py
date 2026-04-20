"""Data loading and per-tick factor precomputation.

Precomputes scoring factor arrays once per day so they can be reused
across different zone/scoring/T/S configurations without recomputation.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pytz

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from targeted_backtest import DayCache, load_cached_days, load_day, preprocess_day

_ET = pytz.timezone("America/New_York")


@dataclass
class DayArrays:
    """Precomputed per-tick factor arrays for one day."""
    tick_rates: np.ndarray       # trades/min in 3-min window
    range_30m_pts: np.ndarray    # 30-min high-low range in points
    approach_speed: np.ndarray   # pts/sec in last 10s
    tick_density: np.ndarray     # ticks/sec in last 10s
    et_mins: np.ndarray          # ET minutes since midnight
    session_move: np.ndarray     # price - first_price (points)


def load_all_days() -> tuple[list[datetime.date], dict[datetime.date, DayCache]]:
    """Load all cached days. Returns (sorted_dates, day_caches)."""
    days = load_cached_days()
    caches = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                caches[date] = dc
        except Exception:
            pass
    valid = sorted(caches.keys())
    return valid, caches


def precompute_arrays(dc: DayCache) -> DayArrays:
    """Precompute all scoring factor arrays for one day. O(n) per factor."""
    fp = dc.full_prices
    ft = dc.full_ts_ns
    nf = len(fp)
    s = dc.post_ib_start_idx
    fp0 = float(dc.post_ib_prices[0])

    # Tick rate (3-min sliding window).
    tr = np.zeros(nf, dtype=np.float64)
    left = s
    for i in range(s, nf):
        w = ft[i] - np.int64(180_000_000_000)
        while left < i and ft[left] < w:
            left += 1
        tr[i] = (i - left) / 3.0

    # 30-min range in points.
    r30 = np.zeros(nf, dtype=np.float64)
    for i in range(s, nf):
        ws = int(np.searchsorted(ft, ft[i] - np.int64(1_800_000_000_000), side="left"))
        if ws < i:
            wp = fp[ws : i + 1]
            r30[i] = float(np.max(wp) - np.min(wp))

    # Approach speed + tick density (10s window).
    asp = np.zeros(nf, dtype=np.float64)
    td = np.zeros(nf, dtype=np.float64)
    l10 = s
    for i in range(s, nf):
        w10 = ft[i] - np.int64(10_000_000_000)
        while l10 < i and ft[l10] < w10:
            l10 += 1
        if l10 < i:
            elapsed = (ft[i] - ft[l10]) / 1e9
            asp[i] = abs(float(fp[i]) - float(fp[l10])) / max(elapsed, 0.1)
            td[i] = (i - l10) / 10.0

    # ET minutes since midnight.
    dt_local = _ET.localize(datetime.datetime.combine(dc.date, datetime.time(12, 0)))
    utc_off = np.int64(dt_local.utcoffset().total_seconds() * 1e9)
    em = ((ft + utc_off) // 60_000_000_000 % 1440).astype(np.int32)

    # Session move (points).
    sm = np.zeros(nf, dtype=np.float64)
    for i in range(s, nf):
        sm[i] = float(fp[i]) - fp0

    return DayArrays(tr, r30, asp, td, em, sm)
