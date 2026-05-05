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


def load_all_days(
    *, verbose: bool = True
) -> tuple[list[datetime.date], dict[datetime.date, DayCache]]:
    """Load all cached days. Returns (sorted_dates, day_caches).

    Prints progress every ~50 days when `verbose` (default) so a
    long-running cache load isn't silent for ~3 minutes.
    """
    import time as _time
    days = load_cached_days()
    caches = {}
    t0 = _time.time()
    if verbose:
        print(f"  loading {len(days)} day caches...", flush=True)
    for i, date in enumerate(days):
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                caches[date] = dc
        except Exception:
            pass
        if verbose and (i + 1) % 50 == 0:
            elapsed = _time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(days) - i - 1) / rate
            print(
                f"    {i+1}/{len(days)} loaded "
                f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                flush=True,
            )
    valid = sorted(caches.keys())
    return valid, caches


_ARRAY_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".array_cache")


def precompute_arrays(dc: DayCache) -> DayArrays:
    """Precompute all scoring factor arrays for one day. O(n) per factor.

    Caches to disk so subsequent runs skip the expensive computation.
    """
    os.makedirs(_ARRAY_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_ARRAY_CACHE_DIR, f"{dc.date}.npz")
    if os.path.exists(cache_path):
        try:
            d = np.load(cache_path)
            return DayArrays(d["tr"], d["r30"], d["asp"], d["td"], d["em"], d["sm"])
        except Exception:
            pass  # recompute if cache is corrupt
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

    # Cache to disk.
    try:
        np.savez_compressed(cache_path, tr=tr, r30=r30, asp=asp, td=td, em=em, sm=sm)
    except Exception:
        pass

    return DayArrays(tr, r30, asp, td, em, sm)
