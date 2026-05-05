"""
ml_backtest.py — Walk-forward ML model for alert filtering.

Replaces the hand-tuned composite scoring with a trained classifier.
Uses walk-forward validation (not random splits) for honest OOS estimates.

Target: 80%+ win rate at 4+ alerts/day (vs current 81% at 1.6/day).

Usage:
    python ml_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

import json

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    ALERT_THRESHOLD,
    EXIT_THRESHOLD,
    IB_END,
    MARKET_CLOSE,
    MARKET_OPEN,
    STOP_POINTS,
    TARGET_POINTS,
    Alert,
    DayCache,
    _run_zone_numpy,
    evaluate_outcome_np,
    load_cached_days,
    load_day,
)

# Feature extraction window (seconds before alert).
FEATURE_SECS = 3 * 60  # 3 minutes

# Walk-forward parameters.
INITIAL_TRAIN_DAYS = 100
TEST_WINDOW_DAYS = 20

# S/R confluence thresholds.
SR_60M_PTS = 5.0
SR_120M_PTS = 5.0


# ── VIX data (free from Yahoo Finance) ────────────────────────────────────────

VIX_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data_cache", "vix_daily.csv")


def fetch_vix_data(
    start: datetime.date, end: datetime.date
) -> dict[datetime.date, float]:
    """Fetch daily VIX closes from Yahoo Finance. Caches to CSV."""
    # Try cached file first
    if os.path.exists(VIX_CACHE_PATH):
        vix_df = pd.read_csv(VIX_CACHE_PATH)
        vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
        cached_start = vix_df["date"].min()
        cached_end = vix_df["date"].max()
        # Use cache if it covers our range (with 10-day buffer for lookback)
        if cached_start <= start - datetime.timedelta(days=10) and cached_end >= end:
            return dict(zip(vix_df["date"], vix_df["close"]))

    # Fetch from Yahoo Finance
    pad_start = start - datetime.timedelta(days=15)  # extra for lookback
    start_ts = int(datetime.datetime.combine(pad_start, datetime.time()).timestamp())
    end_ts = int(
        datetime.datetime.combine(
            end + datetime.timedelta(days=1), datetime.time()
        ).timestamp()
    )
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
        f"?period1={start_ts}&period2={end_ts}&interval=1d"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]

        rows = []
        for ts, close in zip(timestamps, closes):
            if close is not None:
                d = datetime.datetime.fromtimestamp(ts).date()
                rows.append({"date": d, "close": close})
        vix_df = pd.DataFrame(rows)
        vix_df.to_csv(VIX_CACHE_PATH, index=False)
        print(f"  VIX data: {len(rows)} days fetched and cached.")
        return {r["date"]: r["close"] for r in rows}
    except Exception as e:
        print(f"  Warning: VIX fetch failed ({e}), proceeding without VIX features.")
        return {}


# ── Cross-day features from cached MNQ data ─────────────────────────────────


@dataclass
class DaySummary:
    """End-of-day stats for cross-day feature computation."""

    date: datetime.date
    day_open: float
    day_close: float
    day_high: float
    day_low: float
    day_range: float  # high - low


def compute_day_summaries(
    day_caches: dict[datetime.date, "MLDayCache"],
    days: list[datetime.date],
) -> dict[datetime.date, DaySummary]:
    """Compute end-of-day stats from cached MNQ data for cross-day features."""
    summaries: dict[datetime.date, DaySummary] = {}
    for date in days:
        dc = day_caches.get(date)
        if dc is None:
            continue
        prices = dc.full_prices
        summaries[date] = DaySummary(
            date=date,
            day_open=float(prices[0]),
            day_close=float(prices[-1]),
            day_high=float(np.max(prices)),
            day_low=float(np.min(prices)),
            day_range=float(np.max(prices) - np.min(prices)),
        )
    return summaries


def get_cross_day_features(
    date: datetime.date,
    day_summaries: dict[datetime.date, DaySummary],
    sorted_dates: list[datetime.date],
    vix_data: dict[datetime.date, float],
) -> dict[str, float]:
    """Compute cross-day features for a given date."""
    date_idx = None
    for i, d in enumerate(sorted_dates):
        if d == date:
            date_idx = i
            break
    if date_idx is None:
        return _default_cross_day()

    feats: dict[str, float] = {}

    # Day of week (0=Mon, 4=Fri)
    feats["day_of_week"] = float(date.weekday())

    # Prior day features
    prior = _get_prior_summary(date_idx, sorted_dates, day_summaries)
    if prior is not None:
        feats["prior_day_range"] = prior.day_range
        feats["prior_day_direction"] = (
            1.0 if prior.day_close >= prior.day_open else -1.0
        )
        # Gap: today's open vs yesterday's close
        current = day_summaries.get(date)
        if current is not None:
            feats["gap_size"] = current.day_open - prior.day_close
        else:
            feats["gap_size"] = 0.0
    else:
        feats["prior_day_range"] = 0.0
        feats["prior_day_direction"] = 0.0
        feats["gap_size"] = 0.0

    # Multi-day trend: sum of (close - open) for prior 3 trading days
    trend_3d = 0.0
    count_3d = 0
    for lookback in range(1, 6):  # search up to 5 indices back for 3 trading days
        idx = date_idx - lookback
        if idx < 0:
            break
        s = day_summaries.get(sorted_dates[idx])
        if s is not None:
            trend_3d += s.day_close - s.day_open
            count_3d += 1
            if count_3d >= 3:
                break
    feats["prior_3d_trend"] = trend_3d

    # 5-day average range
    ranges_5d: list[float] = []
    for lookback in range(1, 10):
        idx = date_idx - lookback
        if idx < 0:
            break
        s = day_summaries.get(sorted_dates[idx])
        if s is not None:
            ranges_5d.append(s.day_range)
            if len(ranges_5d) >= 5:
                break
    avg_range_5d = float(np.mean(ranges_5d)) if ranges_5d else 0.0
    feats["avg_range_5d"] = avg_range_5d

    # Range expansion: today's IB range vs 5-day average full range
    current = day_summaries.get(date)
    if current is not None and avg_range_5d > 0:
        feats["range_expansion"] = current.day_range / avg_range_5d
    else:
        feats["range_expansion"] = 1.0

    # VIX features (use prior trading day's close — no lookahead)
    vix_close = None
    for lookback in range(1, 6):
        idx = date_idx - lookback
        if idx < 0:
            break
        prior_date = sorted_dates[idx]
        if prior_date in vix_data:
            vix_close = vix_data[prior_date]
            break
    # Also try the date itself if it's in VIX data (same-day close is available
    # before market open next day, but for intraday we use prior day)
    if vix_close is None and date in vix_data:
        vix_close = vix_data[date]

    if vix_close is not None:
        feats["vix_close"] = vix_close
        # VIX 5-day average
        vix_vals: list[float] = []
        for lookback in range(1, 10):
            idx = date_idx - lookback
            if idx < 0:
                break
            d = sorted_dates[idx]
            if d in vix_data:
                vix_vals.append(vix_data[d])
                if len(vix_vals) >= 5:
                    break
        feats["vix_5d_avg"] = float(np.mean(vix_vals)) if vix_vals else vix_close
        # VIX change from prior day
        if len(vix_vals) >= 1:
            feats["vix_change"] = vix_close - vix_vals[0]
        else:
            feats["vix_change"] = 0.0
    else:
        feats["vix_close"] = 20.0  # neutral default
        feats["vix_5d_avg"] = 20.0
        feats["vix_change"] = 0.0

    return feats


def _get_prior_summary(
    date_idx: int,
    sorted_dates: list[datetime.date],
    summaries: dict[datetime.date, DaySummary],
) -> DaySummary | None:
    """Find the most recent prior day with a summary."""
    for lookback in range(1, 6):
        idx = date_idx - lookback
        if idx < 0:
            return None
        s = summaries.get(sorted_dates[idx])
        if s is not None:
            return s
    return None


def _default_cross_day() -> dict[str, float]:
    """Default cross-day features when no history is available."""
    return {
        "day_of_week": 0.0,
        "prior_day_range": 0.0,
        "prior_day_direction": 0.0,
        "gap_size": 0.0,
        "prior_3d_trend": 0.0,
        "avg_range_5d": 0.0,
        "range_expansion": 1.0,
        "vix_close": 20.0,
        "vix_5d_avg": 20.0,
        "vix_change": 0.0,
    }


# ── Enhanced DayCache with extra arrays for feature extraction ───────────────


@dataclass
class MLDayCache:
    """DayCache extended with arrays needed for ML feature extraction."""

    # Core from DayCache
    date: datetime.date
    ibh: float
    ibl: float
    fib_lo: float
    fib_hi: float
    post_ib_prices: np.ndarray
    post_ib_vwaps: np.ndarray
    post_ib_timestamps: pd.DatetimeIndex
    post_ib_start_idx: int
    full_prices: np.ndarray
    full_ts_ns: np.ndarray
    # ML extras
    full_vwaps: np.ndarray
    full_sizes: np.ndarray
    session_highs: np.ndarray
    session_lows: np.ndarray
    day_open: float
    ib_range: float
    # Order flow (requires re-cached data with 'side' column)
    full_sides: np.ndarray | None  # 1=buy, -1=sell, 0=unknown


def preprocess_day_ml(df: pd.DataFrame, date: datetime.date) -> MLDayCache | None:
    """Extract numpy arrays from a day's DataFrame for ML feature extraction."""
    if df.empty:
        return None
    ib = df[(df.index.time >= MARKET_OPEN) & (df.index.time < IB_END)]
    if ib.empty:
        return None
    ibh = float(ib["price"].max())
    ibl = float(ib["price"].min())

    prices = df["price"].values.astype(np.float64)
    sizes = df["size"].values.astype(np.int64)

    # Order flow: convert side chars to numeric (B=1, A=-1, N=0)
    has_side = "side" in df.columns
    if has_side:
        side_raw = df["side"].values
        sides = np.where(side_raw == "B", 1, np.where(side_raw == "A", -1, 0)).astype(
            np.int8
        )
    else:
        sides = None

    cum_pv = np.cumsum(prices * sizes)
    cum_vol = np.cumsum(sizes)
    vwap_arr = cum_pv / cum_vol

    session_highs = np.maximum.accumulate(prices)
    session_lows = np.minimum.accumulate(prices)

    post_ib_mask = df.index.time >= IB_END
    post_ib = df[post_ib_mask]
    if post_ib.empty:
        return None

    post_ib_start = int(np.argmax(post_ib_mask))
    ib_range = ibh - ibl

    return MLDayCache(
        date=date,
        ibh=ibh,
        ibl=ibl,
        fib_lo=ibl - 0.272 * ib_range,
        fib_hi=ibh + 0.272 * ib_range,
        post_ib_prices=post_ib["price"].values.astype(np.float64),
        post_ib_vwaps=vwap_arr[post_ib_mask].astype(np.float64),
        post_ib_timestamps=post_ib.index,
        post_ib_start_idx=post_ib_start,
        full_prices=prices,
        full_ts_ns=df.index.asi8,
        full_vwaps=vwap_arr,
        full_sizes=sizes,
        session_highs=session_highs,
        session_lows=session_lows,
        day_open=float(prices[0]),
        ib_range=ib_range,
        full_sides=sides,
    )


# ── Feature extraction ───────────────────────────────────────────────────────


def extract_features(
    dc: MLDayCache,
    full_idx: int,
    entry_count: int,
    line_price: float,
    entry_price: float,
    direction: str,
    alerts_fired_today: int,
    last_alert_ns: int | None,
) -> dict | None:
    """Extract features for one alert using numpy arrays. Returns None if insufficient data."""
    ts_ns = dc.full_ts_ns
    prices = dc.full_prices
    sizes = dc.full_sizes
    alert_ns = ts_ns[full_idx]

    # Find approach window: [alert_time - FEATURE_SECS, alert_time]
    window_start_ns = alert_ns - np.int64(FEATURE_SECS * 1_000_000_000)
    win_start_idx = int(np.searchsorted(ts_ns, window_start_ns, side="left"))
    win_end_idx = full_idx + 1  # inclusive of alert tick

    if win_end_idx - win_start_idx < 3:
        return None

    win_prices = prices[win_start_idx:win_end_idx]
    win_sizes = sizes[win_start_idx:win_end_idx]
    n = len(win_prices)

    # Split window into two halves
    mid = n // 2
    first_prices = win_prices[:mid] if mid >= 2 else win_prices[:1]
    second_prices = win_prices[mid:] if n - mid >= 2 else win_prices[-1:]
    first_sizes = win_sizes[:mid]
    second_sizes = win_sizes[mid:]

    is_up = direction == "up"

    def toward(val: float) -> float:
        return -val if is_up else val

    # ── Approach momentum ────────────────────────────────────────────────────
    overall_change = float(win_prices[-1] - win_prices[0])
    approach_momentum = toward(overall_change)

    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, win_prices, 1)[0])
    approach_slope = toward(slope)

    first_change = (
        float(first_prices[-1] - first_prices[0]) if len(first_prices) >= 2 else 0.0
    )
    second_change = (
        float(second_prices[-1] - second_prices[0]) if len(second_prices) >= 2 else 0.0
    )
    approach_first = toward(first_change)
    approach_second = toward(second_change)
    approach_accel = approach_second - approach_first

    volatility = float(np.std(win_prices))
    norm_approach = approach_momentum / volatility if volatility > 1e-9 else 0.0

    # ── Max pullback ─────────────────────────────────────────────────────────
    if is_up:
        max_pullback = max(0.0, float(np.max(win_prices)) - win_prices[0])
    else:
        max_pullback = max(0.0, win_prices[0] - float(np.min(win_prices)))

    # ── Volume / activity ────────────────────────────────────────────────────
    first_vol = int(np.sum(first_sizes))
    second_vol = int(np.sum(second_sizes))
    volume_trend = second_vol - first_vol
    tick_rate = n / (FEATURE_SECS / 60)

    # ── Session context ──────────────────────────────────────────────────────
    session_high_now = float(dc.session_highs[full_idx])
    session_low_now = float(dc.session_lows[full_idx])
    session_move_pts = entry_price - dc.day_open
    dist_from_high = session_high_now - entry_price
    dist_from_low = entry_price - session_low_now

    # ── Time features ────────────────────────────────────────────────────────
    alert_dt = pd.Timestamp(alert_ns, tz="America/New_York")
    alert_mins = alert_dt.hour * 60 + alert_dt.minute
    time_of_day_mins = alert_mins - (9 * 60 + 30)  # minutes since market open
    minutes_since_ib = alert_mins - (10 * 60 + 30)  # minutes since IB lock
    minutes_to_close = (16 * 60) - alert_mins
    is_afternoon = 1 if (13 * 60) <= alert_mins < (15 * 60) else 0
    is_power_hour = 1 if alert_mins >= (15 * 60) else 0

    # ── Trend (30-min lookback) ──────────────────────────────────────────────
    trend_start_ns = alert_ns - np.int64(30 * 60 * 1_000_000_000)
    trend_start_idx = int(np.searchsorted(ts_ns, trend_start_ns, side="left"))
    if trend_start_idx < full_idx:
        trend_30m = float(prices[full_idx] - prices[trend_start_idx])
    else:
        trend_30m = 0.0
    trend_aligned = (
        1.0
        if (direction == "up" and trend_30m > 0)
        or (direction == "down" and trend_30m < 0)
        else -1.0
    )

    # ── Rolling S/R (60m / 120m high/low near level) ─────────────────────────
    def check_sr(lookback_min: int, threshold: float) -> int:
        sr_start_ns = alert_ns - np.int64(lookback_min * 60 * 1_000_000_000)
        sr_start_idx = int(np.searchsorted(ts_ns, sr_start_ns, side="left"))
        if sr_start_idx >= full_idx:
            return 0
        seg = prices[sr_start_idx:full_idx]
        high = float(np.max(seg))
        low = float(np.min(seg))
        return (
            1
            if (
                abs(line_price - high) <= threshold
                or abs(line_price - low) <= threshold
            )
            else 0
        )

    recent_sr_60m = check_sr(60, SR_60M_PTS)
    recent_sr_120m = check_sr(120, SR_120M_PTS)

    # ── Price structure ──────────────────────────────────────────────────────
    ib_mid = (dc.ibh + dc.ibl) / 2.0
    price_vs_ib_midpoint = (
        (entry_price - ib_mid) / dc.ib_range if dc.ib_range > 0 else 0.0
    )
    current_vwap = float(dc.full_vwaps[full_idx])
    price_vs_vwap = entry_price - current_vwap

    # ── Level one-hots ───────────────────────────────────────────────────────
    # (level name will be passed separately, but we need it here)
    # We'll handle this in the caller since we have the level name there.
    # For now, return placeholder — caller fills in.

    # ── Alert fatigue ────────────────────────────────────────────────────────
    if last_alert_ns is not None:
        minutes_since_last = (alert_ns - last_alert_ns) / 1e9 / 60.0
        minutes_since_last = min(minutes_since_last, 120.0)
    else:
        minutes_since_last = 120.0  # cap / first alert

    feats = {
        # Approach momentum (5)
        "approach_momentum": approach_momentum,
        "approach_slope": approach_slope,
        "approach_first": approach_first,
        "approach_second": approach_second,
        "approach_accel": approach_accel,
        # Approach quality (3)
        "norm_approach": norm_approach,
        "volatility": volatility,
        "max_pullback": max_pullback,
        # Volume / activity (2)
        "volume_trend": volume_trend,
        "tick_rate": tick_rate,
        # Time (5)
        "time_of_day_mins": time_of_day_mins,
        "minutes_since_ib": minutes_since_ib,
        "minutes_to_close": minutes_to_close,
        "is_afternoon": is_afternoon,
        "is_power_hour": is_power_hour,
        # Session context (3)
        "session_move_pts": session_move_pts,
        "dist_from_high": dist_from_high,
        "dist_from_low": dist_from_low,
        # Level quality (3)
        "level_test_count": entry_count,
        "recent_sr_60m": recent_sr_60m,
        "recent_sr_120m": recent_sr_120m,
        # Trend (2)
        "trend_30m": trend_30m,
        "trend_aligned_30m": trend_aligned,
        # Entry (1)
        "entry_distance": abs(entry_price - line_price),
        # Price structure (3)
        "ib_range_size": dc.ib_range,
        "price_vs_ib_midpoint": price_vs_ib_midpoint,
        "price_vs_vwap": price_vs_vwap,
        # Alert fatigue (2)
        "alerts_fired_today": alerts_fired_today,
        "minutes_since_last_alert": minutes_since_last,
    }

    # ── Order flow features — multi-timeframe (requires side data) ──────────
    if dc.full_sides is not None:

        def _imbalance(start_i: int, end_i: int) -> tuple[float, float, int]:
            """Compute (volume_imbalance, trade_count_ratio, total_vol) for a slice."""
            seg_sides = dc.full_sides[start_i:end_i]
            seg_sizes = dc.full_sizes[start_i:end_i]
            total = int(np.sum(seg_sizes))
            if total == 0:
                return 0.0, 0.5, 0
            bv = int(np.sum(seg_sizes[seg_sides == 1]))
            sv = int(np.sum(seg_sizes[seg_sides == -1]))
            n_trades = end_i - start_i
            n_buys = int(np.sum(seg_sides == 1))
            return (bv - sv) / total, n_buys / n_trades if n_trades > 0 else 0.5, total

        # 3-minute window (full approach)
        imb_3m, ratio_3m, _ = _imbalance(win_start_idx, win_end_idx)
        feats["imbalance_3m"] = imb_3m
        feats["buy_ratio_3m"] = ratio_3m

        # 60-second window
        ns_60s = alert_ns - np.int64(60 * 1_000_000_000)
        idx_60s = int(np.searchsorted(ts_ns, ns_60s, side="left"))
        idx_60s = max(idx_60s, win_start_idx)
        imb_60s, ratio_60s, _ = _imbalance(idx_60s, win_end_idx)
        feats["imbalance_60s"] = imb_60s
        feats["buy_ratio_60s"] = ratio_60s

        # 30-second window
        ns_30s = alert_ns - np.int64(30 * 1_000_000_000)
        idx_30s = int(np.searchsorted(ts_ns, ns_30s, side="left"))
        idx_30s = max(idx_30s, win_start_idx)
        imb_30s, ratio_30s, _ = _imbalance(idx_30s, win_end_idx)
        feats["imbalance_30s"] = imb_30s
        feats["buy_ratio_30s"] = ratio_30s

        # 10-second window (micro flow right at the touch)
        ns_10s = alert_ns - np.int64(10 * 1_000_000_000)
        idx_10s = int(np.searchsorted(ts_ns, ns_10s, side="left"))
        idx_10s = max(idx_10s, win_start_idx)
        imb_10s, ratio_10s, _ = _imbalance(idx_10s, win_end_idx)
        feats["imbalance_10s"] = imb_10s
        feats["buy_ratio_10s"] = ratio_10s

        # Imbalance acceleration: 30s vs prior 30s
        ns_60s_start = alert_ns - np.int64(60 * 1_000_000_000)
        idx_60s_start = int(np.searchsorted(ts_ns, ns_60s_start, side="left"))
        idx_60s_start = max(idx_60s_start, win_start_idx)
        imb_prior_30s, _, _ = _imbalance(idx_60s_start, idx_30s)
        feats["imbalance_accel_30s"] = imb_30s - imb_prior_30s

        # Large trade imbalance (≥3 contracts) in last 60s
        seg_sides_60 = dc.full_sides[idx_60s:win_end_idx]
        seg_sizes_60 = dc.full_sizes[idx_60s:win_end_idx]
        large_mask = seg_sizes_60 >= 3
        large_sides = seg_sides_60[large_mask]
        large_sizes = seg_sizes_60[large_mask]
        large_total = int(np.sum(large_sizes))
        if large_total > 0:
            lb = int(np.sum(large_sizes[large_sides == 1]))
            ls = int(np.sum(large_sizes[large_sides == -1]))
            feats["large_imbalance_60s"] = (lb - ls) / large_total
        else:
            feats["large_imbalance_60s"] = 0.0

        # Aggressor aligned with trade direction (using 30s window)
        if is_up:
            feats["aggressor_aligned_30s"] = imb_30s
            feats["aggressor_aligned_10s"] = imb_10s
        else:
            feats["aggressor_aligned_30s"] = -imb_30s
            feats["aggressor_aligned_10s"] = -imb_10s
    else:
        for k in [
            "imbalance_3m",
            "buy_ratio_3m",
            "imbalance_60s",
            "buy_ratio_60s",
            "imbalance_30s",
            "buy_ratio_30s",
            "imbalance_10s",
            "buy_ratio_10s",
            "imbalance_accel_30s",
            "large_imbalance_60s",
            "aggressor_aligned_30s",
            "aggressor_aligned_10s",
        ]:
            feats[k] = 0.0

    return feats


# ── Composite score baseline (replicates alert_manager._composite_score) ─────


def composite_score(
    level_name: str,
    entry_count: int,
    alert_mins: int,
    tick_rate: float | None,
    session_move_pts: float | None,
    direction: str | None = None,
) -> int:
    """Replicate production _composite_score without streak (unavailable in backtest)."""
    s = 0

    if level_name == "IBL":
        s += 3
    elif level_name == "IBH":
        s -= 1
    elif level_name == "FIB_EXT_LO_1.272":
        s += 2
    elif level_name == "FIB_EXT_HI_1.272":
        s += 1

    if direction is not None:
        combo = (level_name, direction)
        if combo in (
            ("FIB_EXT_HI_1.272", "up"),
            ("FIB_EXT_LO_1.272", "down"),
            ("IBL", "down"),
        ):
            s += 1
        elif combo in (("IBH", "up"),):
            s -= 1

    mins = alert_mins + 9 * 60 + 30  # convert back to absolute minutes
    if (13 * 60) <= mins < (15 * 60):
        s += 2
    elif (10 * 60 + 30) <= mins < (11 * 60 + 30):
        s -= 3
    elif (11 * 60 + 30) <= mins < (13 * 60):
        s -= 1
    else:
        s += 1

    if tick_rate is not None:
        if tick_rate >= 2000:
            s += 2
        elif tick_rate >= 1750:
            s += 1
        elif tick_rate < 1000:
            s -= 2

    if entry_count == 1:
        s -= 4
    elif entry_count == 3:
        s += 2
    elif entry_count == 4:
        s += 1
    elif entry_count == 5:
        s -= 2
    elif entry_count >= 6:
        s -= 4

    if session_move_pts is not None:
        if -50 < session_move_pts <= 0:
            s += 2
        elif session_move_pts > 50:
            s -= 1

    return s


# ── Hybrid exit: +8 target / -20 stop / 1-min timeout ────────────────────────
# Entry at exactly line_price. Clock starts when price first crosses the line.

EXIT_MINUTES = 1
EXIT_NS = np.int64(EXIT_MINUTES * 60 * 1_000_000_000)
PROFIT_TARGET = 8.0  # take profit at +8 from line
STOP_LOSS = 20.0  # stop out at -20 from line


def evaluate_time_exit_np(
    alert_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
) -> tuple[str, float]:
    """Evaluate hybrid exit: +8 target, -20 stop, or 1-minute timeout.

    Assumes entry at exactly line_price.

    1. After alert fires, scan forward for the first tick where price crosses
       to or past the line — that's the fill moment (you buy/sell at line_price).
    2. Start the 1-minute clock from that tick.
    3. Scan for +8 target or -20 stop from line_price (whichever first).
    4. If neither hit after 1 minute, exit at current price.

    outcome: 'correct' if P&L > 0, 'incorrect' if P&L <= 0, 'inconclusive' if no data.
    """
    n = len(prices)
    alert_price = float(prices[alert_idx])

    # Step 1: Find the fill — first tick where price crosses to/past the line
    fill_idx = None
    if alert_price >= line_price:
        # Price is above line — wait for price to come down to or below line
        for i in range(alert_idx, n):
            if float(prices[i]) <= line_price:
                fill_idx = i
                break
            if ts_ns[i] - ts_ns[alert_idx] > np.int64(30_000_000_000):
                break
    else:
        # Price is below line — wait for price to come up to or above line
        for i in range(alert_idx, n):
            if float(prices[i]) >= line_price:
                fill_idx = i
                break
            if ts_ns[i] - ts_ns[alert_idx] > np.int64(30_000_000_000):
                break

    if fill_idx is None:
        return "inconclusive", 0.0

    # Step 2: 1-minute clock starts from fill
    fill_ns = ts_ns[fill_idx]
    exit_ns = fill_ns + EXIT_NS

    time_exit_idx = int(np.searchsorted(ts_ns, exit_ns, side="right")) - 1
    if time_exit_idx <= fill_idx:
        return "inconclusive", 0.0

    # Step 3: Scan from fill+1 for target/stop (P&L from line_price)
    window_prices = prices[fill_idx + 1 : time_exit_idx + 1]
    if len(window_prices) == 0:
        return "inconclusive", 0.0

    if direction == "up":
        pnls = window_prices - line_price
    else:
        pnls = line_price - window_prices

    for i in range(len(pnls)):
        pnl_i = float(pnls[i])
        if pnl_i >= PROFIT_TARGET:
            return "correct", pnl_i
        if pnl_i <= -STOP_LOSS:
            return "incorrect", pnl_i

    # Step 4: Neither hit — exit at 1-minute mark from fill
    pnl = float(pnls[-1])
    outcome = "correct" if pnl > 0 else "incorrect"
    return outcome, pnl


# ── Dataset assembly ─────────────────────────────────────────────────────────


def build_dataset(
    day_caches: dict[datetime.date, MLDayCache],
    days: list[datetime.date],
    vix_data: dict[datetime.date, float] | None = None,
) -> pd.DataFrame:
    """Generate features for every alert across all days.

    Each alert produces TWO rows (one BUY, one SELL) so the model can learn
    whether to take continuation or bounce. Rows sharing the same alert are
    linked by (date, level, line_price, _alert_ts) for grouping at inference.
    """
    if vix_data is None:
        vix_data = {}

    # Pre-compute cross-day context
    day_summaries = compute_day_summaries(day_caches, days)
    sorted_dates = sorted(day_summaries.keys())

    rows: list[dict] = []

    for day_i, date in enumerate(days):
        dc = day_caches.get(date)
        if dc is None:
            continue

        # Get cross-day features (same for all alerts on this day)
        cross_day = get_cross_day_features(date, day_summaries, sorted_dates, vix_data)

        prices = dc.post_ib_prices
        n = len(prices)

        all_levels = [
            ("IBH", np.full(n, dc.ibh), EXIT_THRESHOLD, False),
            ("IBL", np.full(n, dc.ibl), EXIT_THRESHOLD, False),
            ("VWAP", dc.post_ib_vwaps, EXIT_THRESHOLD, False),
            ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), EXIT_THRESHOLD, False),
            ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), EXIT_THRESHOLD, False),
        ]

        # Collect all alerts for this day with their features
        day_alerts: list[tuple[int, dict]] = []  # (full_idx, row_dict)

        for level_name, level_arr, et, use_current in all_levels:
            entries = _run_zone_numpy(
                prices, level_arr, ALERT_THRESHOLD, et, use_current
            )

            for idx, entry_count, ref_price in entries:
                price = prices[idx]
                full_idx = dc.post_ib_start_idx + idx

                # Evaluate BOTH directions for each alert
                for direction in ("up", "down"):
                    outcome, pnl = evaluate_time_exit_np(
                        full_idx, ref_price, direction, dc.full_ts_ns, dc.full_prices
                    )
                    if outcome not in ("correct", "incorrect"):
                        continue

                    day_alerts.append(
                        (
                            full_idx,
                            {
                                "date": date,
                                "level": level_name,
                                "line_price": ref_price,
                                "entry_price": price,
                                "direction": direction,
                                "entry_count": entry_count,
                                "outcome": 1 if outcome == "correct" else 0,
                                "pnl": pnl,
                                "_full_idx": full_idx,
                                "_alert_id": f"{date}_{level_name}_{full_idx}",
                            },
                        )
                    )

        # Sort by timestamp for fatigue features
        day_alerts.sort(key=lambda x: x[0])

        alerts_fired = 0
        last_alert_ns: int | None = None
        seen_alert_ids: set[str] = set()

        for full_idx, row in day_alerts:
            feats = extract_features(
                dc,
                full_idx,
                row["entry_count"],
                row["line_price"],
                row["entry_price"],
                row["direction"],
                alerts_fired,
                last_alert_ns,
            )
            if feats is None:
                continue

            # Direction feature
            feats["direction_is_up"] = 1 if row["direction"] == "up" else 0

            # Level one-hots
            feats["is_ibl"] = 1 if row["level"] == "IBL" else 0
            feats["is_vwap"] = 1 if row["level"] == "VWAP" else 0
            feats["is_fib"] = (
                1 if row["level"] in ("FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272") else 0
            )

            # Cross-day features (10)
            feats.update(cross_day)

            # Composite score for baseline comparison
            feats["composite_score"] = composite_score(
                row["level"],
                row["entry_count"],
                int(feats["time_of_day_mins"]),
                feats["tick_rate"],
                feats["session_move_pts"],
                row["direction"],
            )

            row.update(feats)
            del row["_full_idx"]
            rows.append(row)

            # Track fatigue per unique alert (not per direction row)
            aid = row["_alert_id"]
            if aid not in seen_alert_ids:
                seen_alert_ids.add(aid)
                alerts_fired += 1
                last_alert_ns = int(dc.full_ts_ns[full_idx])

        if (day_i + 1) % 50 == 0:
            print(f"  {day_i + 1}/{len(days)} days processed...", flush=True)

    return pd.DataFrame(rows)


# ── Walk-forward cross-validation ────────────────────────────────────────────


def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward CV. Returns (actuals, predicted_probas) for all OOS alerts."""
    dates = sorted(df["date"].unique())
    all_actuals = []
    all_probas = []

    fold = 0
    train_end = INITIAL_TRAIN_DAYS

    while train_end < len(dates):
        test_end = min(train_end + TEST_WINDOW_DAYS, len(dates))
        train_dates = set(dates[:train_end])
        test_dates = set(dates[train_end:test_end])

        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"].isin(test_dates)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "outcome"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "outcome"].values

        if len(X_test) == 0 or len(np.unique(y_train)) < 2:
            train_end = test_end
            continue

        model = model_factory()
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)[:, 1]  # P(correct)

        all_actuals.append(y_test)
        all_probas.append(probas)
        fold += 1
        train_end = test_end

    return np.concatenate(all_actuals), np.concatenate(all_probas)


def best_direction_results(
    df_test: pd.DataFrame,
    probas: np.ndarray,
    n_days: int,
) -> pd.DataFrame:
    """For each alert, pick the direction with higher P(correct).

    Returns a DataFrame with one row per alert (the chosen direction),
    with columns: outcome, pnl, proba, level, date.
    """
    df_t = df_test.copy()
    df_t["proba"] = probas

    # Group by alert_id and pick direction with highest proba
    best_rows = []
    for alert_id, group in df_t.groupby("_alert_id"):
        best_idx = group["proba"].idxmax()
        best = group.loc[best_idx]
        best_rows.append(
            {
                "date": best["date"],
                "level": best["level"],
                "direction": best["direction"],
                "outcome": int(best["outcome"]),
                "pnl": best["pnl"],
                "proba": best["proba"],
            }
        )
    return pd.DataFrame(best_rows)


# ── Results printing ─────────────────────────────────────────────────────────


def print_threshold_sweep(
    actuals: np.ndarray,
    probas: np.ndarray,
    n_days: int,
    label: str,
    pnls: np.ndarray | None = None,
) -> None:
    """Print win rate / alerts per day at various probability thresholds."""
    print(f"\n  {label} — Probability threshold sweep:")
    print(
        f"  {'P(correct)≥':>12}  {'W':>5}  {'L':>5}  {'Decided':>8}  {'Win%':>6}  {'Avg P&L':>8}  {'/day':>5}"
    )
    print(f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*5}")

    best_row = None
    for thr in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.72,
        0.74,
        0.76,
        0.78,
        0.80,
        0.82,
        0.85,
        0.88,
        0.90,
    ]:
        mask = probas >= thr
        w = int((actuals[mask] == 1).sum())
        l = int((actuals[mask] == 0).sum())
        t = w + l
        if t == 0:
            continue
        wr = w / t
        avg_pnl = float(pnls[mask].mean()) if pnls is not None else 0.0
        per_day = t / n_days
        marker = ""
        if wr >= 0.80 and per_day >= 4.0:
            marker = "  ★ TARGET"
        elif wr >= 0.80 and per_day >= 2.0:
            marker = "  ← 80%+"
        print(
            f"  {thr:>12.2f}  {w:>5}  {l:>5}  {t:>8}  {wr:>5.1%}  {avg_pnl:>+8.2f}  {per_day:>5.1f}{marker}"
        )
        if best_row is None and wr >= 0.80:
            best_row = (thr, w, l, t, wr, avg_pnl, per_day)

    if best_row:
        thr, w, l, t, wr, avg_pnl, per_day = best_row
        print(
            f"\n  → Best 80%+ threshold: P≥{thr:.2f} → {wr:.1%} WR, {per_day:.1f}/day, Avg P&L {avg_pnl:+.2f}"
        )


def print_composite_baseline(df: pd.DataFrame, n_days: int) -> None:
    """Show composite score baseline for comparison."""
    print(f"\n{'─' * 75}")
    print(
        f"  COMPOSITE SCORE BASELINE (+{PROFIT_TARGET:.0f}/-{STOP_LOSS:.0f}/{EXIT_MINUTES}min hybrid, no streak)"
    )
    print(f"{'─' * 75}")
    print(
        f"  {'Score≥':>8}  {'W':>5}  {'L':>5}  {'Decided':>8}  {'Win%':>6}  {'Avg P&L':>8}  {'/day':>5}"
    )
    print(f"  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*5}")

    for min_score in range(-2, 7):
        mask = df["composite_score"] >= min_score
        sub = df[mask]
        w = int(sub["outcome"].sum())
        l = len(sub) - w
        t = w + l
        if t == 0:
            continue
        wr = w / t
        avg_pnl = sub["pnl"].mean()
        per_day = t / n_days
        marker = ""
        if min_score == 4:
            marker = "  ← production"
        if wr >= 0.80 and per_day >= 4.0:
            marker = "  ★ TARGET"
        print(
            f"  {min_score:>8}  {w:>5}  {l:>5}  {t:>8}  {wr:>5.1%}  {avg_pnl:>+8.2f}  {per_day:>5.1f}{marker}"
        )


def print_per_level(
    actuals: np.ndarray,
    probas: np.ndarray,
    levels: np.ndarray,
    threshold: float,
    n_days: int,
    label: str,
) -> None:
    """Per-level breakdown at a given threshold."""
    mask = probas >= threshold
    print(f"\n  {label} — Per-level at P≥{threshold:.2f}:")
    print(
        f"  {'Level':<25}  {'W':>5}  {'L':>5}  {'Decided':>8}  {'Win%':>6}  {'/day':>5}"
    )
    print(f"  {'-'*25}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*5}")

    for lvl in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        lvl_mask = mask & (levels == lvl)
        w = int((actuals[lvl_mask] == 1).sum())
        l = int((actuals[lvl_mask] == 0).sum())
        t = w + l
        if t == 0:
            continue
        wr = w / t
        per_day = t / n_days
        print(f"  {lvl:<25}  {w:>5}  {l:>5}  {t:>8}  {wr:>5.1%}  {per_day:>5.1f}")


def print_feature_importances(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory,
) -> None:
    """Train on all data and print feature importances."""
    X = df[feature_cols].values
    y = df["outcome"].values
    model = model_factory()
    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
    elif hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            imp = pd.Series(np.abs(clf.coef_[0]), index=feature_cols)
        else:
            return
    else:
        return

    imp = imp.sort_values(ascending=False)
    print(f"\n  Top 15 feature importances:")
    for i, (feat, val) in enumerate(imp.items()):
        if i >= 15:
            break
        bar = "█" * max(1, int(val / imp.max() * 25))
        print(f"    {feat:<28} {val / imp.sum():>5.1%}  {bar}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    days = load_cached_days()
    print(f"{'═' * 75}")
    print(f"  ML ALERT MODEL  |  {days[0]} → {days[-1]}  ({len(days)} days)")
    print(
        f"  Outcome: +{PROFIT_TARGET:.0f}pt target / -{STOP_LOSS:.0f}pt stop / {EXIT_MINUTES}-min timeout"
    )
    print(f"  Both BUY and SELL evaluated per alert (model picks best direction)")
    print(
        f"  Walk-forward: train {INITIAL_TRAIN_DAYS} days, test {TEST_WINDOW_DAYS}-day windows"
    )
    print(f"{'═' * 75}")

    # ── Load and preprocess all days ─────────────────────────────────────────
    print("\n  Loading data...", flush=True)
    day_caches: dict[datetime.date, MLDayCache] = {}
    for i, date in enumerate(days):
        try:
            df = load_day(date)
            dc = preprocess_day_ml(df, date)
            if dc is not None:
                day_caches[date] = dc
        except Exception as e:
            print(f"  Error loading {date}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(days)} days loaded...", flush=True)
    print(f"  {len(day_caches)} days loaded. ({time.time() - t0:.1f}s)")

    # ── Fetch VIX data ────────────────────────────────────────────────────────
    print("\n  Fetching VIX data...", flush=True)
    vix_data = fetch_vix_data(days[0], days[-1])
    print(f"  VIX data: {len(vix_data)} days available.")

    # ── Build feature dataset ────────────────────────────────────────────────
    print("\n  Extracting features...", flush=True)
    t1 = time.time()
    df = build_dataset(day_caches, days, vix_data=vix_data)
    n_days = len(day_caches)

    n_unique_alerts = df["_alert_id"].nunique()
    w = int(df["outcome"].sum())
    l = len(df) - w
    avg_pnl = df["pnl"].mean()
    print(f"  Dataset: {len(df)} rows ({n_unique_alerts} unique alerts × 2 directions)")
    print(f"  {w} correct, {l} incorrect across all rows")
    print(f"  Unique alerts/day: {n_unique_alerts / n_days:.1f}")
    print(f"  Feature extraction: {time.time() - t1:.1f}s")

    # Feature columns (exclude metadata and composite_score)
    meta_cols = {
        "date",
        "level",
        "line_price",
        "entry_price",
        "direction",
        "entry_count",
        "outcome",
        "pnl",
        "composite_score",
        "_alert_id",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  Features: {len(feature_cols)}")

    # ── Composite score baseline ─────────────────────────────────────────────
    print_composite_baseline(df, n_days)

    # ── Model comparison ─────────────────────────────────────────────────────
    models = [
        (
            "GradientBoosting",
            lambda: GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
        (
            "RandomForest",
            lambda: RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "LogisticRegression",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=0.3,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]

    print(f"\n{'═' * 75}")
    print("  WALK-FORWARD MODEL COMPARISON")
    print(f"{'═' * 75}")

    dates_sorted = sorted(df["date"].unique())
    test_dates = set(dates_sorted[INITIAL_TRAIN_DAYS:])
    test_mask = df["date"].isin(test_dates)
    df_test = df.loc[test_mask].reset_index(drop=True)
    test_levels = df_test["level"].values
    test_pnls = df_test["pnl"].values
    test_n_days = len(test_dates)

    for name, factory in models:
        print(f"\n{'─' * 75}")
        print(f"  {name}")
        print(f"{'─' * 75}")

        t2 = time.time()
        actuals, probas = walk_forward_evaluate(df, feature_cols, factory, n_days)
        elapsed = time.time() - t2

        # ROC-AUC (per-row, both directions)
        if len(np.unique(actuals)) >= 2:
            auc = roc_auc_score(actuals, probas)
            print(f"  Walk-forward ROC-AUC: {auc:.3f}  ({elapsed:.1f}s)")
        else:
            print(
                f"  Walk-forward ROC-AUC: N/A (single class in test)  ({elapsed:.1f}s)"
            )

        # ── Best-direction analysis ───────────────────────────────────────────
        # For each alert, pick the direction with higher P(correct)
        best_df = best_direction_results(df_test, probas, test_n_days)

        bd_actuals = best_df["outcome"].values
        bd_probas = best_df["proba"].values
        bd_pnls = best_df["pnl"].values
        bd_levels = best_df["level"].values

        print(f"\n  BEST-DIRECTION (model picks BUY or SELL per alert):")
        print_threshold_sweep(bd_actuals, bd_probas, test_n_days, name, pnls=bd_pnls)

        # Find best 80%+ threshold for per-level breakdown
        best_thr = None
        for thr in [
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.72,
            0.74,
            0.76,
            0.78,
            0.80,
            0.82,
            0.85,
            0.88,
            0.90,
        ]:
            m = bd_probas >= thr
            t = int(m.sum())
            if t == 0:
                continue
            wr = (bd_actuals[m] == 1).sum() / t
            if wr >= 0.80:
                best_thr = thr
                break

        if best_thr is not None:
            print_per_level(
                bd_actuals, bd_probas, bd_levels, best_thr, test_n_days, name
            )

        # Direction breakdown at best threshold
        if best_thr is not None:
            m = bd_probas >= best_thr
            chosen = best_df[m]
            n_buy = int((chosen["direction"] == "up").sum())
            n_sell = int((chosen["direction"] == "down").sum())
            print(
                f"\n  Direction split at P≥{best_thr:.2f}: {n_buy} BUY, {n_sell} SELL"
            )

        print_feature_importances(df, feature_cols, factory)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 75}")
    print(f"  Total runtime: {time.time() - t0:.1f}s")
    print(f"{'═' * 75}")


if __name__ == "__main__":
    main()
