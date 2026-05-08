"""Feature computation for level-touch events.

Five families: kinematics, aggressor, volume, level context, vol/time.
All features computed strictly from ticks with ts <= event_ts (no leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS_SEC = {"5s": 5, "30s": 30, "5min": 300, "15min": 900}


def _slice_window(ticks: pd.DataFrame, event_ts: pd.Timestamp, seconds: int) -> pd.DataFrame:
    """Return ticks in [event_ts - seconds, event_ts]. Inclusive both ends."""
    lo = event_ts - pd.Timedelta(seconds=seconds)
    return ticks.loc[(ticks.index >= lo) & (ticks.index <= event_ts)]


def compute_kinematics(ticks: pd.DataFrame, event_ts: pd.Timestamp) -> dict:
    """Family 1 — approach kinematics."""
    feats: dict[str, float] = {}
    for win, sec in WINDOWS_SEC.items():
        if win == "15min":
            continue
        sub = _slice_window(ticks, event_ts, sec)
        if len(sub) < 2:
            feats[f"velocity_{win}"] = 0.0
            continue
        dp = float(sub["price"].iloc[-1] - sub["price"].iloc[0])
        dt = (sub.index[-1] - sub.index[0]).total_seconds()
        feats[f"velocity_{win}"] = dp / dt if dt > 0 else 0.0

    # Acceleration: difference of velocities at 30s vs 5s windows.
    feats["acceleration_30s"] = feats.get("velocity_5s", 0.0) - feats.get("velocity_30s", 0.0)

    # Path efficiency over 5 min.
    sub5 = _slice_window(ticks, event_ts, WINDOWS_SEC["5min"])
    if len(sub5) >= 2:
        prices = sub5["price"].to_numpy()
        displacement = abs(prices[-1] - prices[0])
        total_move = float(np.abs(np.diff(prices)).sum())
        feats["path_efficiency_5min"] = displacement / total_move if total_move > 0 else 0.0
    else:
        feats["path_efficiency_5min"] = 0.0
    return feats


def _classify_aggressor(prices: np.ndarray) -> np.ndarray:
    """Tick-rule classification. Returns array of +1 (buy), -1 (sell), 0 (neutral).

    Zero-tick inherits prior non-zero side. First tick is neutral.
    """
    n = len(prices)
    side = np.zeros(n, dtype=int)
    last = 0
    for i in range(1, n):
        if prices[i] > prices[i - 1]:
            last = 1
        elif prices[i] < prices[i - 1]:
            last = -1
        side[i] = last
    return side


def compute_aggressor(ticks: pd.DataFrame, event_ts: pd.Timestamp) -> dict:
    """Family 2 — tick-rule aggressor balance and net dollar flow."""
    feats: dict[str, float] = {}
    for win, sec in WINDOWS_SEC.items():
        if win not in ("5s", "30s", "5min"):
            continue
        sub = _slice_window(ticks, event_ts, sec)
        if len(sub) < 2:
            feats[f"aggressor_balance_{win}"] = 0.0
            continue
        prices = sub["price"].to_numpy()
        sizes = sub["size"].to_numpy()
        side = _classify_aggressor(prices)
        classified = side != 0
        total = float(sizes[classified].sum())
        signed = float((sizes * side).sum())
        feats[f"aggressor_balance_{win}"] = signed / total if total > 0 else 0.0

    sub5 = _slice_window(ticks, event_ts, WINDOWS_SEC["5min"])
    if len(sub5) >= 2:
        prices = sub5["price"].to_numpy()
        sizes = sub5["size"].to_numpy()
        side = _classify_aggressor(prices)
        feats["net_dollar_flow_5min"] = float((prices * sizes * side).sum())
    else:
        feats["net_dollar_flow_5min"] = 0.0
    return feats


def compute_volume_profile(ticks: pd.DataFrame, event_ts: pd.Timestamp) -> dict:
    """Family 3 — volume / size profile."""
    feats: dict[str, float] = {}
    for win in ("5s", "30s", "5min"):
        sub = _slice_window(ticks, event_ts, WINDOWS_SEC[win])
        feats[f"volume_{win}"] = float(sub["size"].sum()) if not sub.empty else 0.0

    sub30 = _slice_window(ticks, event_ts, WINDOWS_SEC["30s"])
    if not sub30.empty:
        sec = max(1.0, (sub30.index[-1] - sub30.index[0]).total_seconds())
        feats["trade_rate_30s"] = len(sub30) / sec
        feats["max_print_size_30s"] = float(sub30["size"].max())
        sizes = sub30["size"].to_numpy().astype(float)
        total = float(sizes.sum())
        feats["volume_concentration_30s"] = float((sizes ** 2).sum() / (total ** 2)) if total > 0 else 0.0
    else:
        feats["trade_rate_30s"] = 0.0
        feats["max_print_size_30s"] = 0.0
        feats["volume_concentration_30s"] = 0.0
    return feats


def compute_level_context(
    prior_touches: list[dict],
    all_levels: dict[str, float],
    event_ts: pd.Timestamp,
    event_price: float,
    level_name: str,
) -> dict:
    """Family 4 — level context.

    `prior_touches` is the list of all touches at THIS level earlier in the day,
    each with keys: event_ts, resolution_ts, outcome (string).

    `all_levels` maps level_name -> price for ALL levels in the session
    (including VWAP, used only as a distance reference).
    """
    feats: dict = {}

    # touches_today: count touches at this level with event_ts < current event_ts.
    earlier = [t for t in prior_touches if t["event_ts"] < event_ts]
    feats["touches_today"] = len(earlier)

    # prior_touch_outcome: outcome of the most-recent touch that is RESOLVED.
    resolved = [t for t in earlier if t["resolution_ts"] <= event_ts]
    if resolved:
        last_resolved = max(resolved, key=lambda t: t["event_ts"])
        feats["prior_touch_outcome"] = last_resolved["outcome"]
        feats["seconds_since_last_touch"] = (event_ts - last_resolved["event_ts"]).total_seconds()
    else:
        feats["prior_touch_outcome"] = "none"
        feats["seconds_since_last_touch"] = -1.0

    # distance_to_vwap (signed: event_price - vwap_price).
    feats["distance_to_vwap"] = float(event_price - all_levels.get("VWAP", event_price))

    # distance_to_nearest_other_level (excluding self AND VWAP).
    others = [p for n, p in all_levels.items() if n != level_name and n != "VWAP"]
    if others:
        feats["distance_to_nearest_other_level"] = min(abs(event_price - p) for p in others)
    else:
        feats["distance_to_nearest_other_level"] = 0.0

    # is_post_IB: events only fire post-IB lock by Task 1, so always 1. Kept for forward-compat.
    feats["is_post_IB"] = 1
    return feats


def compute_vol_time(ticks: pd.DataFrame, event_ts: pd.Timestamp) -> dict:
    """Family 5 — volatility & time-of-day."""
    feats: dict = {}
    sub5 = _slice_window(ticks, event_ts, WINDOWS_SEC["5min"])
    sub30 = _slice_window(ticks, event_ts, 1800)
    if len(sub5) >= 2:
        ret = sub5["price"].pct_change().dropna().to_numpy()
        feats["realized_vol_5min"] = float(np.std(ret)) if len(ret) > 0 else 0.0
    else:
        feats["realized_vol_5min"] = 0.0
    if len(sub30) >= 2:
        ret = sub30["price"].pct_change().dropna().to_numpy()
        feats["realized_vol_30min"] = float(np.std(ret)) if len(ret) > 0 else 0.0
        feats["range_30min"] = float(sub30["price"].max() - sub30["price"].min())
    else:
        feats["realized_vol_30min"] = 0.0
        feats["range_30min"] = 0.0

    et = event_ts.tz_convert("America/New_York")
    close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    feats["seconds_to_market_close"] = max(0.0, (close - et).total_seconds())
    open_dt = et.replace(hour=9, minute=30, second=0, microsecond=0)
    feats["seconds_into_session"] = (et - open_dt).total_seconds()

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for i, d in enumerate(days):
        feats[f"day_of_week_{d}"] = 1 if et.weekday() == i else 0
    return feats


def compute_all_features(
    *,
    ticks: pd.DataFrame,
    event_ts: pd.Timestamp,
    event_price: float,
    level_name: str,
    level_price: float,
    approach_direction: int,
    prior_touches: list[dict],
    all_levels: dict[str, float],
) -> dict:
    """Compute all feature families for one event. Output is a flat dict.

    ALL features are derivable strictly from `ticks.loc[ticks.index <= event_ts]`
    and `prior_touches` filtered to resolution_ts <= event_ts.
    """
    # Defensive: ensure ticks slice doesn't include future.
    ticks_safe = ticks.loc[ticks.index <= event_ts]
    feats: dict = {}
    feats.update(compute_kinematics(ticks_safe, event_ts))
    feats.update(compute_aggressor(ticks_safe, event_ts))
    feats.update(compute_volume_profile(ticks_safe, event_ts))
    feats.update(compute_level_context(prior_touches, all_levels, event_ts, event_price, level_name))
    feats.update(compute_vol_time(ticks_safe, event_ts))
    return feats
