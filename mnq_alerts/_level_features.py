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
