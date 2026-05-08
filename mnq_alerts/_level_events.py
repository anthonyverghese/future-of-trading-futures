"""Level-touch event extraction with per-level arming state machine.

An event fires when price enters within 1.0 points of a level AND the level is
armed. After an event, the level disarms and re-arms only when price moves more
than 3.0 points from the level. All levels arm at 10:31 ET (matches existing
config IB_END_HOUR=10, IB_END_MIN=31).
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
ENTRY_THRESHOLD = 1.0
EXIT_THRESHOLD = 3.0
APPROACH_LOOKBACK_SEC = 60
IB_LOCK_HOUR = 10
IB_LOCK_MINUTE = 31


def extract_events(ticks: pd.DataFrame, levels: dict[str, float]) -> pd.DataFrame:
    """Walk ticks chronologically, emit one row per (level, touch event).

    Parameters
    ----------
    ticks : DataFrame with DatetimeIndex (UTC), columns ["price", "size"].
    levels : mapping level_name -> level_price for this session.

    Returns
    -------
    DataFrame with columns: event_ts, level_name, level_price, event_price,
    approach_direction (sign of event_price - price_60s_before; 0 if no
    earlier tick available).
    """
    if ticks.empty or not levels:
        return _empty_events()

    prices = ticks["price"].to_numpy()
    times = ticks.index.to_numpy()
    times_pd = ticks.index

    # Per-level state. ever_armed flips True at the first eligible tick (>= 10:31 ET);
    # armed disarms after each event and re-arms only when |price - level| > 3.0.
    ever_armed = {name: False for name in levels}
    armed = {name: False for name in levels}

    out_rows = []
    for i in range(len(ticks)):
        t = times_pd[i]
        t_et = t.tz_convert(ET) if t.tzinfo else t.replace(tzinfo=ET)
        if t_et.hour < IB_LOCK_HOUR or (t_et.hour == IB_LOCK_HOUR and t_et.minute < IB_LOCK_MINUTE):
            continue
        # Initial arming: each level arms exactly once, at its first eligible tick.
        for name in levels:
            if not ever_armed[name]:
                armed[name] = True
                ever_armed[name] = True
        p = prices[i]
        for name, level_price in levels.items():
            dist = abs(p - level_price)
            if armed[name] and dist <= ENTRY_THRESHOLD:
                approach = _compute_approach(times, prices, i, p)
                out_rows.append(
                    {
                        "event_ts": t,
                        "level_name": name,
                        "level_price": level_price,
                        "event_price": p,
                        "approach_direction": approach,
                    }
                )
                armed[name] = False
            elif not armed[name] and dist > EXIT_THRESHOLD:
                armed[name] = True
    return pd.DataFrame(out_rows) if out_rows else _empty_events()


def _compute_approach(times: np.ndarray, prices: np.ndarray, i: int, p_now: float) -> int:
    """Sign of (p_now - p_60s_before). 0 if no earlier tick exists."""
    target = times[i] - np.timedelta64(APPROACH_LOOKBACK_SEC, "s")
    j = np.searchsorted(times, target, side="left")
    if j == 0 and times[0] > target:
        return 0
    j = max(0, j - 1) if times[j] > target else j
    diff = p_now - prices[j]
    if diff > 0:
        return 1
    if diff < 0:
        return -1
    return 0


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["event_ts", "level_name", "level_price", "event_price", "approach_direction"]
    )
