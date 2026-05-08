"""Label computation for level-touch events.

For each event, compute 8 binary win/loss labels — one per
(direction in {bounce, breakthrough}) × ((TP, SL) in {(8,25),(8,20),(10,25),(10,20)}).

Resolution window = min(event_ts + 15min, 4:00 PM ET). Win = TP touched first
within window. Loss = SL touched first OR neither within window.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
TP_SL_VARIANTS = [(8, 25), (8, 20), (10, 25), (10, 20)]
RESOLUTION_MINUTES = 15
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0


def label_events(events: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    """Expand each event into 8 labeled rows.

    Returns DataFrame with: event_ts, level_name, level_price, event_price,
    approach_direction, direction (bounce|breakthrough), tp, sl, label (0/1),
    time_to_resolution_sec.
    """
    if events.empty:
        return _empty_labels()

    prices = ticks["price"].to_numpy()
    # Keep as DatetimeIndex to preserve timezone info for tz-aware searchsorted comparisons.
    times_idx = ticks.index

    out_rows = []
    for _, ev in events.iterrows():
        event_ts = ev["event_ts"]
        entry_price = ev["event_price"]
        approach = int(ev["approach_direction"])

        window_end = _resolution_window_end(event_ts)
        # Slice ticks strictly after event_ts up to window_end inclusive.
        start_idx = int(times_idx.searchsorted(event_ts, side="right"))
        end_idx = int(times_idx.searchsorted(window_end, side="right"))
        slice_prices = prices[start_idx:end_idx]
        slice_times = times_idx[start_idx:end_idx]

        for direction in ("bounce", "breakthrough"):
            for tp, sl in TP_SL_VARIANTS:
                label, ttr = _resolve(direction, approach, entry_price, tp, sl, slice_prices, slice_times, event_ts)
                out_rows.append(
                    {
                        "event_ts": event_ts,
                        "level_name": ev["level_name"],
                        "level_price": ev["level_price"],
                        "event_price": entry_price,
                        "approach_direction": approach,
                        "direction": direction,
                        "tp": tp,
                        "sl": sl,
                        "label": label,
                        "time_to_resolution_sec": ttr,
                    }
                )
    return pd.DataFrame(out_rows)


def _resolution_window_end(event_ts: pd.Timestamp) -> pd.Timestamp:
    cap_15min = event_ts + pd.Timedelta(minutes=RESOLUTION_MINUTES)
    et_dt = event_ts.tz_convert(ET)
    close_dt = et_dt.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    close_utc = close_dt.tz_convert("UTC")
    return min(cap_15min, close_utc)


def _resolve(direction: str, approach: int, entry: float, tp: float, sl: float,
             prices: np.ndarray, times: pd.DatetimeIndex, event_ts: pd.Timestamp) -> tuple[int, float]:
    """Return (label, time_to_resolution_sec). label=1 if TP first, 0 otherwise."""
    if approach == 0 or len(prices) == 0:
        return 0, float("nan")
    if direction == "bounce":
        # Target is AGAINST approach direction. Stop is WITH approach direction.
        tp_price = entry - approach * tp
        sl_price = entry + approach * sl
    else:
        # breakthrough: target WITH approach. Stop AGAINST approach.
        tp_price = entry + approach * tp
        sl_price = entry - approach * sl

    for k, p in enumerate(prices):
        tp_hit = (approach == 1 and ((direction == "bounce" and p <= tp_price) or (direction == "breakthrough" and p >= tp_price))) or \
                 (approach == -1 and ((direction == "bounce" and p >= tp_price) or (direction == "breakthrough" and p <= tp_price)))
        sl_hit = (approach == 1 and ((direction == "bounce" and p >= sl_price) or (direction == "breakthrough" and p <= sl_price))) or \
                 (approach == -1 and ((direction == "bounce" and p <= sl_price) or (direction == "breakthrough" and p >= sl_price)))
        if tp_hit and not sl_hit:
            ttr = (pd.Timestamp(times[k]) - event_ts).total_seconds()
            return 1, ttr
        if sl_hit:
            ttr = (pd.Timestamp(times[k]) - event_ts).total_seconds()
            return 0, ttr
    return 0, float("nan")  # timeout = loss


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["event_ts", "level_name", "level_price", "event_price",
                 "approach_direction", "direction", "tp", "sl", "label",
                 "time_to_resolution_sec"]
    )
