"""Per-day dataset builder: ticks -> events -> labels."""
from __future__ import annotations

import pandas as pd

from _level_events import extract_events
from _level_labels import label_events
from levels import calculate_fib_levels, calculate_interior_fibs

LEVELS_IN_SCOPE = (
    "IBH", "IBL",
    "FIB_0.236", "FIB_0.618", "FIB_0.764",
    "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272",
)


def compute_session_levels(ticks: pd.DataFrame) -> dict[str, float]:
    """Compute IB-locked level prices for one session from ticks.

    Reuses existing `levels.calculate_fib_levels` and `calculate_interior_fibs`.
    IBH/IBL are the high/low during 9:30 ET to IB lock (10:31 ET).
    """
    if ticks.empty:
        return {}
    et_idx = ticks.index.tz_convert("America/New_York")
    ib_mask = (
        ((et_idx.hour == 9) & (et_idx.minute >= 30))
        | ((et_idx.hour == 10) & (et_idx.minute < 31))
    )
    ib_window = ticks[ib_mask]
    if ib_window.empty:
        return {}
    ibh = float(ib_window["price"].max())
    ibl = float(ib_window["price"].min())
    levels: dict[str, float] = {"IBH": ibh, "IBL": ibl}
    levels.update(calculate_fib_levels(ibh, ibl))
    interior = calculate_interior_fibs(ibh, ibl)
    # Keep only levels in scope.
    for k, v in interior.items():
        if k in LEVELS_IN_SCOPE:
            levels[k] = v
    return {k: v for k, v in levels.items() if k in LEVELS_IN_SCOPE}


def build_day(ticks: pd.DataFrame) -> pd.DataFrame:
    """Build the labeled-event dataset for a single trading day."""
    levels = compute_session_levels(ticks)
    if not levels:
        return _empty_dataset()
    events = extract_events(ticks, levels)
    labels = label_events(events, ticks)
    return labels


def _empty_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["event_ts", "level_name", "level_price", "event_price",
                 "approach_direction", "direction", "tp", "sl", "label",
                 "time_to_resolution_sec"]
    )
