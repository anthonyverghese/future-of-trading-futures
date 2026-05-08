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


from _level_features import compute_all_features


def build_day_with_features(ticks: pd.DataFrame) -> pd.DataFrame:
    """Build labeled+featured rows for one day. Each row = (event, direction, tp, sl)."""
    levels = compute_session_levels(ticks)
    if not levels:
        return _empty_dataset()
    events = extract_events(ticks, levels)
    if events.empty:
        return _empty_dataset()
    labels = label_events(events, ticks)

    # Build prior_touches list (resolved touches at same level earlier in day).
    # Resolution timestamp = event_ts + time_to_resolution_sec (NaN treated as resolved at event_ts + 15min).
    label_index = labels.assign(
        resolution_ts=lambda d: d["event_ts"] + pd.to_timedelta(
            d["time_to_resolution_sec"].fillna(15 * 60), unit="s"
        ),
        outcome=lambda d: d.apply(_outcome_label, axis=1),
    )

    feature_rows = []
    for _, row in labels.iterrows():
        # Find prior touches at same level whose event_ts < this event_ts.
        same_level = label_index[
            (label_index["level_name"] == row["level_name"]) &
            (label_index["event_ts"] < row["event_ts"]) &
            (label_index["direction"] == "bounce") &  # one outcome per touch
            (label_index["tp"] == 8) & (label_index["sl"] == 25)
        ]
        prior_touches = [
            {"event_ts": t.event_ts, "resolution_ts": t.resolution_ts, "outcome": t.outcome}
            for t in same_level.itertuples()
        ]
        feats = compute_all_features(
            ticks=ticks, event_ts=row["event_ts"], event_price=float(row["event_price"]),
            level_name=row["level_name"], level_price=float(row["level_price"]),
            approach_direction=int(row["approach_direction"]),
            prior_touches=prior_touches,
            all_levels=levels,
        )
        feature_rows.append({**row.to_dict(), **feats})
    return pd.DataFrame(feature_rows)


def _outcome_label(row: pd.Series) -> str:
    """Map (direction, label) to canonical prior_touch_outcome string."""
    if row["label"] == 1:
        return f"{row['direction']}_held"
    return f"{row['direction']}_failed"


def build_full_history(parquet_dir: str, out_path: str) -> int:
    """Iterate every parquet day and concatenate labeled+featured rows.

    Returns row count written.
    """
    import os
    import glob
    files = sorted(glob.glob(os.path.join(parquet_dir, "MNQ_*.parquet")))
    frames = []
    for f in files:
        ticks = pd.read_parquet(f)
        if not ticks.index.tz:
            ticks.index = ticks.index.tz_localize("UTC")
        day = build_day_with_features(ticks)
        if not day.empty:
            frames.append(day)
    if not frames:
        return 0
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(out_path)
    return len(full)
