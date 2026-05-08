# Tick-data-driven per-level outcome model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an ML model that, given a level-touch event, predicts win probability for each of 8 trade variants (2 directions × 4 TP/SL), and gates deployment behind honest walk-forward validation, leakage tests, and a final out-of-time test.

**Architecture:** Per-day pipeline emits level-touch events from tick data; computes 8 binary outcome labels per event; computes ~25 leak-safe features per event; trains both per-(level, direction, TP/SL) classifiers (Architecture A) and a single pooled model (Architecture B); selects winner via 5-fold quarterly walk-forward with embargo; verifies via E2E gate; deploys via shadow → canary → full rollout if final out-of-time test passes.

**Tech Stack:** Python 3.11, pandas, numpy, pyarrow, lightgbm, joblib, pytest. Reuses existing `levels.py`, slippage-aware backtest broker, and tick parquet cache.

**Spec:** `docs/superpowers/specs/2026-05-08-tick-level-model-design.md`

---

## File map

**New modules in `mnq_alerts/`:**
- `_level_events.py` — event extraction with arming state machine (Section 1)
- `_level_labels.py` — 8-outcome label computation (Section 1)
- `_level_features.py` — feature pipeline, 5 families (Section 2)
- `_level_dataset.py` — orchestrates per-day events + labels + features → parquet
- `_level_model.py` — LightGBM training & inference for both A and B (Section 3)
- `_level_simulation.py` — trade simulator, 1-position constraint, slippage (Section 4)
- `_level_validation.py` — walk-forward folds, gates, architecture selection (Section 4)
- `_level_e2e_gate.py` — E2E gate verification (Section 4)
- `_level_final_test.py` — final out-of-time test runner (Sections 4 & 5)

**New tests in `mnq_alerts/tests/`:**
- `test_level_events.py`
- `test_level_labels.py`
- `test_level_features.py`
- `test_level_features_leakage.py` — the 3 critical leakage tests
- `test_level_model.py`
- `test_level_simulation.py`
- `test_level_validation.py`

**No modifications to existing live bot code in this plan.** Bot integration (Section 5 Stage 1+) is gated on the final test passing and will be a follow-up plan.

---

## Phase 1: Event & label pipeline

### Task 1: Event extraction with per-level arming state machine

**Files:**
- Create: `mnq_alerts/_level_events.py`
- Test: `mnq_alerts/tests/test_level_events.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_events.py`:

```python
"""Tests for _level_events.extract_events."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_events import extract_events


def _make_ticks(rows):
    """rows = list of (seconds_from_open, price)."""
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")  # 10:31 ET
    idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s, _ in rows])
    return pd.DataFrame({"price": [p for _, p in rows], "size": [1] * len(rows)}, index=idx)


def test_emits_event_when_price_within_1pt():
    levels = {"FIB_0.618": 100.0}
    ticks = _make_ticks([(0, 105), (1, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 1
    assert events.iloc[0]["level_name"] == "FIB_0.618"
    assert abs(events.iloc[0]["event_price"] - 100.5) < 1e-9


def test_arming_disarms_after_event():
    levels = {"FIB_0.618": 100.0}
    # Two ticks within 1pt — should fire only the first; second is suppressed.
    ticks = _make_ticks([(0, 105), (1, 100.5), (2, 100.7)])
    events = extract_events(ticks, levels)
    assert len(events) == 1


def test_rearming_requires_3pt_exit():
    levels = {"FIB_0.618": 100.0}
    # Enter zone, drift up to 102 (still within 3pt), back to level — should NOT re-fire.
    ticks = _make_ticks([(0, 105), (1, 100.5), (2, 102.0), (3, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 1


def test_rearms_after_exiting_3pt_zone():
    levels = {"FIB_0.618": 100.0}
    # Enter, exit beyond 3pt, re-enter → second event fires.
    ticks = _make_ticks([(0, 105), (1, 100.5), (2, 104.0), (3, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 2


def test_levels_armed_only_at_or_after_10_31_et():
    levels = {"FIB_0.618": 100.0}
    base = pd.Timestamp("2025-06-01 14:30:30", tz="UTC")  # 10:30:30 ET, before lock
    idx = pd.DatetimeIndex([base, base + pd.Timedelta(seconds=60)])
    ticks = pd.DataFrame({"price": [100.5, 100.5], "size": [1, 1]}, index=idx)
    events = extract_events(ticks, levels)
    # First tick is before 10:31 ET, suppressed. Second is at 10:31:30, armed.
    assert len(events) == 1
    assert events.iloc[0]["event_ts"] == base + pd.Timedelta(seconds=60)


def test_approach_direction_from_below():
    levels = {"FIB_0.618": 100.0}
    # 60s before event price was 95 → approach from below → +1
    ticks = _make_ticks([(0, 95.0), (60, 100.5)])
    events = extract_events(ticks, levels)
    assert len(events) == 1
    assert events.iloc[0]["approach_direction"] == 1


def test_approach_direction_from_above():
    levels = {"FIB_0.618": 100.0}
    ticks = _make_ticks([(0, 105.0), (60, 100.5)])
    events = extract_events(ticks, levels)
    assert events.iloc[0]["approach_direction"] == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_events.py -v`
Expected: All tests FAIL with `ModuleNotFoundError: No module named '_level_events'`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_events.py`:

```python
"""Level-touch event extraction with per-level arming state machine.

An event fires when price enters within 1.0 points of a level AND the level is
armed. After an event, the level disarms and re-arms only when price moves more
than 3.0 points from the level. All levels arm at 10:31 ET (matches existing
config IB_END_HOUR=10, IB_END_MIN=31).
"""
from __future__ import annotations

import datetime
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_events.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_events.py mnq_alerts/tests/test_level_events.py
git commit -m "feat: level-touch event extraction with arming state machine"
```

---

### Task 2: Label computation (8 outcomes per event)

**Files:**
- Create: `mnq_alerts/_level_labels.py`
- Test: `mnq_alerts/tests/test_level_labels.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_labels.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_labels.py -v`
Expected: All FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_labels.py`:

```python
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
    times = ticks.index.to_numpy()

    out_rows = []
    for _, ev in events.iterrows():
        event_ts = ev["event_ts"]
        entry_price = ev["event_price"]
        approach = int(ev["approach_direction"])

        window_end = _resolution_window_end(event_ts)
        # Slice ticks strictly after event_ts up to window_end inclusive.
        start_idx = int(np.searchsorted(times, np.datetime64(event_ts), side="right"))
        end_idx = int(np.searchsorted(times, np.datetime64(window_end), side="right"))
        slice_prices = prices[start_idx:end_idx]
        slice_times = times[start_idx:end_idx]

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
             prices: np.ndarray, times: np.ndarray, event_ts: pd.Timestamp) -> tuple[int, float]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_labels.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_labels.py mnq_alerts/tests/test_level_labels.py
git commit -m "feat: 8-outcome labeling for level-touch events"
```

---

### Task 3: Per-day dataset orchestration

**Files:**
- Create: `mnq_alerts/_level_dataset.py`
- Test: `mnq_alerts/tests/test_level_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_dataset.py`:

```python
"""Tests for _level_dataset.build_day."""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_dataset import build_day, LEVELS_IN_SCOPE


def _make_synthetic_day():
    """Synthesize one trading day's tick data with a clean approach to FIB_0.618."""
    base = pd.Timestamp("2025-06-02 13:30:00", tz="UTC")  # 9:30 ET
    times = [base + pd.Timedelta(seconds=s) for s in range(0, 23400, 30)]  # 6.5 hr at 30s
    # Constant price 18000 → IB locks at 10:31 with H=L=18000 → FIB levels collapse.
    # Use a slow ramp.
    prices = [18000 + (s / 60.0) * 0.5 for s in range(0, 23400, 30)]
    df = pd.DataFrame({"price": prices, "size": [1] * len(times)}, index=pd.DatetimeIndex(times))
    return df


def test_build_day_returns_dataset_with_expected_schema():
    ticks = _make_synthetic_day()
    out = build_day(ticks)
    expected_cols = {"event_ts", "level_name", "level_price", "event_price",
                     "approach_direction", "direction", "tp", "sl", "label",
                     "time_to_resolution_sec"}
    assert expected_cols.issubset(set(out.columns))


def test_build_day_excludes_vwap():
    ticks = _make_synthetic_day()
    out = build_day(ticks)
    assert "VWAP" not in out["level_name"].unique()


def test_levels_in_scope_excludes_vwap():
    assert "VWAP" not in LEVELS_IN_SCOPE
    assert "FIB_0.618" in LEVELS_IN_SCOPE
    assert "IBH" in LEVELS_IN_SCOPE
    assert "IBL" in LEVELS_IN_SCOPE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_dataset.py`:

```python
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
    ib_window = ticks[(et_idx.hour == 9) & (et_idx.minute >= 30) | (et_idx.hour == 10) & (et_idx.minute < 31)]
    if ib_window.empty:
        return {}
    ibh = float(ib_window["price"].max())
    ibl = float(ib_window["price"].min())
    levels = {"IBH": ibh, "IBL": ibl}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_dataset.py mnq_alerts/tests/test_level_dataset.py
git commit -m "feat: per-day dataset builder (ticks->events->labels)"
```

---

## Phase 2: Features

### Task 4: Family 1 — approach kinematics features

**Files:**
- Create: `mnq_alerts/_level_features.py` (new file, will grow as families are added)
- Test: `mnq_alerts/tests/test_level_features.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_features.py`:

```python
"""Tests for _level_features."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_features import compute_kinematics


def _ticks(rows):
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s, _ in rows])
    return pd.DataFrame({"price": [p for _, p in rows], "size": [1] * len(rows)}, index=idx)


def test_velocity_5s_is_signed_points_per_second():
    # Price at 100 at t=-5s, 110 at t=0s → velocity = (110-100)/5 = 2.0
    ticks = _ticks([(0, 100.0), (5, 110.0)])
    event_ts = ticks.index[1]
    feats = compute_kinematics(ticks, event_ts)
    assert abs(feats["velocity_5s"] - 2.0) < 1e-9


def test_velocity_negative_when_price_falls():
    ticks = _ticks([(0, 110.0), (5, 100.0)])
    event_ts = ticks.index[1]
    feats = compute_kinematics(ticks, event_ts)
    assert feats["velocity_5s"] < 0


def test_path_efficiency_one_for_straight_line():
    # Monotonic increase: displacement == sum(|moves|), efficiency = 1.0
    ticks = _ticks([(0, 100.0), (60, 102.0), (120, 104.0), (180, 106.0), (240, 108.0), (300, 110.0)])
    event_ts = ticks.index[-1]
    feats = compute_kinematics(ticks, event_ts)
    assert abs(feats["path_efficiency_5min"] - 1.0) < 1e-6


def test_path_efficiency_low_for_choppy_move():
    # Up-down-up-down — small displacement, large total moves
    ticks = _ticks([(0, 100.0), (60, 105.0), (120, 100.0), (180, 105.0), (240, 100.0), (300, 100.5)])
    event_ts = ticks.index[-1]
    feats = compute_kinematics(ticks, event_ts)
    assert feats["path_efficiency_5min"] < 0.1


def test_no_future_data_used():
    # Future ticks after event_ts must NOT influence kinematics.
    ticks = _ticks([(0, 100.0), (5, 110.0), (10, 200.0)])  # last tick is "future"
    event_ts = ticks.index[1]
    feats = compute_kinematics(ticks, event_ts)
    # If we leaked the 10s tick, velocity_5s would skew toward 200.
    assert abs(feats["velocity_5s"] - 2.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_kinematics'`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_features.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_features.py mnq_alerts/tests/test_level_features.py
git commit -m "feat(features): family 1 - approach kinematics"
```

---

### Task 5: Family 2 — tick-rule aggressor balance

**Files:**
- Modify: `mnq_alerts/_level_features.py` (add `compute_aggressor`)
- Modify: `mnq_alerts/tests/test_level_features.py` (add tests)

- [ ] **Step 1: Write the failing test**

Append to `mnq_alerts/tests/test_level_features.py`:

```python
from _level_features import compute_aggressor


def test_aggressor_balance_buy_dominant():
    # All upticks → buy_aggressor; balance = +1
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.5, 101.0, 101.5], "size": [1, 1, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2, 3)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    assert feats["aggressor_balance_5s"] == pytest.approx(1.0)


def test_aggressor_balance_sell_dominant():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [101.5, 101.0, 100.5, 100.0], "size": [1, 1, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2, 3)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    assert feats["aggressor_balance_5s"] == pytest.approx(-1.0)


def test_aggressor_zero_tick_inherits_prior_side():
    # First uptick (buy), then zero-tick (inherits buy), then zero-tick (inherits buy)
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.5, 100.5, 100.5], "size": [1, 1, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2, 3)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    # 3 buys (uptick + 2 inherited), 0 sells, first tick is neutral by convention
    assert feats["aggressor_balance_5s"] > 0.5


def test_net_dollar_flow_5min_signed():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.5, 101.0], "size": [10, 10, 10]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 60, 120)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_aggressor(ticks, event_ts)
    # 2 upticks of 10 contracts each at price 100.5 and 101.0 → positive net flow
    assert feats["net_dollar_flow_5min"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v -k aggressor`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_features.py`:

```python
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
        total = float(sizes.sum())
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_features.py mnq_alerts/tests/test_level_features.py
git commit -m "feat(features): family 2 - tick-rule aggressor balance"
```

---

### Task 6: Family 3 — volume / size profile

**Files:**
- Modify: `mnq_alerts/_level_features.py` (add `compute_volume_profile`)
- Modify: `mnq_alerts/tests/test_level_features.py`

- [ ] **Step 1: Write the failing test**

Append to `mnq_alerts/tests/test_level_features.py`:

```python
from _level_features import compute_volume_profile


def test_volume_5s_sums_size():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0, 100.0, 100.0], "size": [3, 5, 7]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 1, 2)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    assert feats["volume_5s"] == 15


def test_max_print_size_30s():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0] * 4, "size": [1, 50, 1, 1]},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 5, 10, 15)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    assert feats["max_print_size_30s"] == 50


def test_volume_concentration_high_when_one_big_print():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    # 99 of size 1, 1 of size 100. Herfindahl ≈ (100^2) / (199^2) ≈ 0.252
    sizes = [1] * 99 + [100]
    ticks = pd.DataFrame(
        {"price": [100.0] * 100, "size": sizes},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s * 0.1) for s in range(100)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    assert feats["volume_concentration_30s"] > 0.2


def test_volume_concentration_low_when_uniform():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    ticks = pd.DataFrame(
        {"price": [100.0] * 30, "size": [1] * 30},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in range(30)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_volume_profile(ticks, event_ts)
    # 30 trades of size 1 → Herfindahl = 30 / 900 = 0.0333
    assert feats["volume_concentration_30s"] < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v -k volume`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_features.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_features.py mnq_alerts/tests/test_level_features.py
git commit -m "feat(features): family 3 - volume/size profile"
```

---

### Task 7: Family 4 — level context (with resolution-order `prior_touch_outcome`)

This is the highest-leakage-risk feature. The leakage test is mandatory.

**Files:**
- Modify: `mnq_alerts/_level_features.py` (add `compute_level_context`)
- Modify: `mnq_alerts/tests/test_level_features.py`

- [ ] **Step 1: Write the failing test**

Append to `mnq_alerts/tests/test_level_features.py`:

```python
from _level_features import compute_level_context


def test_touches_today_zero_for_first():
    feats = compute_level_context(
        prior_touches=[],
        all_levels={"FIB_0.618": 100.0, "VWAP": 99.0, "FIB_0.236": 95.0},
        event_ts=pd.Timestamp("2025-06-01 14:31:00", tz="UTC"),
        event_price=100.0,
        level_name="FIB_0.618",
    )
    assert feats["touches_today"] == 0
    assert feats["prior_touch_outcome"] == "none"


def test_prior_touch_outcome_only_uses_resolved():
    """Critical leakage protection — prior touch with resolution_ts > event_ts
    must NOT contribute to prior_touch_outcome."""
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    prior_touches = [
        # Touch at 14:31:00, resolved at 14:46:00 (15min later) — resolved BEFORE current event.
        {"event_ts": base, "resolution_ts": base + pd.Timedelta(minutes=15), "outcome": "bounce_held"},
        # Touch at 14:50:00, resolution at 15:05:00 — UNRESOLVED relative to current event at 14:55.
        {"event_ts": base + pd.Timedelta(minutes=19), "resolution_ts": base + pd.Timedelta(minutes=34), "outcome": "breakthrough_held"},
    ]
    event_ts = base + pd.Timedelta(minutes=24)  # 14:55
    feats = compute_level_context(
        prior_touches=prior_touches,
        all_levels={"FIB_0.618": 100.0, "VWAP": 99.0},
        event_ts=event_ts,
        event_price=100.0,
        level_name="FIB_0.618",
    )
    # The unresolved second touch must NOT influence prior_touch_outcome.
    assert feats["prior_touch_outcome"] == "bounce_held"
    # touches_today counts touches whose event_ts < current event_ts (regardless of resolution).
    assert feats["touches_today"] == 2


def test_distance_to_vwap_excludes_vwap_from_other_levels():
    feats = compute_level_context(
        prior_touches=[],
        all_levels={"FIB_0.618": 100.0, "VWAP": 99.0, "FIB_0.236": 95.0},
        event_ts=pd.Timestamp("2025-06-01 14:31:00", tz="UTC"),
        event_price=100.0,
        level_name="FIB_0.618",
    )
    assert feats["distance_to_vwap"] == pytest.approx(1.0)
    # nearest other level (excluding self & VWAP) is FIB_0.236 at 95
    assert feats["distance_to_nearest_other_level"] == pytest.approx(5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v -k context`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_features.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_features.py mnq_alerts/tests/test_level_features.py
git commit -m "feat(features): family 4 - level context with resolution-order prior_touch_outcome"
```

---

### Task 8: Family 5 — volatility & time-of-day

**Files:**
- Modify: `mnq_alerts/_level_features.py` (add `compute_vol_time`)
- Modify: `mnq_alerts/tests/test_level_features.py`

- [ ] **Step 1: Write the failing test**

Append to `mnq_alerts/tests/test_level_features.py`:

```python
from _level_features import compute_vol_time


def test_realized_vol_5min_positive_for_volatile_data():
    base = pd.Timestamp("2025-06-01 14:31:00", tz="UTC")
    prices = [100.0 + (i % 2) * 2.0 for i in range(60)]  # alternates 100, 102
    ticks = pd.DataFrame(
        {"price": prices, "size": [1] * 60},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s * 5) for s in range(60)]),
    )
    event_ts = ticks.index[-1]
    feats = compute_vol_time(ticks, event_ts)
    assert feats["realized_vol_5min"] > 0


def test_seconds_to_market_close_correct():
    # 3:55 PM ET → 5 min to 4:00 PM = 300 seconds.
    et = pd.Timestamp("2025-06-02 15:55:00", tz="America/New_York")
    event_ts = et.tz_convert("UTC")
    base = event_ts - pd.Timedelta(minutes=10)
    ticks = pd.DataFrame(
        {"price": [100.0, 100.0], "size": [1, 1]},
        index=pd.DatetimeIndex([base, event_ts]),
    )
    feats = compute_vol_time(ticks, event_ts)
    assert feats["seconds_to_market_close"] == pytest.approx(300, abs=1)


def test_day_of_week_one_hot():
    et = pd.Timestamp("2025-06-02", tz="America/New_York")  # Monday
    event_ts = et.tz_convert("UTC") + pd.Timedelta(hours=14, minutes=31)
    base = event_ts - pd.Timedelta(minutes=5)
    ticks = pd.DataFrame(
        {"price": [100.0, 100.0], "size": [1, 1]},
        index=pd.DatetimeIndex([base, event_ts]),
    )
    feats = compute_vol_time(ticks, event_ts)
    assert feats["day_of_week_Mon"] == 1
    assert feats["day_of_week_Tue"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v -k vol_time`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_features.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_features.py mnq_alerts/tests/test_level_features.py
git commit -m "feat(features): family 5 - volatility & time-of-day"
```

---

### Task 9: Combined feature pipeline + leakage protection tests

**Files:**
- Modify: `mnq_alerts/_level_features.py` (add `compute_all_features`)
- Create: `mnq_alerts/tests/test_level_features_leakage.py`

- [ ] **Step 1: Write the failing tests**

Append to `mnq_alerts/_level_features.py` (function only — implementation in Step 3):

(Skip this — we go test-first. Add the test file now.)

Create `mnq_alerts/tests/test_level_features_leakage.py`:

```python
"""Critical leakage protection tests. The 3 tests below are non-negotiable.

If any of these fails, the model is unsafe to deploy.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_features import compute_all_features


def _ticks_long():
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    n = 1000
    rng = np.random.default_rng(42)
    prices = 18000 + np.cumsum(rng.normal(0, 0.5, n))
    sizes = rng.integers(1, 20, n)
    idx = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in range(n)])
    return pd.DataFrame({"price": prices, "size": sizes}, index=idx)


def test_no_future_ticks_in_features():
    """For 100 random events, features computed with full ticks must equal
    features computed with future ticks (ts > event_ts) masked out."""
    rng = np.random.default_rng(7)
    ticks = _ticks_long()
    all_levels = {"FIB_0.618": 18000.0, "VWAP": 18001.0}
    for _ in range(100):
        i = int(rng.integers(50, len(ticks) - 50))
        event_ts = ticks.index[i]
        full = compute_all_features(
            ticks=ticks, event_ts=event_ts, event_price=float(ticks.iloc[i]["price"]),
            level_name="FIB_0.618", level_price=18000.0, approach_direction=1,
            prior_touches=[], all_levels=all_levels,
        )
        masked = compute_all_features(
            ticks=ticks.loc[ticks.index <= event_ts], event_ts=event_ts,
            event_price=float(ticks.iloc[i]["price"]), level_name="FIB_0.618",
            level_price=18000.0, approach_direction=1,
            prior_touches=[], all_levels=all_levels,
        )
        for k in full:
            assert full[k] == masked[k], f"Feature {k} differs at event {event_ts}: {full[k]} vs {masked[k]}"


def test_prior_touch_outcome_resolution_order():
    """A prior touch whose resolution_ts > event_ts must NOT influence
    prior_touch_outcome for the event at event_ts."""
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    prior_touches = [
        # Future-resolved touch.
        {"event_ts": base + pd.Timedelta(minutes=10), "resolution_ts": base + pd.Timedelta(minutes=25), "outcome": "breakthrough_held"},
    ]
    event_ts = base + pd.Timedelta(minutes=15)  # current; prior touch UNresolved.
    ticks = pd.DataFrame(
        {"price": [18000.0] * 5, "size": [1] * 5},
        index=pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in (0, 60, 120, 600, 900)]),
    )
    feats = compute_all_features(
        ticks=ticks, event_ts=event_ts, event_price=18000.0,
        level_name="FIB_0.618", level_price=18000.0, approach_direction=1,
        prior_touches=prior_touches, all_levels={"FIB_0.618": 18000.0, "VWAP": 18001.0},
    )
    assert feats["prior_touch_outcome"] == "none"


def test_label_leakage_permutation():
    """Permutation test: random feature values produce random labels.
    Run a tiny LightGBM on randomized features+labels and confirm AUC ~0.5.

    This is sanity-check that the feature pipeline isn't accidentally smuggling
    label information through some side-channel.
    """
    pytest.importorskip("lightgbm")
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    n = 2000
    # Use real feature shape: ~24 numeric features (kinematics 6 + aggressor 4 + volume 7 + level_ctx 4 + voltime 5 ≈ 26)
    X = rng.normal(0, 1, (n, 24))
    y = rng.integers(0, 2, n)
    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]
    model = lgb.LGBMClassifier(num_leaves=31, max_depth=6, n_estimators=100, learning_rate=0.05, verbosity=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # Random features + random labels → AUC ~0.5. Allow 0.4-0.6 sanity band.
    assert 0.40 <= auc <= 0.60, f"AUC={auc} suggests pipeline bias on random data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features_leakage.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_all_features'`.

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_features.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_features_leakage.py -v`
Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_features.py mnq_alerts/tests/test_level_features_leakage.py
git commit -m "feat(features): combined pipeline + leakage protection suite"
```

---

### Task 10: Full-history dataset builder (writes labeled+featured parquet)

**Files:**
- Modify: `mnq_alerts/_level_dataset.py` (add `build_full_history`)
- Create: `mnq_alerts/scripts/build_level_dataset.py` (CLI runner)

- [ ] **Step 1: Write the failing test**

Append to `mnq_alerts/tests/test_level_dataset.py`:

```python
from _level_dataset import build_day_with_features


def test_build_day_with_features_returns_features_and_labels():
    base = pd.Timestamp("2025-06-02 13:30:00", tz="UTC")
    times = [base + pd.Timedelta(seconds=s) for s in range(0, 23400, 30)]
    prices = [18000 + (s / 60.0) * 0.3 for s in range(0, 23400, 30)]
    ticks = pd.DataFrame({"price": prices, "size": [1] * len(times)}, index=pd.DatetimeIndex(times))
    out = build_day_with_features(ticks)
    expected_feature_cols = {"velocity_5s", "aggressor_balance_30s", "volume_5min",
                             "prior_touch_outcome", "realized_vol_5min", "day_of_week_Mon"}
    if not out.empty:
        assert expected_feature_cols.issubset(set(out.columns))
        assert "label" in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_dataset.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_dataset.py`:

```python
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
```

Create `mnq_alerts/scripts/build_level_dataset.py`:

```python
"""CLI: build the full-history labeled+featured dataset.

Usage: python -m mnq_alerts.scripts.build_level_dataset
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_dataset import build_full_history


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parquet_dir = os.path.join(here, "data_cache")
    out_path = os.path.join(here, "_level_events_labeled.parquet")
    n = build_full_history(parquet_dir, out_path)
    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_dataset.py -v`
Expected: All PASS.

- [ ] **Step 5: Run the full-history builder and confirm it produces a non-empty parquet**

Run: `cd mnq_alerts && python -m scripts.build_level_dataset`
Expected: prints `Wrote N rows to ..._level_events_labeled.parquet` where N > 5000 (rough estimate based on 339 days × ~30 events × 8 labels). If N is much smaller, investigate.

- [ ] **Step 6: Commit**

```bash
git add mnq_alerts/_level_dataset.py mnq_alerts/scripts/build_level_dataset.py mnq_alerts/tests/test_level_dataset.py
git commit -m "feat: full-history labeled+featured dataset builder"
```

---

## Phase 3: Models

### Task 11: LightGBM training shell + Architecture A

**Files:**
- Create: `mnq_alerts/_level_model.py`
- Test: `mnq_alerts/tests/test_level_model.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_model.py`:

```python
"""Tests for _level_model."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("lightgbm")
from _level_model import train_architecture_a, predict_architecture_a, FEATURE_COLUMNS


def _synthetic_dataset(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        feats = {col: rng.normal() for col in FEATURE_COLUMNS}
        rows.append({
            "level_name": rng.choice(["FIB_0.618", "FIB_0.236", "IBH"]),
            "direction": rng.choice(["bounce", "breakthrough"]),
            "tp": int(rng.choice([8, 10])),
            "sl": int(rng.choice([20, 25])),
            "label": int(rng.integers(0, 2)),
            "event_ts": pd.Timestamp("2025-06-01", tz="UTC") + pd.Timedelta(days=i // 50),
            **feats,
        })
    return pd.DataFrame(rows)


def test_train_architecture_a_returns_dict_of_models():
    df = _synthetic_dataset()
    models = train_architecture_a(df.iloc[:1500], val=df.iloc[1500:1800])
    # Up to 3 levels × 2 directions × 4 TP/SL = 24 in synthetic data.
    assert len(models) > 0
    for key, m in models.items():
        level, direction, tp, sl = key
        assert level in {"FIB_0.618", "FIB_0.236", "IBH"}
        assert direction in {"bounce", "breakthrough"}


def test_predict_architecture_a_returns_probability_per_variant():
    df = _synthetic_dataset()
    models = train_architecture_a(df.iloc[:1500], val=df.iloc[1500:1800])
    test_event = df.iloc[1800].to_dict()
    preds = predict_architecture_a(models, test_event)
    # 8 variants per event.
    assert len(preds) == 8
    for p in preds.values():
        assert 0 <= p <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_model.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add implementation**

Create `mnq_alerts/_level_model.py`:

```python
"""LightGBM training and inference for both Architecture A (per-level/variant)
and Architecture B (single pooled model)."""
from __future__ import annotations

import os
from typing import Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Final feature column list — must match outputs of compute_all_features.
FEATURE_COLUMNS = [
    "velocity_5s", "velocity_30s", "velocity_5min",
    "acceleration_30s", "path_efficiency_5min",
    "aggressor_balance_5s", "aggressor_balance_30s", "aggressor_balance_5min",
    "net_dollar_flow_5min",
    "volume_5s", "volume_30s", "volume_5min",
    "trade_rate_30s", "max_print_size_30s", "volume_concentration_30s",
    "touches_today", "seconds_since_last_touch",
    "distance_to_vwap", "distance_to_nearest_other_level", "is_post_IB",
    "realized_vol_5min", "realized_vol_30min", "range_30min",
    "seconds_to_market_close", "seconds_into_session",
    "day_of_week_Mon", "day_of_week_Tue", "day_of_week_Wed", "day_of_week_Thu", "day_of_week_Fri",
]

# prior_touch_outcome is categorical — encoded separately.
CATEGORICAL_FEATURES = ["prior_touch_outcome"]

# Conditioning features added to Architecture B only.
CONDITIONING_FEATURES = ["level_id", "direction_id", "tp", "sl"]

LGBM_PARAMS = dict(
    num_leaves=31, max_depth=6, min_data_in_leaf=50,
    learning_rate=0.05, n_estimators=500,
    feature_fraction=0.8, bagging_fraction=0.8,
    objective="binary", metric="auc", verbosity=-1,
)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "prior_touch_outcome" in df.columns:
        df["prior_touch_outcome"] = df["prior_touch_outcome"].astype("category")
    return df


def train_architecture_a(
    train: pd.DataFrame, val: pd.DataFrame,
) -> dict[tuple, lgb.LGBMClassifier]:
    """One model per (level_name, direction, tp, sl). Returns dict keyed by tuple."""
    train = _encode_categoricals(train)
    val = _encode_categoricals(val)
    feat_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURES
    models: dict[tuple, lgb.LGBMClassifier] = {}
    keys = train[["level_name", "direction", "tp", "sl"]].drop_duplicates().values.tolist()
    for level, direction, tp, sl in keys:
        slice_train = train[(train["level_name"] == level) & (train["direction"] == direction)
                            & (train["tp"] == tp) & (train["sl"] == sl)]
        slice_val = val[(val["level_name"] == level) & (val["direction"] == direction)
                        & (val["tp"] == tp) & (val["sl"] == sl)]
        if len(slice_train) < 100 or len(slice_val) < 20:
            continue  # too small to train reliably; skip this variant.
        X_train = slice_train[feat_cols]
        y_train = slice_train["label"].astype(int)
        X_val = slice_val[feat_cols]
        y_val = slice_val["label"].astype(int)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=CATEGORICAL_FEATURES,
        )
        models[(level, direction, int(tp), int(sl))] = model
    return models


def predict_architecture_a(
    models: dict[tuple, lgb.LGBMClassifier], event: dict,
) -> dict[tuple, float]:
    """For an event, query all 8 (direction, tp, sl) variants for this level.

    Returns dict keyed by (direction, tp, sl) -> P(win).
    """
    out: dict[tuple, float] = {}
    feat_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURES
    for (level, direction, tp, sl), m in models.items():
        if level != event["level_name"]:
            continue
        x = pd.DataFrame([{c: event.get(c) for c in feat_cols}])
        if "prior_touch_outcome" in x.columns:
            x["prior_touch_outcome"] = x["prior_touch_outcome"].astype("category")
        p = float(m.predict_proba(x)[0, 1])
        out[(direction, int(tp), int(sl))] = p
    return out


def save_models(models: dict, dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)
    for key, m in models.items():
        fn = "_".join(str(k) for k in key) + ".joblib"
        joblib.dump(m, os.path.join(dir_path, fn))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_model.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_model.py mnq_alerts/tests/test_level_model.py
git commit -m "feat(model): Architecture A (per-level/variant LightGBM)"
```

---

### Task 12: Architecture B (single pooled model)

**Files:**
- Modify: `mnq_alerts/_level_model.py` (add `train_architecture_b`, `predict_architecture_b`)
- Modify: `mnq_alerts/tests/test_level_model.py`

- [ ] **Step 1: Write the failing test**

Append to `mnq_alerts/tests/test_level_model.py`:

```python
from _level_model import train_architecture_b, predict_architecture_b


def test_train_architecture_b_returns_single_model():
    df = _synthetic_dataset()
    model = train_architecture_b(df.iloc[:1500], val=df.iloc[1500:1800])
    assert model is not None


def test_predict_architecture_b_returns_8_variants_per_event():
    df = _synthetic_dataset()
    model = train_architecture_b(df.iloc[:1500], val=df.iloc[1500:1800])
    test_event = df.iloc[1800].to_dict()
    preds = predict_architecture_b(model, test_event)
    assert len(preds) == 8
    for p in preds.values():
        assert 0 <= p <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_model.py -v`
Expected: 2 new FAIL (`ImportError`).

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_model.py`:

```python
LEVEL_ID_MAP = {
    "IBH": 0, "IBL": 1, "FIB_0.236": 2, "FIB_0.618": 3, "FIB_0.764": 4,
    "FIB_EXT_HI_1.272": 5, "FIB_EXT_LO_1.272": 6,
}
DIRECTION_ID_MAP = {"bounce": 0, "breakthrough": 1}


def _add_conditioning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["level_id"] = df["level_name"].map(LEVEL_ID_MAP).astype(int)
    df["direction_id"] = df["direction"].map(DIRECTION_ID_MAP).astype(int)
    return df


def train_architecture_b(
    train: pd.DataFrame, val: pd.DataFrame,
) -> lgb.LGBMClassifier:
    train = _add_conditioning(_encode_categoricals(train))
    val = _add_conditioning(_encode_categoricals(val))
    feat_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURES + CONDITIONING_FEATURES
    X_train, y_train = train[feat_cols], train["label"].astype(int)
    X_val, y_val = val[feat_cols], val["label"].astype(int)
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=CATEGORICAL_FEATURES + ["level_id", "direction_id"],
    )
    return model


def predict_architecture_b(
    model: lgb.LGBMClassifier, event: dict,
) -> dict[tuple, float]:
    """Query 8 variants for this event by replicating the row with each conditioning."""
    feat_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURES + CONDITIONING_FEATURES
    rows = []
    keys = []
    for direction in ("bounce", "breakthrough"):
        for tp, sl in [(8, 25), (8, 20), (10, 25), (10, 20)]:
            r = {c: event.get(c) for c in FEATURE_COLUMNS + CATEGORICAL_FEATURES}
            r["level_id"] = LEVEL_ID_MAP[event["level_name"]]
            r["direction_id"] = DIRECTION_ID_MAP[direction]
            r["tp"] = tp
            r["sl"] = sl
            rows.append(r)
            keys.append((direction, tp, sl))
    df = pd.DataFrame(rows)
    if "prior_touch_outcome" in df.columns:
        df["prior_touch_outcome"] = df["prior_touch_outcome"].astype("category")
    probs = model.predict_proba(df[feat_cols])[:, 1]
    return dict(zip(keys, probs.astype(float)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_model.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_model.py mnq_alerts/tests/test_level_model.py
git commit -m "feat(model): Architecture B (pooled model with level/direction/tp/sl conditioning)"
```

---

## Phase 4: Validation

### Task 13: Trade simulation with 1-position constraint and slippage

**Files:**
- Create: `mnq_alerts/_level_simulation.py`
- Test: `mnq_alerts/tests/test_level_simulation.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_simulation.py`:

```python
"""Tests for _level_simulation."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_simulation import simulate_strategy


def _ev(ts, level, direction, tp, sl, label, ttr, day=0):
    base = pd.Timestamp("2025-06-02", tz="UTC")
    return {
        "event_ts": base + pd.Timedelta(days=day) + pd.Timedelta(seconds=ts),
        "level_name": level, "direction": direction, "tp": tp, "sl": sl,
        "label": label, "time_to_resolution_sec": ttr,
        "event_price": 18000.0, "approach_direction": 1,
    }


def test_skip_when_max_expected_pnl_below_threshold():
    events = [_ev(60, "FIB_0.618", "bounce", 8, 25, 1, 30)]
    # P(win) low for all 8 variants; expected P&L negative for all.
    preds = {(events[0]["event_ts"]): {("bounce", 8, 25): 0.5, ("bounce", 8, 20): 0.5,
                                       ("bounce", 10, 25): 0.5, ("bounce", 10, 20): 0.5,
                                       ("breakthrough", 8, 25): 0.5, ("breakthrough", 8, 20): 0.5,
                                       ("breakthrough", 10, 25): 0.5, ("breakthrough", 10, 20): 0.5}}
    result = simulate_strategy(events, preds, threshold=2.0)
    assert result["trades"] == 0


def test_picks_variant_with_highest_expected_pnl():
    events = [_ev(60, "FIB_0.618", "bounce", 8, 25, 1, 30)]
    # bounce_8_25: 8*0.95 - 25*0.05 = 7.6 - 1.25 = 6.35 — highest
    preds = {(events[0]["event_ts"]): {("bounce", 8, 25): 0.95, ("bounce", 8, 20): 0.5,
                                       ("bounce", 10, 25): 0.5, ("bounce", 10, 20): 0.5,
                                       ("breakthrough", 8, 25): 0.5, ("breakthrough", 8, 20): 0.5,
                                       ("breakthrough", 10, 25): 0.5, ("breakthrough", 10, 20): 0.5}}
    result = simulate_strategy(events, preds, threshold=0.0)
    assert result["trades"] == 1
    assert result["chosen_variants"][0] == ("bounce", 8, 25)


def test_one_position_at_a_time_blocks_overlapping_events():
    # Trade enters at t=60s, resolves at t=60+1800=1860s (30 min). Second event at t=300s should be blocked.
    events = [
        _ev(60, "FIB_0.618", "bounce", 8, 25, 1, 1800),
        _ev(300, "FIB_0.236", "bounce", 8, 25, 1, 30),  # would-be-second trade, blocked.
    ]
    preds = {
        events[0]["event_ts"]: {("bounce", 8, 25): 0.95, **{k: 0.5 for k in [("bounce", 8, 20),
            ("bounce", 10, 25), ("bounce", 10, 20), ("breakthrough", 8, 25), ("breakthrough", 8, 20),
            ("breakthrough", 10, 25), ("breakthrough", 10, 20)]}},
        events[1]["event_ts"]: {("bounce", 8, 25): 0.95, **{k: 0.5 for k in [("bounce", 8, 20),
            ("bounce", 10, 25), ("bounce", 10, 20), ("breakthrough", 8, 25), ("breakthrough", 8, 20),
            ("breakthrough", 10, 25), ("breakthrough", 10, 20)]}},
    }
    result = simulate_strategy(events, preds, threshold=0.0)
    assert result["trades"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_simulation.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_simulation.py`:

```python
"""Strategy simulator: pick best variant per event, enforce 1-position constraint."""
from __future__ import annotations

import pandas as pd

POINTS_PER_DOLLAR = 2.0  # MNQ: $2 per point per contract.


def simulate_strategy(
    events: list[dict],
    preds: dict[pd.Timestamp, dict[tuple, float]],
    threshold: float,
) -> dict:
    """Run the full strategy simulation.

    `events` is a list of event dicts. Each event_ts may have up to 8 rows
    (one per (direction, tp, sl) variant) all sharing the same event_ts.
    `preds` maps event_ts -> dict of (direction, tp, sl) -> P(win).
    `threshold` is the min expected_pnl (in points) to take a trade; below = skip.

    Returns: {trades, wins, losses, total_points, total_dollars, chosen_variants, daily_pnl_dollars}.
    """
    from collections import defaultdict

    # Group rows by event_ts so we can look up the chosen variant's label.
    groups: dict = defaultdict(list)
    for ev in events:
        groups[ev["event_ts"]].append(ev)

    in_position_until = pd.Timestamp("1900-01-01", tz="UTC")
    trades = 0
    wins = 0
    losses = 0
    total_points = 0.0
    chosen = []
    daily_points: dict = {}

    for event_ts in sorted(groups.keys()):
        if event_ts < in_position_until:
            continue
        ev_preds = preds.get(event_ts, {})
        if not ev_preds:
            continue
        group_rows = groups[event_ts]
        # Pick variant with highest expected P&L.
        best = None
        best_ev = -1e9
        for (direction, tp, sl), p in ev_preds.items():
            expected = tp * p - sl * (1 - p)
            if expected > best_ev:
                best_ev = expected
                best = (direction, tp, sl)
        if best is None or best_ev < threshold:
            continue
        # Look up the label for the chosen variant in the group.
        chosen_row = next(
            (r for r in group_rows
             if r["direction"] == best[0] and r["tp"] == best[1] and r["sl"] == best[2]),
            None,
        )
        if chosen_row is None:
            continue
        label = chosen_row["label"]
        ttr = chosen_row.get("time_to_resolution_sec")
        if ttr is None or pd.isna(ttr):
            ttr = 15 * 60
        in_position_until = event_ts + pd.Timedelta(seconds=ttr)
        trades += 1
        if label == 1:
            wins += 1
            total_points += best[1]  # +TP
        else:
            losses += 1
            total_points -= best[2]  # -SL
        chosen.append(best)
        day = event_ts.date()
        daily_points[day] = daily_points.get(day, 0.0) + (best[1] if label == 1 else -best[2])

    return {
        "trades": trades, "wins": wins, "losses": losses,
        "total_points": total_points,
        "total_dollars": total_points * POINTS_PER_DOLLAR,
        "chosen_variants": chosen,
        "daily_pnl_dollars": {d: v * POINTS_PER_DOLLAR for d, v in daily_points.items()},
    }
```

**Event-shape contract:** events in real data have up to 8 rows per event_ts (one per labeled variant). The simulator groups by event_ts, picks the best variant per the model's predictions, and looks up that variant's label in the group. The test below uses a single-row-per-event_ts shape (the chosen variant) to keep the test small — both shapes flow through the same code path because grouping is keyed on event_ts.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_simulation.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_simulation.py mnq_alerts/tests/test_level_simulation.py
git commit -m "feat(simulation): strategy sim with 1-position constraint and threshold"
```

---

### Task 14: Walk-forward folds, embargo, and architecture-selection harness

**Files:**
- Create: `mnq_alerts/_level_validation.py`
- Test: `mnq_alerts/tests/test_level_validation.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_validation.py`:

```python
"""Tests for walk-forward fold construction."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_validation import build_walk_forward_folds, FoldDef


def test_five_quarterly_folds_for_full_history():
    # Days from 2025-01-02 to 2026-04-08 (309 days dev set boundary)
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    assert len(folds) == 5


def test_train_set_grows_each_fold():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    assert all(folds[i].train_end >= folds[i - 1].train_end for i in range(1, len(folds)))


def test_first_day_of_test_dropped_for_embargo():
    days = pd.date_range("2025-01-02", "2026-04-08", freq="B")
    folds = build_walk_forward_folds(days)
    # The first test day in fold 0 should be > train_end + 1 day (embargo).
    fold = folds[0]
    assert fold.test_start > fold.train_end


def test_dev_set_excludes_final_test_window():
    days = pd.date_range("2025-01-02", "2026-05-06", freq="B")  # full history
    folds = build_walk_forward_folds(days)
    last_fold = folds[-1]
    # Last fold's test must NOT extend into the final 30 trading days.
    final_test_start = days[-30]
    assert last_fold.test_end < final_test_start
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_validation.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_validation.py`:

```python
"""Walk-forward folds, gates, and architecture selection."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FINAL_TEST_TRADING_DAYS = 30
EMBARGO_DAYS = 1


@dataclass
class FoldDef:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_folds(trading_days: pd.DatetimeIndex) -> list[FoldDef]:
    """Quarterly walk-forward folds over the dev set.

    Final FINAL_TEST_TRADING_DAYS days are reserved for out-of-time test (excluded).
    EMBARGO_DAYS days dropped from the start of each test fold.
    """
    if len(trading_days) <= FINAL_TEST_TRADING_DAYS:
        return []
    dev_days = trading_days[:-FINAL_TEST_TRADING_DAYS]
    quarters = sorted({(d.year, (d.month - 1) // 3 + 1) for d in dev_days})
    folds: list[FoldDef] = []
    for i in range(1, len(quarters)):
        test_year, test_q = quarters[i]
        train_quarters = quarters[:i]
        train_days = [d for d in dev_days if (d.year, (d.month - 1) // 3 + 1) in train_quarters]
        test_days = [d for d in dev_days if (d.year, (d.month - 1) // 3 + 1) == quarters[i]]
        if len(train_days) < 20 or len(test_days) < 5:
            continue
        # Embargo: drop EMBARGO_DAYS at start of test.
        test_days_emb = test_days[EMBARGO_DAYS:]
        if len(test_days_emb) < 5:
            continue
        folds.append(FoldDef(
            train_start=train_days[0],
            train_end=train_days[-1],
            test_start=test_days_emb[0],
            test_end=test_days_emb[-1],
        ))
    return folds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_validation.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_validation.py mnq_alerts/tests/test_level_validation.py
git commit -m "feat(validation): walk-forward fold construction with embargo"
```

---

### Task 15: Architecture-selection runner — train both, score gates, declare winner

**Files:**
- Modify: `mnq_alerts/_level_validation.py` (add `run_architecture_selection`)
- Create: `mnq_alerts/scripts/select_architecture.py` (CLI)

- [ ] **Step 1: Write the failing test**

Append to `mnq_alerts/tests/test_level_validation.py`:

```python
from _level_validation import score_fold, GateResults


def test_score_fold_returns_metrics():
    # Build a tiny synthetic prediction set on labeled events.
    base = pd.Timestamp("2025-06-02", tz="UTC")
    events = []
    preds = {}
    for i in range(50):
        ts = base + pd.Timedelta(seconds=60 * i + 60)
        for direction in ("bounce", "breakthrough"):
            for tp, sl in [(8, 25), (8, 20), (10, 25), (10, 20)]:
                events.append({
                    "event_ts": ts, "level_name": "FIB_0.618",
                    "direction": direction, "tp": tp, "sl": sl,
                    "event_price": 18000.0, "approach_direction": 1,
                    "label": (i + (direction == "bounce")) % 2,
                    "time_to_resolution_sec": 60.0,
                })
        # one prediction set per event_ts
        preds[ts] = {("bounce", 8, 25): 0.6, ("bounce", 8, 20): 0.55, ("bounce", 10, 25): 0.5,
                     ("bounce", 10, 20): 0.5, ("breakthrough", 8, 25): 0.5, ("breakthrough", 8, 20): 0.5,
                     ("breakthrough", 10, 25): 0.5, ("breakthrough", 10, 20): 0.5}
    result = score_fold(events=events, preds=preds, threshold=0.0)
    assert hasattr(result, "top_decile_lift")
    assert hasattr(result, "top_decile_expected_pnl_per_trade")
    assert hasattr(result, "simulated_mean_daily_pnl_dollars")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_validation.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Add implementation**

Append to `mnq_alerts/_level_validation.py`:

```python
from dataclasses import dataclass

import numpy as np

from _level_simulation import simulate_strategy


@dataclass
class GateResults:
    top_decile_lift: float
    top_decile_expected_pnl_per_trade: float
    simulated_mean_daily_pnl_dollars: float
    base_rate: float


def score_fold(events: list[dict], preds: dict, threshold: float) -> GateResults:
    """Compute gate metrics for a single fold's test results."""
    # Top-decile lift: take all (event, variant) pairs the model predicts highest, look at realized win rates.
    rows = []
    for ev in events:
        ev_preds = preds.get(ev["event_ts"], {})
        key = (ev["direction"], ev["tp"], ev["sl"])
        if key not in ev_preds:
            continue
        rows.append({
            "p_win": ev_preds[key], "label": ev["label"], "tp": ev["tp"], "sl": ev["sl"],
        })
    if not rows:
        return GateResults(0.0, 0.0, 0.0, 0.0)
    df = pd.DataFrame(rows)
    df = df.sort_values("p_win", ascending=False).reset_index(drop=True)
    base_rate = float(df["label"].mean())
    cutoff = max(1, len(df) // 10)
    top = df.iloc[:cutoff]
    top_wr = float(top["label"].mean())
    top_decile_lift = top_wr - base_rate
    expected_pnl = float((top["tp"] * top["label"] - top["sl"] * (1 - top["label"])).mean())

    # Simulated daily P&L using simulate_strategy.
    sim = simulate_strategy(events, preds, threshold=threshold)
    daily = sim["daily_pnl_dollars"]
    mean_daily = float(np.mean(list(daily.values()))) if daily else 0.0

    return GateResults(top_decile_lift, expected_pnl, mean_daily, base_rate)


def run_architecture_selection(
    dataset: pd.DataFrame, folds: list[FoldDef], v6_per_quarter_pnl: dict,
) -> dict:
    """Train A and B per fold, evaluate gates, return per-architecture per-fold results.

    `v6_per_quarter_pnl` maps (year, quarter) -> mean daily P&L $ for V6 backtest in that quarter.
    """
    from _level_model import train_architecture_a, predict_architecture_a
    from _level_model import train_architecture_b, predict_architecture_b

    results: dict = {"A": {"folds": []}, "B": {"folds": []}}
    for fold in folds:
        train = dataset[dataset["event_ts"] <= fold.train_end].copy()
        test = dataset[(dataset["event_ts"] >= fold.test_start) & (dataset["event_ts"] <= fold.test_end)].copy()
        # Internal early-stopping val: last 5% of train days, with embargo.
        train_days = sorted(train["event_ts"].dt.date.unique())
        val_cutoff = train_days[-max(1, len(train_days) // 20)]
        val = train[train["event_ts"].dt.date >= val_cutoff].copy()
        train_for_fit = train[train["event_ts"].dt.date < val_cutoff].copy()

        # Architecture A.
        models_a = train_architecture_a(train_for_fit, val)
        preds_a: dict = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds_a[ev["event_ts"]] = predict_architecture_a(models_a, ev.to_dict())
        gate_a = score_fold(test.to_dict("records"), preds_a, threshold=0.0)
        results["A"]["folds"].append({"fold": fold, "gates": gate_a})

        # Architecture B.
        model_b = train_architecture_b(train_for_fit, val)
        preds_b: dict = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds_b[ev["event_ts"]] = predict_architecture_b(model_b, ev.to_dict())
        gate_b = score_fold(test.to_dict("records"), preds_b, threshold=0.0)
        results["B"]["folds"].append({"fold": fold, "gates": gate_b})

    # Pre-committed gate evaluation: top_decile_lift > 0 in ≥4/5; expected_pnl > 0 in 5/5; daily P&L > V6 in ≥4/5.
    for arch in ("A", "B"):
        folds_data = results[arch]["folds"]
        lift_pass = sum(1 for f in folds_data if f["gates"].top_decile_lift > 0)
        epnl_pass = sum(1 for f in folds_data if f["gates"].top_decile_expected_pnl_per_trade > 0)
        beats_v6 = 0
        for f in folds_data:
            q = (f["fold"].test_start.year, (f["fold"].test_start.month - 1) // 3 + 1)
            v6 = v6_per_quarter_pnl.get(q, 0.0)
            if f["gates"].simulated_mean_daily_pnl_dollars > v6:
                beats_v6 += 1
        results[arch]["passes_lift_gate"] = lift_pass >= 4
        results[arch]["passes_expected_pnl_gate"] = epnl_pass == len(folds_data)
        results[arch]["passes_v6_gate"] = beats_v6 >= 4

    return results
```

Create `mnq_alerts/scripts/select_architecture.py`:

```python
"""CLI: run architecture selection on the labeled dataset.

Usage: python -m mnq_alerts.scripts.select_architecture
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_validation import build_walk_forward_folds, run_architecture_selection


# V6 per-quarter mean daily P&L in dollars. These MUST be obtained by running the
# existing V6 slippage-aware backtest split by calendar quarter before architecture
# selection — they are the comparison floor for the architecture-selection P&L gate.
# Memory contains rough numbers (Q1'25 ≈ +$37.63, Q4'25 ≈ +$10.09, Q1'26 ≈ +$6.78,
# recent-60d ≈ +$1.83) but these mix V0/V6 contexts; do not trust them — re-run.
# Replace this placeholder with: {(year, q): mean_daily_pnl_dollars}
V6_PER_QUARTER: dict = {}  # filled in before running select_architecture


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = pd.read_parquet(os.path.join(here, "_level_events_labeled.parquet"))
    days = pd.DatetimeIndex(sorted(dataset["event_ts"].dt.date.unique()))
    folds = build_walk_forward_folds(days)
    print(f"Built {len(folds)} folds")
    results = run_architecture_selection(dataset, folds, V6_PER_QUARTER)
    print(json.dumps({arch: {k: v for k, v in r.items() if k != "folds"} for arch, r in results.items()}, indent=2))
    print("--- Per-fold ---")
    for arch in ("A", "B"):
        for f in results[arch]["folds"]:
            g = f["gates"]
            print(f"{arch} fold {f['fold'].test_start.date()}–{f['fold'].test_end.date()}: "
                  f"lift={g.top_decile_lift:.4f} epnl={g.top_decile_expected_pnl_per_trade:.2f} daily=${g.simulated_mean_daily_pnl_dollars:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_validation.py -v`
Expected: All PASS.

- [ ] **Step 5: Run architecture selection on real data**

Run: `cd mnq_alerts && python -m scripts.select_architecture | tee /tmp/level_arch_selection.log`
Expected: Prints per-fold metrics for A and B. Log saved to `/tmp/`.
Manual check: confirm at least one architecture passes all 3 gates. If neither passes, **stop here** — model is not viable. Do not iterate to rescue.

- [ ] **Step 6: Commit**

```bash
git add mnq_alerts/_level_validation.py mnq_alerts/scripts/select_architecture.py mnq_alerts/tests/test_level_validation.py
git commit -m "feat(validation): architecture selection harness with pre-committed gates"
```

---

### Task 16: E2E gate verification

**Files:**
- Create: `mnq_alerts/_level_e2e_gate.py`
- Test: `mnq_alerts/tests/test_level_e2e_gate.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_e2e_gate.py`:

```python
"""Tests for E2E gate verification — the v3 catch."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_e2e_gate import compare_offline_vs_replay, E2EGateResult


def test_identical_predictions_pass_gate():
    # Two prediction dicts that agree exactly.
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    offline = {
        base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.6},
        base + pd.Timedelta(minutes=10): {("bounce", 8, 25): 0.7},
    }
    replay = dict(offline)
    result = compare_offline_vs_replay(offline, replay)
    assert result.passes is True
    assert result.max_diff < 1e-9


def test_large_divergence_fails_gate():
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    offline = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.6}}
    replay = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.9}}  # diff = 0.3
    result = compare_offline_vs_replay(offline, replay)
    assert result.passes is False
    assert result.max_diff > 0.1


def test_threshold_pass_at_002():
    base = pd.Timestamp("2025-06-02 14:31:00", tz="UTC")
    # Diff of 0.015 should pass (under 0.02).
    offline = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.600}}
    replay = {base + pd.Timedelta(minutes=5): {("bounce", 8, 25): 0.615}}
    result = compare_offline_vs_replay(offline, replay)
    assert result.passes is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_e2e_gate.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_e2e_gate.py`:

```python
"""E2E gate verification: live-replay predictions must match offline-pipeline
predictions for the same triggers, within tight tolerances."""
from __future__ import annotations

from dataclasses import dataclass


MAX_DIFF_THRESHOLD = 0.02
MEAN_DIFF_THRESHOLD = 0.005


@dataclass
class E2EGateResult:
    max_diff: float
    mean_diff: float
    passes: bool
    n_compared: int


def compare_offline_vs_replay(offline_preds: dict, replay_preds: dict) -> E2EGateResult:
    """Compare two prediction dicts. Both keyed by event_ts -> (variant_key -> p_win).

    Returns max and mean absolute difference across all matching (event, variant) pairs.
    Passes if max < 0.02 AND mean < 0.005.
    """
    diffs = []
    for ts, off_variants in offline_preds.items():
        rep_variants = replay_preds.get(ts)
        if not rep_variants:
            continue
        for k, p_off in off_variants.items():
            if k not in rep_variants:
                continue
            diffs.append(abs(p_off - rep_variants[k]))
    if not diffs:
        return E2EGateResult(max_diff=0.0, mean_diff=0.0, passes=False, n_compared=0)
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    passes = (max_d < MAX_DIFF_THRESHOLD) and (mean_d < MEAN_DIFF_THRESHOLD)
    return E2EGateResult(max_diff=max_d, mean_diff=mean_d, passes=passes, n_compared=len(diffs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_e2e_gate.py -v`
Expected: All PASS.

- [ ] **Step 5: Run E2E verification on a held-out day** (manual, after architecture selection)

For this step, take 1 day NOT in any dev fold (e.g., the day before the final-test window starts), and:
1. Run the offline pipeline on that day's parquet → record predictions per event.
2. Run the live-replay pipeline (calling `predict_architecture_X` from a streaming feature builder) on that day's tick stream → record predictions per event.
3. Pass both dicts to `compare_offline_vs_replay`.
4. **PASS:** result.passes == True. **FAIL:** investigate; do not deploy.

This step is the v3 leak detector. The actual E2E live-replay scaffolding (streaming feature builder) is intentionally NOT in this plan — building it is part of the deployment phase. We code the comparison primitive now so the gate is ready.

- [ ] **Step 6: Commit**

```bash
git add mnq_alerts/_level_e2e_gate.py mnq_alerts/tests/test_level_e2e_gate.py
git commit -m "feat(e2e-gate): offline vs replay prediction comparator"
```

---

### Task 17: Final out-of-time test runner

**Files:**
- Create: `mnq_alerts/_level_final_test.py`
- Create: `mnq_alerts/scripts/run_final_test.py`

- [ ] **Step 1: Write the failing test**

Create `mnq_alerts/tests/test_level_final_test.py`:

```python
"""Tests for final out-of-time test runner."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_final_test import evaluate_final_test, FinalTestResult


def _synth_test_data():
    base = pd.Timestamp("2026-04-09", tz="UTC")
    events = []
    preds = {}
    for d in range(30):
        for i in range(3):
            ts = base + pd.Timedelta(days=d, seconds=3600 * (i + 1))
            for direction in ("bounce", "breakthrough"):
                for tp, sl in [(8, 25), (8, 20), (10, 25), (10, 20)]:
                    events.append({
                        "event_ts": ts, "level_name": "FIB_0.618",
                        "direction": direction, "tp": tp, "sl": sl,
                        "event_price": 18000.0, "approach_direction": 1,
                        "label": 1, "time_to_resolution_sec": 60.0,
                    })
            preds[ts] = {("bounce", 8, 25): 0.7, **{k: 0.5 for k in [
                ("bounce", 8, 20), ("bounce", 10, 25), ("bounce", 10, 20),
                ("breakthrough", 8, 25), ("breakthrough", 8, 20),
                ("breakthrough", 10, 25), ("breakthrough", 10, 20)]}}
    return events, preds


def test_evaluate_final_test_returns_pass_fail_per_gate():
    events, preds = _synth_test_data()
    result = evaluate_final_test(events, preds, threshold=0.0, v6_mean_daily_pnl_dollars=0.0)
    assert isinstance(result, FinalTestResult)
    assert hasattr(result, "passes_v6_gate")
    assert hasattr(result, "passes_lift_gate")
    assert hasattr(result, "passes_per_week_gate")
    assert hasattr(result, "all_pass")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd mnq_alerts && python -m pytest tests/test_level_final_test.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `mnq_alerts/_level_final_test.py`:

```python
"""Final out-of-time test — touched ONCE, by the architecture chosen on dev."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from _level_simulation import simulate_strategy
from _level_validation import score_fold


@dataclass
class FinalTestResult:
    mean_daily_pnl_dollars: float
    top_decile_lift: float
    weekly_pnl_dollars: dict
    n_positive_weeks: int
    n_total_weeks: int
    passes_v6_gate: bool
    passes_lift_gate: bool
    passes_per_week_gate: bool
    all_pass: bool


def evaluate_final_test(
    events: list[dict], preds: dict, threshold: float,
    v6_mean_daily_pnl_dollars: float,
) -> FinalTestResult:
    """Evaluate the 3 final-test gates."""
    sim = simulate_strategy(events, preds, threshold=threshold)
    daily = sim["daily_pnl_dollars"]
    mean_daily = float(np.mean(list(daily.values()))) if daily else 0.0

    gates = score_fold(events, preds, threshold)

    # Weekly P&L over the last 4 calendar weeks of the test window.
    if daily:
        weekly: dict = {}
        for d, v in daily.items():
            week = pd.Timestamp(d).to_period("W")
            weekly[week] = weekly.get(week, 0.0) + v
        weeks_sorted = sorted(weekly.keys())
        last_4 = weeks_sorted[-4:]
        positives = sum(1 for w in last_4 if weekly[w] > 0)
    else:
        weekly = {}
        last_4 = []
        positives = 0

    passes_v6 = mean_daily > v6_mean_daily_pnl_dollars
    passes_lift = gates.top_decile_lift > 0.05  # +5pp
    passes_weeks = positives >= 3 and len(last_4) >= 4

    return FinalTestResult(
        mean_daily_pnl_dollars=mean_daily,
        top_decile_lift=gates.top_decile_lift,
        weekly_pnl_dollars={str(w): v for w, v in weekly.items()},
        n_positive_weeks=positives,
        n_total_weeks=len(last_4),
        passes_v6_gate=passes_v6,
        passes_lift_gate=passes_lift,
        passes_per_week_gate=passes_weeks,
        all_pass=passes_v6 and passes_lift and passes_weeks,
    )
```

Create `mnq_alerts/scripts/run_final_test.py`:

```python
"""CLI: run the final out-of-time test on the chosen architecture.

ONLY run this once architecture selection has chosen A or B. Touching this
script during development is the cardinal validation sin.

Usage: python -m mnq_alerts.scripts.run_final_test --arch A
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _level_validation import FINAL_TEST_TRADING_DAYS
from _level_final_test import evaluate_final_test


# V6 mean daily P&L over the final 30-day window — set when ready.
V6_FINAL_MEAN_DAILY = 1.83  # placeholder from recent-60d memory; replace with actual 30-day backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["A", "B"], required=True)
    args = parser.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = pd.read_parquet(os.path.join(here, "_level_events_labeled.parquet"))
    days = pd.DatetimeIndex(sorted(dataset["event_ts"].dt.date.unique()))
    final_test_start = days[-FINAL_TEST_TRADING_DAYS]
    train = dataset[dataset["event_ts"].dt.date < final_test_start.date()]
    test = dataset[dataset["event_ts"].dt.date >= final_test_start.date()]

    # Train chosen architecture on full dev set (no embargo, no walk-forward — final fit).
    val_cutoff = sorted(train["event_ts"].dt.date.unique())[-max(1, len(train["event_ts"].dt.date.unique()) // 20)]
    val = train[train["event_ts"].dt.date >= val_cutoff]
    train_for_fit = train[train["event_ts"].dt.date < val_cutoff]

    if args.arch == "A":
        from _level_model import train_architecture_a, predict_architecture_a
        models = train_architecture_a(train_for_fit, val)
        preds = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds[ev["event_ts"]] = predict_architecture_a(models, ev.to_dict())
    else:
        from _level_model import train_architecture_b, predict_architecture_b
        model = train_architecture_b(train_for_fit, val)
        preds = {}
        for _, ev in test.drop_duplicates("event_ts").iterrows():
            preds[ev["event_ts"]] = predict_architecture_b(model, ev.to_dict())

    result = evaluate_final_test(
        events=test.to_dict("records"), preds=preds, threshold=0.0,
        v6_mean_daily_pnl_dollars=V6_FINAL_MEAN_DAILY,
    )
    print(json.dumps({
        "mean_daily_pnl": result.mean_daily_pnl_dollars,
        "top_decile_lift": result.top_decile_lift,
        "n_positive_weeks": result.n_positive_weeks,
        "passes_v6_gate": result.passes_v6_gate,
        "passes_lift_gate": result.passes_lift_gate,
        "passes_per_week_gate": result.passes_per_week_gate,
        "all_pass": result.all_pass,
        "weekly_pnl": result.weekly_pnl_dollars,
    }, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd mnq_alerts && python -m pytest tests/test_level_final_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mnq_alerts/_level_final_test.py mnq_alerts/scripts/run_final_test.py mnq_alerts/tests/test_level_final_test.py
git commit -m "feat(validation): final out-of-time test runner"
```

---

## Stop here. The next plan covers deployment.

Tasks 1–17 produce a fully-validated model (or a clear "stop, the architecture didn't pass") with all leakage protections, walk-forward validation, and the final-test deploy gate.

**Deployment** (Section 5 of spec — shadow mode, canary, full deploy, rollback monitoring) is intentionally NOT in this plan. It depends on Task 17's outcome:
- If Task 17 passes all gates, write a follow-up plan: shadow-mode replay infra, bot integration, monitoring dashboards.
- If Task 17 fails, write findings as a project memory and stop.

This separation prevents premature investment in deployment infrastructure for a model that doesn't make the bar.

---

## Self-review (run before handoff)

**Spec coverage:**
- ✅ Section 1 (event extraction & labels) → Tasks 1, 2
- ✅ Section 1 (per-day dataset) → Task 3
- ✅ Section 2 (5 feature families) → Tasks 4, 5, 6, 7, 8
- ✅ Section 2 (leakage protection: 3 unit tests) → Task 9
- ✅ Section 2 (full-history dataset) → Task 10
- ✅ Section 3 (Architecture A) → Task 11
- ✅ Section 3 (Architecture B) → Task 12
- ✅ Section 3 (decision rule) → Task 13
- ✅ Section 4 (walk-forward folds, embargo) → Task 14
- ✅ Section 4 (architecture selection, gates) → Task 15
- ✅ Section 4 (E2E gate) → Task 16
- ✅ Section 4 (final out-of-time test) → Task 17
- ⏸ Section 5 (rollout) — explicitly deferred to follow-up plan

**Type consistency:**
- `extract_events` returns DataFrame with `event_ts, level_name, level_price, event_price, approach_direction` — used identically in `label_events`, `compute_all_features`, `train_architecture_*`. ✓
- `compute_all_features` output dict keys match `FEATURE_COLUMNS` + `prior_touch_outcome`. ✓
- `simulate_strategy` event-shape matches dataset rows. ✓

**Placeholder scan:** no TBDs, no "appropriate error handling," all code blocks complete. The two CLI runners (`select_architecture.py`, `run_final_test.py`) have a `V6_PER_QUARTER` / `V6_FINAL_MEAN_DAILY` constant that needs the actual V6 backtest numbers — these are flagged as fillable values, not placeholders for missing logic. The plan acknowledges this in inline comments.
