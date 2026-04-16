"""
targeted_backtest.py — Focused backtests on specific weaknesses.

Tests:
  1. Fast approach momentum — do alerts where price is plummeting/surging
     toward the level underperform?
  2. VWAP exit threshold — is 20 pts too wide for a drifting level?
     Should VWAP use current price vs current VWAP instead of locked reference?
  3. Exit threshold sensitivity — would a tighter exit let the zone reset
     faster and catch more opportunities?
  4. Direction-flip speed — after price crosses a level, how quickly can
     the system alert in the new direction?

Usage:
    python targeted_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))
from config import DATABENTO_API_KEY, IB_END_HOUR, IB_END_MIN

ET = pytz.timezone("America/New_York")
PT = pytz.timezone("America/Los_Angeles")

MARKET_OPEN = datetime.time(9, 30)
IB_END = datetime.time(IB_END_HOUR, IB_END_MIN)
MARKET_CLOSE = datetime.time(16, 0)

ALERT_THRESHOLD = 7.0
EXIT_THRESHOLD = 20.0
HIT_THRESHOLD = 1.0
TARGET_POINTS = 8.0
STOP_POINTS = 20.0
WINDOW_SECS = 15 * 60

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


@dataclass
class Alert:
    date: datetime.date
    alert_time: datetime.datetime
    level: str
    line_price: float
    entry_price: float
    direction: str
    level_test_count: int = 1
    approach_speed: float = 0.0  # pts/min price moved toward level in last 3 min
    outcome: str = "inconclusive"


class ZoneState:
    def __init__(self, name: str, price: float, exit_threshold: float = EXIT_THRESHOLD):
        self.name = name
        self.price = price
        self.in_zone = False
        self.ref: float | None = None
        self.entry_count = 0
        self.exit_threshold = exit_threshold

    def update(
        self, current_price: float, new_level_price: float | None = None
    ) -> bool:
        if new_level_price is not None:
            self.price = new_level_price
        if self.in_zone:
            if abs(current_price - self.ref) > self.exit_threshold:
                self.in_zone = False
                self.ref = None
            return False
        if abs(current_price - self.price) <= ALERT_THRESHOLD:
            self.in_zone = True
            self.ref = self.price
            self.entry_count += 1
            return True
        return False


class VWAPZoneState(ZoneState):
    """VWAP-specific zone that checks exit against current VWAP, not locked reference."""

    def update(
        self, current_price: float, new_level_price: float | None = None
    ) -> bool:
        if new_level_price is not None:
            self.price = new_level_price
        if self.in_zone:
            # Key difference: exit check uses current VWAP, not locked reference.
            if abs(current_price - self.price) > self.exit_threshold:
                self.in_zone = False
                self.ref = None
            return False
        if abs(current_price - self.price) <= ALERT_THRESHOLD:
            self.in_zone = True
            self.ref = self.price
            self.entry_count += 1
            return True
        return False


def load_cached_days() -> list[datetime.date]:
    """Find all cached parquet files."""
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet"))
    dates = []
    for f in files:
        try:
            d = datetime.date.fromisoformat(
                f.replace("MNQ_", "").replace(".parquet", "")
            )
            dates.append(d)
        except ValueError:
            pass
    return dates


def load_day(date: datetime.date) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"MNQ_{date}.parquet")
    return pd.read_parquet(path)


def evaluate_outcome_np(
    alert_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
) -> str:
    """Evaluate outcome using pre-extracted numpy arrays. Much faster than pandas."""
    alert_ns = ts_ns[alert_idx]
    window_ns = np.int64(WINDOW_SECS * 1_000_000_000)

    # Find hit: price within HIT_THRESHOLD of line_price within WINDOW_SECS
    start = alert_idx + 1
    end_ns = alert_ns + window_ns
    # Scan forward for hit
    hit_idx = -1
    for i in range(start, len(prices)):
        if ts_ns[i] > end_ns:
            break
        if abs(prices[i] - line_price) <= HIT_THRESHOLD:
            hit_idx = i
            break
    if hit_idx < 0:
        return "inconclusive"

    # Evaluate target/stop from hit point
    eval_end_ns = ts_ns[hit_idx] + window_ns
    target_idx = -1
    stop_idx = -1
    if direction == "up":
        target_price = line_price + TARGET_POINTS
        stop_price = line_price - STOP_POINTS
        for i in range(hit_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            if target_idx < 0 and prices[i] >= target_price:
                target_idx = i
            if stop_idx < 0 and prices[i] <= stop_price:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break
    else:
        target_price = line_price - TARGET_POINTS
        stop_price = line_price + STOP_POINTS
        for i in range(hit_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            if target_idx < 0 and prices[i] <= target_price:
                target_idx = i
            if stop_idx < 0 and prices[i] >= stop_price:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break

    if target_idx >= 0 and stop_idx >= 0:
        return "correct" if target_idx <= stop_idx else "incorrect"
    elif target_idx >= 0:
        return "correct"
    return "incorrect"


def evaluate_outcome_from_entry_np(
    alert_idx: int,
    entry_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
) -> str:
    """Evaluate outcome from ENTRY PRICE, not line price. No hit requirement.

    Used for wide-threshold tests where the bounce happens away from the line.
    Target/stop are measured from where the alert fired.
    """
    alert_ns = ts_ns[alert_idx]
    window_ns = np.int64(WINDOW_SECS * 1_000_000_000)
    eval_end_ns = alert_ns + window_ns

    target_idx = -1
    stop_idx = -1

    if direction == "up":
        target_price = entry_price + TARGET_POINTS
        stop_price = entry_price - STOP_POINTS
        for i in range(alert_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            if target_idx < 0 and prices[i] >= target_price:
                target_idx = i
            if stop_idx < 0 and prices[i] <= stop_price:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break
    else:
        target_price = entry_price - TARGET_POINTS
        stop_price = entry_price + STOP_POINTS
        for i in range(alert_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            if target_idx < 0 and prices[i] <= target_price:
                target_idx = i
            if stop_idx < 0 and prices[i] >= stop_price:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break

    if target_idx >= 0 and stop_idx >= 0:
        return "correct" if target_idx <= stop_idx else "incorrect"
    elif target_idx >= 0:
        return "correct"
    elif stop_idx >= 0:
        return "incorrect"
    return "inconclusive"


def compute_approach_speed(
    df: pd.DataFrame, alert_ts: pd.Timestamp, line_price: float, direction: str
) -> float:
    """Compute how fast price moved toward the level in the 3 min before alert.

    Returns pts/min. Positive = moving toward level, negative = moving away.
    """
    window_start = alert_ts - pd.Timedelta(minutes=3)
    window = df[(df.index >= window_start) & (df.index <= alert_ts)]
    if len(window) < 2:
        return 0.0
    price_change = float(window["price"].iloc[-1] - window["price"].iloc[0])
    minutes = max((window.index[-1] - window.index[0]).total_seconds() / 60, 0.01)

    # Positive = toward line. For "up" (price above line), approaching = price falling.
    if direction == "up":
        return -price_change / minutes
    else:
        return price_change / minutes


def _run_zone_numpy(
    prices: np.ndarray,
    level_prices: np.ndarray,
    alert_threshold: float,
    exit_threshold: float,
    use_current_exit: bool = False,
) -> list[tuple[int, int, float]]:
    """Run zone state machine on numpy arrays. Returns list of (index, entry_count, ref_price).

    This replaces the Python-level ZoneState.update() loop with tight numpy array access.
    ~10-20x faster than itertuples + object attribute access.
    """
    n = len(prices)
    entries: list[tuple[int, int, float]] = []
    in_zone = False
    ref = 0.0
    entry_count = 0

    for i in range(n):
        p = prices[i]
        lp = level_prices[i]
        if in_zone:
            check_price = lp if use_current_exit else ref
            if abs(p - check_price) > exit_threshold:
                in_zone = False
        else:
            if abs(p - lp) <= alert_threshold:
                in_zone = True
                ref = lp
                entry_count += 1
                entries.append((i, entry_count, ref))

    return entries


@dataclass
class DayCache:
    """Pre-extracted numpy arrays for a single day to avoid repeated pandas ops."""

    date: datetime.date
    ibh: float
    ibl: float
    fib_lo: float
    fib_hi: float
    post_ib_prices: np.ndarray
    post_ib_vwaps: np.ndarray
    post_ib_timestamps: pd.DatetimeIndex
    post_ib_start_idx: int  # offset of first post-IB row in full arrays
    full_prices: np.ndarray  # full day prices for outcome evaluation
    full_ts_ns: np.ndarray  # full day timestamps as int64 nanoseconds
    full_df: pd.DataFrame  # kept for approach speed calculation only


def preprocess_day(df: pd.DataFrame, date: datetime.date) -> DayCache | None:
    """Extract numpy arrays from a day's DataFrame once."""
    if df.empty:
        return None
    ib = df[(df.index.time >= MARKET_OPEN) & (df.index.time < IB_END)]
    if ib.empty:
        return None
    ibh = float(ib["price"].max())
    ibl = float(ib["price"].min())

    cum_pv = (df["price"] * df["size"]).cumsum()
    cum_vol = df["size"].cumsum()
    vwap_arr = (cum_pv / cum_vol).values

    post_ib_mask = df.index.time >= IB_END
    post_ib = df[post_ib_mask]
    if post_ib.empty:
        return None

    # Find the index offset for the first post-IB row
    post_ib_start = np.argmax(post_ib_mask)

    ib_range = ibh - ibl
    return DayCache(
        date=date,
        ibh=ibh,
        ibl=ibl,
        fib_lo=ibl - 0.272 * ib_range,
        fib_hi=ibh + 0.272 * ib_range,
        post_ib_prices=post_ib["price"].values.astype(np.float64),
        post_ib_vwaps=vwap_arr[post_ib_mask].astype(np.float64),
        post_ib_timestamps=post_ib.index,
        post_ib_start_idx=int(post_ib_start),
        full_prices=df["price"].values.astype(np.float64),
        full_ts_ns=df.index.asi8,
        full_df=df,
    )


def simulate_day(
    dc: DayCache,
    exit_threshold: float = EXIT_THRESHOLD,
    vwap_exit_threshold: float | None = None,
    use_vwap_current_exit: bool = False,
    calc_approach_speed: bool = False,
    levels_filter: set[str] | None = None,
    alert_threshold: float = ALERT_THRESHOLD,
    eval_from_entry: bool = False,
) -> list[Alert]:
    """Simulate one day with configurable thresholds.

    Set calc_approach_speed=True only for Test 1 (adds ~1ms per alert of overhead).
    Set levels_filter to only simulate specific levels (e.g. {"VWAP"}).
    Set alert_threshold to widen/narrow the zone entry distance.
    Set eval_from_entry=True to evaluate target/stop from entry price instead of line price.
    """
    prices = dc.post_ib_prices
    n = len(prices)

    vwap_et = vwap_exit_threshold if vwap_exit_threshold is not None else exit_threshold

    all_levels = [
        ("IBH", np.full(n, dc.ibh), exit_threshold, False),
        ("IBL", np.full(n, dc.ibl), exit_threshold, False),
        ("VWAP", dc.post_ib_vwaps, vwap_et, use_vwap_current_exit),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), exit_threshold, False),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), exit_threshold, False),
    ]
    levels_config = [
        lc for lc in all_levels if levels_filter is None or lc[0] in levels_filter
    ]

    alerts: list[Alert] = []

    for level_name, level_arr, et, use_current in levels_config:
        entries = _run_zone_numpy(prices, level_arr, alert_threshold, et, use_current)
        for idx, entry_count, ref_price in entries:
            ts = dc.post_ib_timestamps[idx]
            price = prices[idx]
            direction = "up" if price > ref_price else "down"
            speed = 0.0
            if calc_approach_speed:
                speed = compute_approach_speed(dc.full_df, ts, ref_price, direction)
            # Evaluate outcome using numpy (avoid pandas per-alert)
            full_idx = dc.post_ib_start_idx + idx
            if eval_from_entry:
                outcome = evaluate_outcome_from_entry_np(
                    full_idx, price, direction, dc.full_ts_ns, dc.full_prices
                )
            else:
                outcome = evaluate_outcome_np(
                    full_idx, ref_price, direction, dc.full_ts_ns, dc.full_prices
                )
            alerts.append(
                Alert(
                    date=dc.date,
                    alert_time=ts.to_pydatetime(warn=False),
                    level=level_name,
                    line_price=ref_price,
                    entry_price=price,
                    direction=direction,
                    level_test_count=entry_count,
                    approach_speed=speed,
                    outcome=outcome,
                )
            )

    return alerts


def win_rate_table(groups: list[tuple[str, list[Alert]]]) -> None:
    print(f"  {'Group':<45}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}")
    print(f"  {'-'*45}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}")
    for label, alerts in groups:
        w = sum(1 for a in alerts if a.outcome == "correct")
        l = sum(1 for a in alerts if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        warn = "  ⚠ n<30" if 0 < t < 30 else ""
        print(f"  {label:<45}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}{warn}")


def main() -> None:
    days = load_cached_days()
    print(f"{'═' * 75}")
    print(f"  TARGETED BACKTESTS  |  {days[0]} → {days[-1]}  ({len(days)} days)")
    print(f"{'═' * 75}")

    # Load all days and preprocess into DayCache
    day_caches: dict[datetime.date, DayCache] = {}
    for i, date in enumerate(days):
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except Exception as e:
            print(f"  Error loading {date}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(days)} days...", flush=True)
    print(f"  All {len(day_caches)} days loaded and preprocessed.", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1: APPROACH SPEED ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 1: APPROACH SPEED (pts/min toward level in 3 min before alert)")
    print(f"  Does price plummeting toward a level predict worse outcomes?")
    print(f"{'═' * 75}")

    all_alerts: list[Alert] = []
    for i, date in enumerate(days):
        dc = day_caches.get(date)
        if dc is None:
            continue
        all_alerts.extend(simulate_day(dc, calc_approach_speed=True))
        if (i + 1) % 50 == 0:
            print(f"  [Test 1] {i + 1}/{len(days)} days...", flush=True)

    decided = [a for a in all_alerts if a.outcome in ("correct", "incorrect")]
    print(f"\n  Total alerts: {len(all_alerts)}, decided: {len(decided)}")

    # Speed quartiles
    speeds = sorted(a.approach_speed for a in decided)
    q25 = speeds[len(speeds) // 4]
    q50 = speeds[len(speeds) // 2]
    q75 = speeds[3 * len(speeds) // 4]
    print(
        f"  Approach speed quartiles: Q25={q25:.1f}, Q50={q50:.1f}, Q75={q75:.1f} pts/min"
    )

    print(f"\n  Overall approach speed vs win rate:")
    win_rate_table(
        [
            (
                f"Very fast approach (>{q75:.0f} pts/min)",
                [a for a in decided if a.approach_speed > q75],
            ),
            (
                f"Fast approach ({q50:.0f}–{q75:.0f} pts/min)",
                [a for a in decided if q50 < a.approach_speed <= q75],
            ),
            (
                f"Moderate approach ({q25:.0f}–{q50:.0f} pts/min)",
                [a for a in decided if q25 < a.approach_speed <= q50],
            ),
            (
                f"Slow/retreating (<{q25:.0f} pts/min)",
                [a for a in decided if a.approach_speed <= q25],
            ),
        ]
    )

    # Fine-grained speed sweep
    print(f"\n  Approach speed sweep (filter out alerts faster than threshold):")
    print(
        f"  {'Max speed':>12}  {'W':>5}  {'L':>5}  {'Total':>7}  {'Win%':>6}  {'Removed':>8}"
    )
    print(f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*8}")
    for max_speed in [999, 50, 40, 30, 25, 20, 15, 10, 5]:
        subset = [a for a in decided if a.approach_speed <= max_speed]
        w = sum(1 for a in subset if a.outcome == "correct")
        l = sum(1 for a in subset if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        removed = len(decided) - len(subset)
        label = f"≤{max_speed} pts/min" if max_speed < 999 else "No filter"
        print(f"  {label:>12}  {w:>5}  {l:>5}  {t:>7}  {wr:>5.1%}  {removed:>8}")

    # Speed × level cross-tab
    print(f"\n  Approach speed × level (fast = top 25%):")
    for lvl in ["IBL", "IBH", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        lvl_alerts = [a for a in decided if a.level == lvl]
        if not lvl_alerts:
            continue
        lvl_median = sorted(a.approach_speed for a in lvl_alerts)[len(lvl_alerts) // 2]
        win_rate_table(
            [
                (
                    f"{lvl} fast approach (>{lvl_median:.0f} pts/min)",
                    [a for a in lvl_alerts if a.approach_speed > lvl_median],
                ),
                (
                    f"{lvl} slow approach (≤{lvl_median:.0f} pts/min)",
                    [a for a in lvl_alerts if a.approach_speed <= lvl_median],
                ),
            ]
        )

    # Speed × direction
    print(f"\n  Approach speed × direction (fast = top 25%):")
    win_rate_table(
        [
            (
                "Fast approach + BUY",
                [a for a in decided if a.approach_speed > q75 and a.direction == "up"],
            ),
            (
                "Fast approach + SELL",
                [
                    a
                    for a in decided
                    if a.approach_speed > q75 and a.direction == "down"
                ],
            ),
            (
                "Slow approach + BUY",
                [a for a in decided if a.approach_speed <= q25 and a.direction == "up"],
            ),
            (
                "Slow approach + SELL",
                [
                    a
                    for a in decided
                    if a.approach_speed <= q25 and a.direction == "down"
                ],
            ),
        ]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: VWAP EXIT THRESHOLD
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 2: VWAP EXIT THRESHOLD")
    print(f"  Current: exit when price moves 20 pts from locked reference.")
    print(f"  Problem: VWAP drifts, so zone stays active and misses re-entries.")
    print(f"{'═' * 75}")

    # Count how many VWAP alerts fire with different exit thresholds
    vwap_exits = [10, 12, 15, 20, 25, 30]
    vwap_results = []
    for vi, vwap_exit in enumerate(vwap_exits):
        print(f"  [Test 2] exit={vwap_exit} ({vi+1}/{len(vwap_exits)})...", flush=True)
        vwap_alerts = []
        for date in days:
            dc = day_caches.get(date)
            if dc is None:
                continue
            day_alerts = simulate_day(
                dc, vwap_exit_threshold=vwap_exit, levels_filter={"VWAP"}
            )
            vwap_alerts.extend(day_alerts)

        decided_vwap = [a for a in vwap_alerts if a.outcome in ("correct", "incorrect")]
        w = sum(1 for a in decided_vwap if a.outcome == "correct")
        l = sum(1 for a in decided_vwap if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        total = len(vwap_alerts)
        per_day = total / len(days) if days else 0
        ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS if t > 0 else 0.0
        marker = " ← current" if vwap_exit == 20 else ""
        vwap_results.append((vwap_exit, w, l, t, wr, total, per_day, ev))

    print(f"\n  VWAP exit threshold sweep (other levels stay at 20):")
    print(
        f"  {'Exit pts':>9}  {'W':>5}  {'L':>5}  {'Decided':>8}  {'Win%':>6}  {'Total':>6}  {'/day':>5}  {'EV':>6}"
    )
    print(f"  {'-'*9}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}")
    for vwap_exit, w, l, t, wr, total, per_day, ev in vwap_results:
        marker = " ← current" if vwap_exit == 20 else ""
        print(
            f"  {vwap_exit:>9}  {w:>5}  {l:>5}  {t:>8}  {wr:>5.1%}  {total:>6}  {per_day:>5.1f}  {ev:>+5.1f}{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3: VWAP EXIT AGAINST CURRENT VWAP (not locked reference)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 3: VWAP EXIT — CURRENT VWAP vs LOCKED REFERENCE")
    print(f"  Instead of exiting when price moves 20 pts from the VWAP value at")
    print(f"  zone entry, exit when price moves N pts from CURRENT VWAP.")
    print(f"{'═' * 75}")

    cv_exits = [10, 12, 15, 20]
    current_vwap_results = []
    for vi, vwap_exit in enumerate(cv_exits):
        print(f"  [Test 3] exit={vwap_exit} ({vi+1}/{len(cv_exits)})...", flush=True)
        vwap_alerts = []
        for date in days:
            dc = day_caches.get(date)
            if dc is None:
                continue
            day_alerts = simulate_day(
                dc,
                vwap_exit_threshold=vwap_exit,
                use_vwap_current_exit=True,
                levels_filter={"VWAP"},
            )
            vwap_alerts.extend(day_alerts)

        decided_vwap = [a for a in vwap_alerts if a.outcome in ("correct", "incorrect")]
        w = sum(1 for a in decided_vwap if a.outcome == "correct")
        l = sum(1 for a in decided_vwap if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        total = len(vwap_alerts)
        per_day = total / len(days) if days else 0
        ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS if t > 0 else 0.0
        current_vwap_results.append((vwap_exit, w, l, t, wr, total, per_day, ev))

    print(
        f"\n  VWAP exit vs CURRENT VWAP (zone resets when price leaves current VWAP):"
    )
    print(
        f"  {'Exit pts':>9}  {'W':>5}  {'L':>5}  {'Decided':>8}  {'Win%':>6}  {'Total':>6}  {'/day':>5}  {'EV':>6}"
    )
    print(f"  {'-'*9}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}")
    for vwap_exit, w, l, t, wr, total, per_day, ev in current_vwap_results:
        print(
            f"  {vwap_exit:>9}  {w:>5}  {l:>5}  {t:>8}  {wr:>5.1%}  {total:>6}  {per_day:>5.1f}  {ev:>+5.1f}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 4: GLOBAL EXIT THRESHOLD SENSITIVITY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 4: GLOBAL EXIT THRESHOLD (all levels)")
    print(f"  Tighter exit = zone resets faster = more alerts but catches re-entries.")
    print(f"{'═' * 75}")

    exit_vals = [7, 10, 12, 15, 20, 25, 30]
    exit_results = []
    for ei, exit_t in enumerate(exit_vals):
        print(f"  [Test 4] exit={exit_t} ({ei+1}/{len(exit_vals)})...", flush=True)
        exit_alerts = []
        for date in days:
            dc = day_caches.get(date)
            if dc is None:
                continue
            exit_alerts.extend(simulate_day(dc, exit_threshold=exit_t))

        decided_exit = [a for a in exit_alerts if a.outcome in ("correct", "incorrect")]
        w = sum(1 for a in decided_exit if a.outcome == "correct")
        l = sum(1 for a in decided_exit if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        total = len(exit_alerts)
        per_day = total / len(days) if days else 0
        ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS if t > 0 else 0.0
        exit_results.append((exit_t, w, l, t, wr, total, per_day, ev))

    print(
        f"\n  {'Exit pts':>9}  {'W':>5}  {'L':>5}  {'Decided':>8}  {'Win%':>6}  {'Total':>6}  {'/day':>5}  {'EV':>6}"
    )
    print(f"  {'-'*9}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}")
    for exit_t, w, l, t, wr, total, per_day, ev in exit_results:
        marker = " ← current" if exit_t == 20 else ""
        print(
            f"  {exit_t:>9}  {w:>5}  {l:>5}  {t:>8}  {wr:>5.1%}  {total:>6}  {per_day:>5.1f}  {ev:>+5.1f}{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 5: MISSED OPPORTUNITIES — VWAP alerts that WOULD have fired
    # with a tighter exit threshold but didn't with 20 pts
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 5: MISSED VWAP OPPORTUNITIES")
    print(f"  Alerts that fire with exit=10 but NOT with exit=20 (missed re-entries)")
    print(f"{'═' * 75}")

    missed_correct = 0
    missed_incorrect = 0
    missed_inconclusive = 0
    for date in days:
        dc = day_caches.get(date)
        if dc is None:
            continue
        tight_alerts = simulate_day(dc, vwap_exit_threshold=10, levels_filter={"VWAP"})
        wide_alerts = simulate_day(dc, vwap_exit_threshold=20, levels_filter={"VWAP"})

        tight_vwap = {(a.alert_time, a.direction) for a in tight_alerts}
        wide_vwap = {(a.alert_time, a.direction) for a in wide_alerts}
        missed_keys = tight_vwap - wide_vwap
        missed = [a for a in tight_alerts if (a.alert_time, a.direction) in missed_keys]
        for a in missed:
            if a.outcome == "correct":
                missed_correct += 1
            elif a.outcome == "incorrect":
                missed_incorrect += 1
            else:
                missed_inconclusive += 1

    total_missed = missed_correct + missed_incorrect + missed_inconclusive
    decided_missed = missed_correct + missed_incorrect
    wr_missed = missed_correct / decided_missed if decided_missed > 0 else 0.0
    print(f"\n  Missed VWAP re-entries (would fire with exit=10, blocked by exit=20):")
    print(f"    Total missed   : {total_missed}")
    print(f"    Correct        : {missed_correct}")
    print(f"    Incorrect      : {missed_incorrect}")
    print(f"    Inconclusive   : {missed_inconclusive}")
    print(f"    Win rate       : {wr_missed:.1%} ({decided_missed} decided)")
    print(f"    Per day        : {total_missed / len(days):.1f}")

    # Same for current-VWAP exit
    missed_correct_cv = 0
    missed_incorrect_cv = 0
    missed_inconclusive_cv = 0
    for date in days:
        dc = day_caches.get(date)
        if dc is None:
            continue
        tight_alerts = simulate_day(
            dc,
            vwap_exit_threshold=10,
            use_vwap_current_exit=True,
            levels_filter={"VWAP"},
        )
        wide_alerts = simulate_day(dc, vwap_exit_threshold=20, levels_filter={"VWAP"})

        tight_vwap = {(a.alert_time, a.direction) for a in tight_alerts}
        wide_vwap = {(a.alert_time, a.direction) for a in wide_alerts}
        missed_keys = tight_vwap - wide_vwap
        missed = [a for a in tight_alerts if (a.alert_time, a.direction) in missed_keys]
        for a in missed:
            if a.outcome == "correct":
                missed_correct_cv += 1
            elif a.outcome == "incorrect":
                missed_incorrect_cv += 1
            else:
                missed_inconclusive_cv += 1

    total_missed_cv = missed_correct_cv + missed_incorrect_cv + missed_inconclusive_cv
    decided_missed_cv = missed_correct_cv + missed_incorrect_cv
    wr_missed_cv = (
        missed_correct_cv / decided_missed_cv if decided_missed_cv > 0 else 0.0
    )
    print(f"\n  Missed VWAP re-entries (current-VWAP exit=10 vs locked-ref exit=20):")
    print(f"    Total missed   : {total_missed_cv}")
    print(f"    Correct        : {missed_correct_cv}")
    print(f"    Incorrect      : {missed_incorrect_cv}")
    print(f"    Inconclusive   : {missed_inconclusive_cv}")
    print(f"    Win rate       : {wr_missed_cv:.1%} ({decided_missed_cv} decided)")
    print(f"    Per day        : {total_missed_cv / len(days):.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 6: WIDE ENTRY THRESHOLD — BOUNCE AWAY FROM THE LINE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 6: WIDE ENTRY THRESHOLD (bounce away from the line)")
    print(f"  Hypothesis: price bounces 10-15 pts from levels, not at the line.")
    print(f"  Widen entry threshold from 7 to 20 and evaluate from entry price.")
    print(f"{'═' * 75}")

    entry_thresholds = [7, 10, 13, 15, 17, 20]

    # Part A: Entry threshold sweep with entry-price evaluation
    print(f"\n  Part A — Entry threshold sweep (eval from entry price):")
    print(
        f"  {'Entry':>6}  {'Exit':>5}  {'W':>5}  {'L':>5}  {'Decided':>8}  "
        f"{'Win%':>6}  {'Total':>6}  {'/day':>5}  {'EV':>6}"
    )
    print(
        f"  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*8}  "
        f"{'-'*6}  {'-'*6}  {'-'*5}  {'-'*6}"
    )
    for at in entry_thresholds:
        et = max(20, at + 10)
        alerts = []
        for date in days:
            dc = day_caches.get(date)
            if dc is None:
                continue
            alerts.extend(
                simulate_day(
                    dc, exit_threshold=et, alert_threshold=at, eval_from_entry=True
                )
            )
        decided = [a for a in alerts if a.outcome in ("correct", "incorrect")]
        w = sum(1 for a in decided if a.outcome == "correct")
        l = sum(1 for a in decided if a.outcome == "incorrect")
        t = w + l
        wr = w / t if t > 0 else 0.0
        total = len(alerts)
        per_day = total / len(days) if days else 0
        ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS if t > 0 else 0.0
        marker = " ← current" if at == 7 else ""
        print(
            f"  {at:>6}  {et:>5}  {w:>5}  {l:>5}  {t:>8}  "
            f"{wr:>5.1%}  {total:>6}  {per_day:>5.1f}  {ev:>+5.1f}{marker}"
        )
    print(flush=True)

    # Part B: Entry-price vs line-price evaluation comparison
    print(f"\n  Part B — Entry-price vs line-price evaluation:")
    print(
        f"  {'Entry':>6}  {'Eval from':<12}  {'W':>5}  {'L':>5}  "
        f"{'Decided':>8}  {'Win%':>6}  {'EV':>6}"
    )
    print(f"  {'-'*6}  {'-'*12}  {'-'*5}  {'-'*5}  " f"{'-'*8}  {'-'*6}  {'-'*6}")
    for at in entry_thresholds:
        et = max(20, at + 10)
        for eval_mode, eval_label in [(True, "entry price"), (False, "line price")]:
            alerts = []
            for date in days:
                dc = day_caches.get(date)
                if dc is None:
                    continue
                alerts.extend(
                    simulate_day(
                        dc,
                        exit_threshold=et,
                        alert_threshold=at,
                        eval_from_entry=eval_mode,
                    )
                )
            decided = [a for a in alerts if a.outcome in ("correct", "incorrect")]
            w = sum(1 for a in decided if a.outcome == "correct")
            l = sum(1 for a in decided if a.outcome == "incorrect")
            t = w + l
            wr = w / t if t > 0 else 0.0
            ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS if t > 0 else 0.0
            print(
                f"  {at:>6}  {eval_label:<12}  {w:>5}  {l:>5}  "
                f"{t:>8}  {wr:>5.1%}  {ev:>+5.1f}"
            )
        print(f"  {'':>6}  {'':>12}  {'':>5}  {'':>5}  {'':>8}  {'':>6}  {'':>6}")
    print(flush=True)

    # Part C: Per-level breakdown at thresholds 7 and 20
    print(f"\n  Part C — Per-level breakdown (eval from entry price):")
    for at in [7, 20]:
        et = max(20, at + 10)
        alerts = []
        for date in days:
            dc = day_caches.get(date)
            if dc is None:
                continue
            alerts.extend(
                simulate_day(
                    dc, exit_threshold=et, alert_threshold=at, eval_from_entry=True
                )
            )
        print(f"\n  Entry threshold = {at} pts (exit = {et}):")
        print(
            f"  {'Level':<22}  {'W':>5}  {'L':>5}  {'Decided':>8}  "
            f"{'Win%':>6}  {'Total':>6}  {'/day':>5}"
        )
        print(f"  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*8}  " f"{'-'*6}  {'-'*6}  {'-'*5}")
        for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
            lvl_alerts = [a for a in alerts if a.level == level]
            decided = [a for a in lvl_alerts if a.outcome in ("correct", "incorrect")]
            w = sum(1 for a in decided if a.outcome == "correct")
            l = sum(1 for a in decided if a.outcome == "incorrect")
            t = w + l
            wr = w / t if t > 0 else 0.0
            total = len(lvl_alerts)
            per_day = total / len(days) if days else 0
            print(
                f"  {level:<22}  {w:>5}  {l:>5}  {t:>8}  "
                f"{wr:>5.1%}  {total:>6}  {per_day:>5.1f}"
            )

    print(f"\n{'═' * 75}")
    print("  DONE")
    print(f"{'═' * 75}")


if __name__ == "__main__":
    main()
