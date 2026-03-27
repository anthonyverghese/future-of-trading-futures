"""
score_optimizer.py — Data-driven scoring weight validation and optimization.

Computes ALL 7 scoring factors from cached parquet data, validates current
weights against actual win rates, suggests data-driven weights, and tests
on held-out days to prevent overfitting.

Usage:
    python score_optimizer.py
"""

from __future__ import annotations

import datetime
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    load_cached_days,
    load_day,
    preprocess_day,
    simulate_day,
    ET,
)


@dataclass
class EnrichedAlert:
    """Alert with all 7 scoring factors computed from cached data."""

    date: datetime.date
    level: str
    direction: str
    entry_count: int
    outcome: str  # "correct" or "incorrect"
    entry_price: float
    line_price: float
    alert_time: datetime.datetime

    # Computed factors
    now_et: datetime.time | None = None
    tick_rate: float | None = None
    session_move_pts: float | None = None
    consecutive_wins: int = 0
    consecutive_losses: int = 0


def compute_tick_rate(
    df: pd.DataFrame, alert_ts: pd.Timestamp, window_minutes: int = 3
) -> float:
    """Count trades in N-minute window before alert, return trades/minute.

    Matches production main.py:324-330: tick_rate = len(tick_times) / 3.0
    where tick_times is trades in last 3 minutes.
    """
    window_start = alert_ts - pd.Timedelta(minutes=window_minutes)
    mask = (df.index >= window_start) & (df.index <= alert_ts)
    count = mask.sum()
    return count / window_minutes


def load_all_alerts(dates: list[datetime.date]) -> list[EnrichedAlert]:
    """Load all days, simulate, compute all 7 factors, return enriched alerts.

    Streak is tracked across days chronologically, matching production behavior.
    """
    all_alerts: list[EnrichedAlert] = []
    consecutive_wins = 0
    consecutive_losses = 0

    for i, date in enumerate(dates):
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is None:
                continue
        except Exception:
            continue

        day_alerts = simulate_day(dc)
        first_price = float(dc.post_ib_prices[0])

        # Sort by time within day (simulate_day returns per-level, not chronological)
        day_alerts.sort(key=lambda a: a.alert_time)

        for a in day_alerts:
            if a.outcome not in ("correct", "incorrect"):
                continue

            # Time of day in ET
            if hasattr(a.alert_time, "astimezone") and a.alert_time.tzinfo:
                now_et = a.alert_time.astimezone(
                    datetime.timezone(datetime.timedelta(hours=-4))
                ).time()
            else:
                now_et = None

            # Tick rate: trades in 3-min window / 3
            tick_rate = compute_tick_rate(dc.full_df, pd.Timestamp(a.alert_time))

            # Session move: current price - first post-IB price
            session_move = a.entry_price - first_price

            ea = EnrichedAlert(
                date=date,
                level=a.level,
                direction=a.direction,
                entry_count=a.level_test_count,
                outcome=a.outcome,
                entry_price=a.entry_price,
                line_price=a.line_price,
                alert_time=a.alert_time,
                now_et=now_et,
                tick_rate=tick_rate,
                session_move_pts=session_move,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
            )
            all_alerts.append(ea)

            # Update streak AFTER recording (matches production: streak reflects
            # state BEFORE this alert, updates AFTER)
            if a.outcome == "correct":
                consecutive_wins += 1
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                consecutive_wins = 0

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(dates)} days...", flush=True)

    return all_alerts


def wr_line(label: str, subset: list[EnrichedAlert], indent: str = "  ") -> None:
    if not subset:
        print(f"{indent}{label:<45} (no data)")
        return
    w = sum(1 for a in subset if a.outcome == "correct")
    n = len(subset)
    warn = " ⚠n<30" if n < 30 else ""
    print(f"{indent}{label:<45} {w:>5}W {n-w:>5}L {n:>7} {w/n*100:>6.1f}%{warn}")


def suggest_weight(bucket_wr: float, baseline_wr: float) -> int:
    """Map WR delta to integer weight. +/- 2.5% per point, capped at [-4, +4]."""
    delta = bucket_wr - baseline_wr
    weight = round(delta / 2.5)
    return max(-4, min(4, weight))


# ── Scoring function with configurable weights ──


@dataclass
class Weights:
    """All scoring weights in one place for easy tuning."""

    # Level quality
    level_fib_hi: int = 2
    level_ibl: int = 1
    level_fib_lo: int = 1
    level_vwap: int = -1
    level_ibh: int = -1

    # Direction x Level combos
    combo_fib_hi_up: int = 2
    combo_fib_lo_down: int = 1
    combo_ibl_down: int = 1
    combo_vwap_up: int = 1
    combo_ibh_up: int = -1
    combo_ibl_up: int = -1
    combo_fib_lo_up: int = -1
    combo_fib_hi_down: int = -1
    combo_vwap_down: int = -1

    # Time of day
    time_power_hour: int = 2

    # Tick rate
    tick_sweet_spot: int = 2
    tick_lo: float = 1750.0
    tick_hi: float = 2000.0

    # Test count
    test_1: int = -1
    test_2: int = 1
    test_3: int = -1
    test_5: int = 1

    # Session move
    move_sweet_green: int = 2  # (10, 20]
    move_sweet_red: int = 2  # (-20, -10]
    move_strong_red: int = 1  # <= -50
    move_near_zero_green: int = -3  # (0, 10]
    move_strong_green: int = 0  # > 50
    move_other: int = 0

    # Streak
    streak_win_bonus: int = 3  # 2+ wins
    streak_loss_penalty: int = -2  # 2+ losses


def score_alert(a: EnrichedAlert, w: Weights) -> int:
    """Score an alert using configurable weights."""
    total = 0

    # Level
    if a.level == "FIB_EXT_HI_1.272":
        total += w.level_fib_hi
    elif a.level == "IBL":
        total += w.level_ibl
    elif a.level == "FIB_EXT_LO_1.272":
        total += w.level_fib_lo
    elif a.level == "VWAP":
        total += w.level_vwap
    elif a.level == "IBH":
        total += w.level_ibh

    # Direction x Level
    if a.direction is not None:
        c = (a.level, a.direction)
        if c == ("FIB_EXT_HI_1.272", "up"):
            total += w.combo_fib_hi_up
        elif c == ("FIB_EXT_LO_1.272", "down"):
            total += w.combo_fib_lo_down
        elif c == ("IBL", "down"):
            total += w.combo_ibl_down
        elif c == ("VWAP", "up"):
            total += w.combo_vwap_up
        elif c == ("IBH", "up"):
            total += w.combo_ibh_up
        elif c == ("IBL", "up"):
            total += w.combo_ibl_up
        elif c == ("FIB_EXT_LO_1.272", "up"):
            total += w.combo_fib_lo_up
        elif c == ("FIB_EXT_HI_1.272", "down"):
            total += w.combo_fib_hi_down
        elif c == ("VWAP", "down"):
            total += w.combo_vwap_down

    # Time of day
    if a.now_et is not None:
        mins = a.now_et.hour * 60 + a.now_et.minute
        if mins >= 15 * 60:
            total += w.time_power_hour

    # Tick rate
    if a.tick_rate is not None:
        if w.tick_lo <= a.tick_rate < w.tick_hi:
            total += w.tick_sweet_spot

    # Test count
    if a.entry_count == 1:
        total += w.test_1
    elif a.entry_count == 2:
        total += w.test_2
    elif a.entry_count == 3:
        total += w.test_3
    elif a.entry_count == 5:
        total += w.test_5

    # Session move
    if a.session_move_pts is not None:
        m = a.session_move_pts
        if 10 < m <= 20:
            total += w.move_sweet_green
        elif -20 < m <= -10:
            total += w.move_sweet_red
        elif m <= -50:
            total += w.move_strong_red
        elif m > 50:
            total += w.move_strong_green
        elif 0 < m <= 10:
            total += w.move_near_zero_green
        else:
            total += w.move_other

    # Streak
    if a.consecutive_wins >= 2:
        total += w.streak_win_bonus
    elif a.consecutive_losses >= 2:
        total += w.streak_loss_penalty

    return total


def threshold_sweep(
    alerts: list[EnrichedAlert], w: Weights, num_days: int, label: str = ""
) -> None:
    """Score all alerts and show WR/volume at each threshold."""
    scored = [(a, score_alert(a, w)) for a in alerts]

    if label:
        print(f"\n  {label}")
    print(
        f"  {'Score >=':>8} {'W':>5} {'L':>5} {'Total':>7} {'WR%':>7} {'/day':>6} {'Target?':>8}"
    )
    print(f"  {'-'*50}")

    min_s = min(s for _, s in scored)
    max_s = max(s for _, s in scored)
    for threshold in range(min_s, max_s + 1):
        passing = [(a, s) for a, s in scored if s >= threshold]
        w_count = sum(1 for a, _ in passing if a.outcome == "correct")
        n = len(passing)
        if n == 0:
            continue
        wr = w_count / n * 100
        per_day = n / num_days
        target = "YES" if wr >= 80 and n >= 600 else ""
        marker = " <-- current" if threshold == 5 else ""
        print(
            f"  {threshold:>8} {w_count:>5} {n-w_count:>5} {n:>7} {wr:>6.1f}% {per_day:>5.1f} {target:>8}{marker}"
        )


def main():
    dates = load_cached_days()
    print(f"Loaded {len(dates)} cached days ({dates[0]} -> {dates[-1]})")

    # ── Split: first 75% train, last 25% test ──
    split_idx = int(len(dates) * 0.75)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]
    print(f"Train: {len(train_dates)} days ({train_dates[0]} -> {train_dates[-1]})")
    print(f"Test:  {len(test_dates)} days ({test_dates[0]} -> {test_dates[-1]})")

    print("\n-- Loading ALL alerts with computed factors --")
    all_alerts = load_all_alerts(dates)
    print(f"\nTotal decided alerts: {len(all_alerts)}")

    train_alerts = [a for a in all_alerts if a.date <= train_dates[-1]]
    test_alerts = [a for a in all_alerts if a.date > train_dates[-1]]
    print(f"Train alerts: {len(train_alerts)}")
    print(f"Test alerts:  {len(test_alerts)}")

    baseline_w = sum(1 for a in all_alerts if a.outcome == "correct")
    baseline_wr = baseline_w / len(all_alerts) * 100
    print(f"Baseline WR (unfiltered): {baseline_wr:.1f}%")

    train_baseline_w = sum(1 for a in train_alerts if a.outcome == "correct")
    train_baseline_wr = train_baseline_w / len(train_alerts) * 100

    # ==================================================================
    # STEP 1: Validate every scoring factor (on TRAIN data)
    # ==================================================================
    print("\n" + "=" * 75)
    print("STEP 1: Factor Validation (TRAIN set only)")
    print("=" * 75)
    print(f"Train baseline WR: {train_baseline_wr:.1f}% ({len(train_alerts)} alerts)")

    # 1a. Level quality
    print("\n-- 1a. Level Quality --")
    for level in ["FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272", "IBL", "VWAP", "IBH"]:
        bucket = [a for a in train_alerts if a.level == level]
        if bucket:
            w = sum(1 for a in bucket if a.outcome == "correct")
            wr = w / len(bucket) * 100
            suggested = suggest_weight(wr, train_baseline_wr)
            print(
                f"  {level:<22} {w:>5}W {len(bucket)-w:>5}L {len(bucket):>7} "
                f"{wr:>6.1f}%  suggested={suggested:+d}"
            )

    # 1b. Direction x Level
    print("\n-- 1b. Direction x Level --")
    for level in ["FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272", "IBL", "IBH", "VWAP"]:
        for direction in ["up", "down"]:
            bucket = [
                a for a in train_alerts if a.level == level and a.direction == direction
            ]
            if bucket:
                w = sum(1 for a in bucket if a.outcome == "correct")
                wr = w / len(bucket) * 100
                suggested = suggest_weight(wr, train_baseline_wr)
                print(
                    f"  {level:<22} x {direction:<5} {w:>5}W {len(bucket)-w:>5}L "
                    f"{len(bucket):>7} {wr:>6.1f}%  suggested={suggested:+d}"
                )

    # 1c. Time of day
    print("\n-- 1c. Time of Day --")
    time_buckets = [
        (
            "10:30-12:00",
            lambda a: a.now_et
            and 10 * 60 + 30 <= a.now_et.hour * 60 + a.now_et.minute < 12 * 60,
        ),
        (
            "12:00-14:00",
            lambda a: a.now_et
            and 12 * 60 <= a.now_et.hour * 60 + a.now_et.minute < 14 * 60,
        ),
        (
            "14:00-15:00",
            lambda a: a.now_et
            and 14 * 60 <= a.now_et.hour * 60 + a.now_et.minute < 15 * 60,
        ),
        (
            "15:00-16:00 (power)",
            lambda a: a.now_et and a.now_et.hour * 60 + a.now_et.minute >= 15 * 60,
        ),
    ]
    for label, fn in time_buckets:
        bucket = [a for a in train_alerts if fn(a)]
        if bucket:
            w = sum(1 for a in bucket if a.outcome == "correct")
            wr = w / len(bucket) * 100
            suggested = suggest_weight(wr, train_baseline_wr)
            print(
                f"  {label:<30} {w:>5}W {len(bucket)-w:>5}L {len(bucket):>7} "
                f"{wr:>6.1f}%  suggested={suggested:+d}"
            )

    # 1d. Tick rate — use data-driven quartiles
    print("\n-- 1d. Tick Rate (trades/min in 3-min window) --")
    tick_rates = [a.tick_rate for a in train_alerts if a.tick_rate is not None]
    if tick_rates:
        tq = np.percentile(tick_rates, [0, 25, 50, 75, 100])
        print(
            f"  Tick rate distribution: min={tq[0]:.0f}, Q25={tq[1]:.0f}, "
            f"median={tq[2]:.0f}, Q75={tq[3]:.0f}, max={tq[4]:.0f}"
        )

        # Show current scoring bands
        for label, lo, hi in [
            ("< 1750", 0, 1750),
            ("1750-2000 (current +2)", 1750, 2000),
            (">= 2000", 2000, float("inf")),
        ]:
            bucket = [
                a
                for a in train_alerts
                if a.tick_rate is not None and lo <= a.tick_rate < hi
            ]
            if bucket:
                w = sum(1 for a in bucket if a.outcome == "correct")
                wr = w / len(bucket) * 100
                suggested = suggest_weight(wr, train_baseline_wr)
                print(
                    f"  {label:<30} {w:>5}W {len(bucket)-w:>5}L {len(bucket):>7} "
                    f"{wr:>6.1f}%  suggested={suggested:+d}"
                )

        # Also try quartile-based buckets
        print("\n  Quartile-based tick rate buckets:")
        for label, lo, hi in [
            (f"Q1 (< {tq[1]:.0f})", 0, tq[1]),
            (f"Q2 ({tq[1]:.0f}-{tq[2]:.0f})", tq[1], tq[2]),
            (f"Q3 ({tq[2]:.0f}-{tq[3]:.0f})", tq[2], tq[3]),
            (f"Q4 (> {tq[3]:.0f})", tq[3], float("inf")),
        ]:
            bucket = [
                a
                for a in train_alerts
                if a.tick_rate is not None and lo <= a.tick_rate < hi
            ]
            if bucket:
                w = sum(1 for a in bucket if a.outcome == "correct")
                wr = w / len(bucket) * 100
                suggested = suggest_weight(wr, train_baseline_wr)
                print(
                    f"  {label:<30} {w:>5}W {len(bucket)-w:>5}L {len(bucket):>7} "
                    f"{wr:>6.1f}%  suggested={suggested:+d}"
                )

    # 1e. Test count
    print("\n-- 1e. Test Count --")
    for tc in range(1, 11):
        bucket = [a for a in train_alerts if a.entry_count == tc]
        if bucket:
            w = sum(1 for a in bucket if a.outcome == "correct")
            wr = w / len(bucket) * 100
            suggested = suggest_weight(wr, train_baseline_wr)
            print(
                f"  Test #{tc:<3} {w:>5}W {len(bucket)-w:>5}L {len(bucket):>7} "
                f"{wr:>6.1f}%  suggested={suggested:+d}"
            )
    bucket_high = [a for a in train_alerts if a.entry_count >= 10]
    if bucket_high:
        w = sum(1 for a in bucket_high if a.outcome == "correct")
        wr = w / len(bucket_high) * 100
        print(
            f"  Test #10+ {w:>5}W {len(bucket_high)-w:>5}L {len(bucket_high):>7} "
            f"{wr:>6.1f}%"
        )

    # 1f. Session move
    print("\n-- 1f. Session Move (pts from day open) --")
    move_buckets = [
        (
            "<= -50 (strong red)",
            lambda a: a.session_move_pts is not None and a.session_move_pts <= -50,
        ),
        (
            "(-50, -20]",
            lambda a: a.session_move_pts is not None
            and -50 < a.session_move_pts <= -20,
        ),
        (
            "(-20, -10] (sweet red)",
            lambda a: a.session_move_pts is not None
            and -20 < a.session_move_pts <= -10,
        ),
        (
            "(-10, 0]",
            lambda a: a.session_move_pts is not None and -10 < a.session_move_pts <= 0,
        ),
        (
            "(0, 10] (near-zero green)",
            lambda a: a.session_move_pts is not None and 0 < a.session_move_pts <= 10,
        ),
        (
            "(10, 20] (sweet green)",
            lambda a: a.session_move_pts is not None and 10 < a.session_move_pts <= 20,
        ),
        (
            "(20, 50]",
            lambda a: a.session_move_pts is not None and 20 < a.session_move_pts <= 50,
        ),
        (
            "> 50 (strong green)",
            lambda a: a.session_move_pts is not None and a.session_move_pts > 50,
        ),
    ]
    for label, fn in move_buckets:
        bucket = [a for a in train_alerts if fn(a)]
        if bucket:
            w = sum(1 for a in bucket if a.outcome == "correct")
            wr = w / len(bucket) * 100
            suggested = suggest_weight(wr, train_baseline_wr)
            print(
                f"  {label:<30} {w:>5}W {len(bucket)-w:>5}L {len(bucket):>7} "
                f"{wr:>6.1f}%  suggested={suggested:+d}"
            )

    # 1g. Streak
    print("\n-- 1g. Streak --")
    streak_buckets = [
        ("2+ consecutive wins", lambda a: a.consecutive_wins >= 2),
        (
            "1 win or fresh",
            lambda a: a.consecutive_wins <= 1 and a.consecutive_losses <= 1,
        ),
        ("2+ consecutive losses", lambda a: a.consecutive_losses >= 2),
    ]
    for label, fn in streak_buckets:
        bucket = [a for a in train_alerts if fn(a)]
        if bucket:
            w = sum(1 for a in bucket if a.outcome == "correct")
            wr = w / len(bucket) * 100
            suggested = suggest_weight(wr, train_baseline_wr)
            print(
                f"  {label:<30} {w:>5}W {len(bucket)-w:>5}L {len(bucket):>7} "
                f"{wr:>6.1f}%  suggested={suggested:+d}"
            )

    # ==================================================================
    # STEP 2: Threshold sweep with CURRENT weights on train and test
    # ==================================================================
    print("\n" + "=" * 75)
    print("STEP 2: Current Weights -- Threshold Sweep")
    print("=" * 75)

    current_weights = Weights()  # defaults match current scoring.py
    threshold_sweep(train_alerts, current_weights, len(train_dates), "TRAIN set:")
    threshold_sweep(
        test_alerts, current_weights, len(test_dates), "TEST set (out-of-sample):"
    )

    # ==================================================================
    # STEP 3: Try optimized weights
    # ==================================================================
    print("\n" + "=" * 75)
    print("STEP 3: Optimized Weights -- Threshold Sweep")
    print("=" * 75)

    # Compute data-driven weights from train set
    def train_wr(filter_fn) -> float:
        bucket = [a for a in train_alerts if filter_fn(a)]
        if len(bucket) < 30:
            return train_baseline_wr  # not enough data, use baseline
        w = sum(1 for a in bucket if a.outcome == "correct")
        return w / len(bucket) * 100

    # Build optimized weights from train data
    opt = Weights()

    # Level weights
    opt.level_fib_hi = suggest_weight(
        train_wr(lambda a: a.level == "FIB_EXT_HI_1.272"), train_baseline_wr
    )
    opt.level_ibl = suggest_weight(
        train_wr(lambda a: a.level == "IBL"), train_baseline_wr
    )
    opt.level_fib_lo = suggest_weight(
        train_wr(lambda a: a.level == "FIB_EXT_LO_1.272"), train_baseline_wr
    )
    opt.level_vwap = suggest_weight(
        train_wr(lambda a: a.level == "VWAP"), train_baseline_wr
    )
    opt.level_ibh = suggest_weight(
        train_wr(lambda a: a.level == "IBH"), train_baseline_wr
    )

    # Direction x Level combo weights
    opt.combo_fib_hi_up = suggest_weight(
        train_wr(lambda a: a.level == "FIB_EXT_HI_1.272" and a.direction == "up"),
        train_baseline_wr,
    )
    opt.combo_fib_lo_down = suggest_weight(
        train_wr(lambda a: a.level == "FIB_EXT_LO_1.272" and a.direction == "down"),
        train_baseline_wr,
    )
    opt.combo_ibl_down = suggest_weight(
        train_wr(lambda a: a.level == "IBL" and a.direction == "down"),
        train_baseline_wr,
    )
    opt.combo_vwap_up = suggest_weight(
        train_wr(lambda a: a.level == "VWAP" and a.direction == "up"), train_baseline_wr
    )
    opt.combo_ibh_up = suggest_weight(
        train_wr(lambda a: a.level == "IBH" and a.direction == "up"), train_baseline_wr
    )
    opt.combo_ibl_up = suggest_weight(
        train_wr(lambda a: a.level == "IBL" and a.direction == "up"), train_baseline_wr
    )
    opt.combo_fib_lo_up = suggest_weight(
        train_wr(lambda a: a.level == "FIB_EXT_LO_1.272" and a.direction == "up"),
        train_baseline_wr,
    )
    opt.combo_fib_hi_down = suggest_weight(
        train_wr(lambda a: a.level == "FIB_EXT_HI_1.272" and a.direction == "down"),
        train_baseline_wr,
    )
    opt.combo_vwap_down = suggest_weight(
        train_wr(lambda a: a.level == "VWAP" and a.direction == "down"),
        train_baseline_wr,
    )

    # Time
    opt.time_power_hour = suggest_weight(
        train_wr(
            lambda a: a.now_et and a.now_et.hour * 60 + a.now_et.minute >= 15 * 60
        ),
        train_baseline_wr,
    )

    # Tick rate — check if current bands have signal
    tick_wr = train_wr(lambda a: a.tick_rate is not None and 1750 <= a.tick_rate < 2000)
    opt.tick_sweet_spot = suggest_weight(tick_wr, train_baseline_wr)

    # Test count
    opt.test_1 = suggest_weight(
        train_wr(lambda a: a.entry_count == 1), train_baseline_wr
    )
    opt.test_2 = suggest_weight(
        train_wr(lambda a: a.entry_count == 2), train_baseline_wr
    )
    opt.test_3 = suggest_weight(
        train_wr(lambda a: a.entry_count == 3), train_baseline_wr
    )
    opt.test_5 = suggest_weight(
        train_wr(lambda a: a.entry_count == 5), train_baseline_wr
    )

    # Session move
    opt.move_sweet_green = suggest_weight(
        train_wr(
            lambda a: a.session_move_pts is not None and 10 < a.session_move_pts <= 20
        ),
        train_baseline_wr,
    )
    opt.move_sweet_red = suggest_weight(
        train_wr(
            lambda a: a.session_move_pts is not None and -20 < a.session_move_pts <= -10
        ),
        train_baseline_wr,
    )
    opt.move_strong_red = suggest_weight(
        train_wr(
            lambda a: a.session_move_pts is not None and a.session_move_pts <= -50
        ),
        train_baseline_wr,
    )
    opt.move_near_zero_green = suggest_weight(
        train_wr(
            lambda a: a.session_move_pts is not None and 0 < a.session_move_pts <= 10
        ),
        train_baseline_wr,
    )
    opt.move_strong_green = suggest_weight(
        train_wr(lambda a: a.session_move_pts is not None and a.session_move_pts > 50),
        train_baseline_wr,
    )

    # Streak
    opt.streak_win_bonus = suggest_weight(
        train_wr(lambda a: a.consecutive_wins >= 2), train_baseline_wr
    )
    opt.streak_loss_penalty = suggest_weight(
        train_wr(lambda a: a.consecutive_losses >= 2), train_baseline_wr
    )

    # Print the optimized weights vs current
    print("\n  Optimized weights (from train data):")
    for field_name in Weights.__dataclass_fields__:
        if field_name.startswith("tick_l") or field_name.startswith("tick_h"):
            continue
        cur = getattr(current_weights, field_name)
        new = getattr(opt, field_name)
        changed = " <-- CHANGED" if cur != new else ""
        print(f"    {field_name:<25} current={cur:>3}  optimized={new:>3}{changed}")

    threshold_sweep(train_alerts, opt, len(train_dates), "TRAIN set (optimized):")
    threshold_sweep(
        test_alerts, opt, len(test_dates), "TEST set (optimized, out-of-sample):"
    )

    # ==================================================================
    # STEP 4: Also try IBL x up = 0 specifically (user's request)
    # ==================================================================
    print("\n" + "=" * 75)
    print("STEP 4: Current weights but IBL x up = 0")
    print("=" * 75)

    ibl_fix = Weights()
    ibl_fix.combo_ibl_up = 0

    threshold_sweep(train_alerts, ibl_fix, len(train_dates), "TRAIN set (IBL x up=0):")
    threshold_sweep(
        test_alerts, ibl_fix, len(test_dates), "TEST set (IBL x up=0, out-of-sample):"
    )

    # ==================================================================
    # STEP 5: Best candidates -- try multiple weight configs
    # ==================================================================
    print("\n" + "=" * 75)
    print("STEP 5: Configuration Comparison at Various Thresholds")
    print("=" * 75)

    configs: list[tuple[str, Weights]] = [
        ("current", current_weights),
        ("IBL-up=0", ibl_fix),
        ("optimized", opt),
    ]

    # Also try a "conservative optimized" — only change weights with strong signal
    conservative = Weights()
    conservative.combo_ibl_up = 0  # IBL x up fix
    # Only adopt optimized weights where the delta from current is significant
    for field_name in Weights.__dataclass_fields__:
        if field_name.startswith("tick_"):
            continue
        cur = getattr(current_weights, field_name)
        new = getattr(opt, field_name)
        if abs(new - cur) >= 2:  # only big changes
            setattr(conservative, field_name, new)
    configs.append(("conservative", conservative))

    print(
        f"\n  {'Config':<25} {'Thresh':>6} {'W':>5} {'L':>5} {'Total':>7} "
        f"{'WR%':>7} {'/day':>6} {'Test WR%':>9} {'Test/day':>9}"
    )
    print(f"  {'-'*85}")

    for label, weights in configs:
        for thresh in [3, 4, 5, 6]:
            train_scored = [(a, score_alert(a, weights)) for a in train_alerts]
            test_scored = [(a, score_alert(a, weights)) for a in test_alerts]

            train_pass = [(a, s) for a, s in train_scored if s >= thresh]
            test_pass = [(a, s) for a, s in test_scored if s >= thresh]

            tw = sum(1 for a, _ in train_pass if a.outcome == "correct")
            tn = len(train_pass)
            twr = tw / tn * 100 if tn > 0 else 0

            ow = sum(1 for a, _ in test_pass if a.outcome == "correct")
            on = len(test_pass)
            owr = ow / on * 100 if on > 0 else 0

            train_per_day = tn / len(train_dates) if train_dates else 0
            test_per_day = on / len(test_dates) if test_dates else 0

            marker = " <--" if label == "current" and thresh == 5 else ""
            print(
                f"  {label:<25} {thresh:>6} {tw:>5} {tn-tw:>5} {tn:>7} "
                f"{twr:>6.1f}% {train_per_day:>5.1f} {owr:>8.1f}% {test_per_day:>8.1f}{marker}"
            )

    # ==================================================================
    # STEP 6: Stability — split train in half
    # ==================================================================
    print("\n" + "=" * 75)
    print("STEP 6: Stability Check (train first-half vs second-half)")
    print("=" * 75)

    half = len(train_dates) // 2
    half1_dates = set(train_dates[:half])
    half2_dates = set(train_dates[half:])
    half1 = [a for a in train_alerts if a.date in half1_dates]
    half2 = [a for a in train_alerts if a.date in half2_dates]

    for label, weights in [
        ("current", current_weights),
        ("optimized", opt),
        ("IBL-up=0", ibl_fix),
        ("conservative", conservative),
    ]:
        for thresh in [4, 5]:
            h1_pass = [
                (a, s) for a in half1 if (s := score_alert(a, weights)) >= thresh
            ]
            h2_pass = [
                (a, s) for a in half2 if (s := score_alert(a, weights)) >= thresh
            ]

            h1w = sum(1 for a, _ in h1_pass if a.outcome == "correct")
            h2w = sum(1 for a, _ in h2_pass if a.outcome == "correct")
            h1n = len(h1_pass)
            h2n = len(h2_pass)
            h1wr = h1w / h1n * 100 if h1n > 0 else 0
            h2wr = h2w / h2n * 100 if h2n > 0 else 0

            stable = "OK" if min(h1wr, h2wr) >= 78 else "UNSTABLE"
            print(
                f"  {label:<20} >={thresh}: Half1={h1wr:.1f}% ({h1n}) "
                f"Half2={h2wr:.1f}% ({h2n})  {stable}"
            )


if __name__ == "__main__":
    main()
