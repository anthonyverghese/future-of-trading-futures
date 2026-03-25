"""
score_optimizer.py — Data-driven composite score optimization.

Computes ALL 7 scoring factors from cached data (including tick_rate,
session_move, and streak which were never backtested), validates each
weight against actual win rates, suggests data-driven weights, and
sweeps thresholds to find 80%+ WR with 600+ samples.

Usage:
    python score_optimizer.py
"""

from __future__ import annotations

import datetime
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    ALERT_THRESHOLD,
    EXIT_THRESHOLD,
    DayCache,
    _run_zone_numpy,
    evaluate_outcome_np,
    load_cached_days,
    load_day,
    preprocess_day,
)

ET = pytz.timezone("America/New_York")

# Match live system constants
TARGET_POINTS = 8.0
STOP_POINTS = 20.0


@dataclass
class ScoredAlert:
    """Alert with all 7 scoring factors computed."""

    date: datetime.date
    alert_time: datetime.datetime
    level: str
    line_price: float
    entry_price: float
    direction: str
    entry_count: int
    outcome: str  # "correct" or "incorrect"
    # Scoring factors
    tick_rate: float  # trades/min in 3-min window before alert
    session_move_pts: float  # current price - day open
    consecutive_wins: int
    consecutive_losses: int
    time_et: datetime.time
    day_index: int  # for split-half validation


def compute_tick_rate(
    full_ts_ns: np.ndarray, alert_idx: int, window_minutes: int = 3
) -> float:
    """Count trades in window before alert, return trades/min."""
    alert_ns = full_ts_ns[alert_idx]
    start_ns = alert_ns - np.int64(window_minutes * 60 * 1_000_000_000)
    start_idx = int(np.searchsorted(full_ts_ns, start_ns, side="left"))
    count = alert_idx - start_idx
    return count / window_minutes


def simulate_day_scored(dc: DayCache, day_index: int) -> list[ScoredAlert]:
    """Simulate one day, computing all scoring factors per alert."""
    prices = dc.post_ib_prices
    n = len(prices)
    day_open = float(dc.full_prices[0])

    all_levels = [
        ("IBH", np.full(n, dc.ibh), EXIT_THRESHOLD, False),
        ("IBL", np.full(n, dc.ibl), EXIT_THRESHOLD, False),
        ("VWAP", dc.post_ib_vwaps, EXIT_THRESHOLD, False),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), EXIT_THRESHOLD, False),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), EXIT_THRESHOLD, False),
    ]

    alerts: list[ScoredAlert] = []

    for level_name, level_arr, et, use_current in all_levels:
        entries = _run_zone_numpy(prices, level_arr, ALERT_THRESHOLD, et, use_current)

        for idx, entry_count, ref_price in entries:
            price = float(prices[idx])
            full_idx = dc.post_ib_start_idx + idx
            ts = dc.post_ib_timestamps[idx]
            direction = "up" if price > ref_price else "down"

            outcome = evaluate_outcome_np(
                full_idx, ref_price, direction, dc.full_ts_ns, dc.full_prices
            )
            if outcome not in ("correct", "incorrect"):
                continue

            # Tick rate
            tick_rate = compute_tick_rate(dc.full_ts_ns, full_idx)

            # Session move
            session_move = price - day_open

            # Time
            t_et = ts.to_pydatetime(warn=False)
            if t_et.tzinfo is None:
                t_et = ET.localize(t_et)
            else:
                t_et = t_et.astimezone(ET)

            alerts.append(
                ScoredAlert(
                    date=dc.date,
                    alert_time=t_et,
                    level=level_name,
                    line_price=ref_price,
                    entry_price=price,
                    direction=direction,
                    entry_count=entry_count,
                    outcome=outcome,
                    tick_rate=tick_rate,
                    session_move_pts=session_move,
                    consecutive_wins=0,  # filled in later
                    consecutive_losses=0,
                    time_et=t_et.time(),
                    day_index=day_index,
                )
            )

    return alerts


def fill_streaks(alerts: list[ScoredAlert]) -> None:
    """Fill consecutive_wins/losses chronologically across all alerts."""
    # Sort by time (should already be mostly sorted by day, but alerts
    # within a day may be interleaved across levels)
    alerts.sort(key=lambda a: a.alert_time)
    recent: list[str] = []
    for a in alerts:
        # Count streak from recent outcomes
        wins = 0
        for o in reversed(recent):
            if o == "correct":
                wins += 1
            else:
                break
        losses = 0
        for o in reversed(recent):
            if o == "incorrect":
                losses += 1
            else:
                break
        a.consecutive_wins = wins
        a.consecutive_losses = losses
        recent.append(a.outcome)


def time_bucket(t: datetime.time) -> str:
    mins = t.hour * 60 + t.minute
    if mins < 11 * 60 + 30:
        return "morning"
    elif mins < 13 * 60:
        return "lunch"
    elif mins < 15 * 60:
        return "afternoon"
    return "power_hour"


def wr(subset: list[ScoredAlert]) -> tuple[int, int, int, float]:
    w = sum(1 for a in subset if a.outcome == "correct")
    t = len(subset)
    return w, t - w, t, w / t if t > 0 else 0.0


def composite_score(a: ScoredAlert, weights: dict) -> int:
    """Compute composite score using given weight configuration."""
    s = 0

    # Level
    s += weights.get(f"level_{a.level}", 0)

    # Direction × Level
    combo = f"combo_{a.level}_{a.direction}"
    s += weights.get(combo, 0)

    # Time of day
    s += weights.get(f"time_{time_bucket(a.time_et)}", 0)

    # Tick rate
    if a.tick_rate >= weights.get("tick_high", 2000):
        s += weights.get("tick_high_score", 0)
    elif a.tick_rate >= weights.get("tick_mid", 1750):
        s += weights.get("tick_mid_score", 0)
    elif a.tick_rate < weights.get("tick_low", 1000):
        s += weights.get("tick_low_score", 0)

    # Test count
    tc = a.entry_count
    if tc == 1:
        s += weights.get("test_1", 0)
    elif tc == 2:
        s += weights.get("test_2", 0)
    elif tc == 3:
        s += weights.get("test_3", 0)
    elif tc == 4:
        s += weights.get("test_4", 0)
    elif tc == 5:
        s += weights.get("test_5", 0)
    else:
        s += weights.get("test_6plus", 0)

    # Session move
    sm = a.session_move_pts
    if -50 < sm <= 0:
        s += weights.get("session_mildly_red", 0)
    elif sm <= -50:
        s += weights.get("session_strongly_red", 0)
    elif sm > 50:
        s += weights.get("session_strongly_green", 0)
    # else: mildly green (0 to 50), no adjustment

    # Streak
    if a.consecutive_wins >= 2:
        s += weights.get("streak_wins", 0)
    elif a.consecutive_losses >= 2:
        s += weights.get("streak_losses", 0)

    return s


# Current production weights
CURRENT_WEIGHTS = {
    # Level
    "level_IBL": 3,
    "level_FIB_EXT_LO_1.272": 2,
    "level_FIB_EXT_HI_1.272": 1,
    "level_VWAP": 0,
    "level_IBH": -1,
    # Direction × Level
    "combo_FIB_EXT_HI_1.272_up": 1,
    "combo_FIB_EXT_LO_1.272_down": 1,
    "combo_IBL_down": 1,
    "combo_IBH_up": -1,
    # Time
    "time_afternoon": 2,
    "time_power_hour": 1,
    "time_lunch": -1,
    "time_morning": -3,
    # Tick rate
    "tick_high": 2000,
    "tick_mid": 1750,
    "tick_low": 1000,
    "tick_high_score": 2,
    "tick_mid_score": 1,
    "tick_low_score": -2,
    # Test count
    "test_1": -4,
    "test_2": 0,
    "test_3": 2,
    "test_4": 1,
    "test_5": -2,
    "test_6plus": -4,
    # Session move
    "session_mildly_red": 2,
    "session_strongly_red": 0,
    "session_strongly_green": -1,
    # Streak
    "streak_wins": 2,
    "streak_losses": -3,
}


def suggest_weight(bucket_wr: float, baseline_wr: float) -> int:
    """Suggest integer weight based on WR deviation from baseline."""
    delta = bucket_wr - baseline_wr
    raw = delta / 0.025  # 2.5% per point
    return max(-4, min(4, round(raw)))


def print_section(title: str) -> None:
    print(f"\n{'═' * 80}")
    print(f"  {title}")
    print(f"{'═' * 80}")


def main() -> None:
    days = load_cached_days()
    print(f"{'═' * 80}")
    print(f"  SCORE OPTIMIZER — Data-Driven Weight Validation")
    print(f"  {days[0]} → {days[-1]}  ({len(days)} days)")
    print(f"{'═' * 80}")

    # Load all days
    all_alerts: list[ScoredAlert] = []
    days_loaded = 0

    for i, date in enumerate(days):
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is None:
                continue
            day_alerts = simulate_day_scored(dc, days_loaded)
            all_alerts.extend(day_alerts)
            days_loaded += 1
        except Exception as e:
            print(f"  Error loading {date}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(days)} days loaded...", flush=True)

    # Fill streaks chronologically
    fill_streaks(all_alerts)

    print(f"  {days_loaded} days loaded, {len(all_alerts)} decided alerts.")
    w_all, l_all, t_all, wr_all = wr(all_alerts)
    print(f"  Baseline: {w_all}W / {l_all}L = {wr_all:.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Current weights with all factors
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 1: Current weights WITH tick_rate + session_move + streak")

    for a in all_alerts:
        a._current_score = composite_score(a, CURRENT_WEIGHTS)

    print(
        f"\n  {'Threshold':<15} {'W':>5}  {'L':>5}  {'Total':>6}  {'WR%':>6}  {'/day':>5}"
    )
    print(f"  {'-' * 15} {'-' * 5}  {'-' * 5}  {'-' * 6}  {'-' * 6}  {'-' * 5}")
    for thr in range(-2, 10):
        filt = [a for a in all_alerts if a._current_score >= thr]
        w, l, t, rate = wr(filt)
        per_day = t / days_loaded
        marker = " ★" if rate >= 0.80 and t >= 600 else ""
        print(
            f"  Score ≥ {thr:<6} {w:>5}  {l:>5}  {t:>6}  {rate:>5.1%}  {per_day:>5.1f}{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Per-factor validation
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 2: Per-Factor Win Rate Analysis")
    baseline = wr_all

    # Factor 1: Level
    print(f"\n  LEVEL (baseline: {baseline:.1%}):")
    print(
        f"  {'Bucket':<30} {'W':>5} {'L':>5} {'Total':>6} {'WR%':>6}  {'Current':>8}  {'Suggested':>10}"
    )
    print(
        f"  {'-' * 30} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}  {'-' * 8}  {'-' * 10}"
    )

    suggested_weights = dict(CURRENT_WEIGHTS)  # start from current

    for lvl in ["IBL", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272", "VWAP", "IBH"]:
        subset = [a for a in all_alerts if a.level == lvl]
        w, l, t, rate = wr(subset)
        cur = CURRENT_WEIGHTS.get(f"level_{lvl}", 0)
        sug = suggest_weight(rate, baseline)
        suggested_weights[f"level_{lvl}"] = sug
        print(f"  {lvl:<30} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}")

    # Factor 2: Direction × Level
    print(f"\n  DIRECTION × LEVEL:")
    print(
        f"  {'Combo':<30} {'W':>5} {'L':>5} {'Total':>6} {'WR%':>6}  {'Current':>8}  {'Suggested':>10}"
    )
    print(
        f"  {'-' * 30} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}  {'-' * 8}  {'-' * 10}"
    )

    for lvl in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        for dir in ["up", "down"]:
            subset = [a for a in all_alerts if a.level == lvl and a.direction == dir]
            if not subset:
                continue
            w, l, t, rate = wr(subset)
            key = f"combo_{lvl}_{dir}"
            cur = CURRENT_WEIGHTS.get(key, 0)
            # Suggest relative to level's own WR (not global baseline)
            lvl_subset = [a for a in all_alerts if a.level == lvl]
            _, _, _, lvl_wr = wr(lvl_subset)
            sug = suggest_weight(rate, lvl_wr)
            suggested_weights[key] = sug
            print(
                f"  {lvl} × {dir:<22} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}"
            )

    # Factor 3: Time of day
    print(f"\n  TIME OF DAY:")
    print(
        f"  {'Bucket':<30} {'W':>5} {'L':>5} {'Total':>6} {'WR%':>6}  {'Current':>8}  {'Suggested':>10}"
    )
    print(
        f"  {'-' * 30} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}  {'-' * 8}  {'-' * 10}"
    )

    for bucket, key in [
        ("morning", "time_morning"),
        ("lunch", "time_lunch"),
        ("afternoon", "time_afternoon"),
        ("power_hour", "time_power_hour"),
    ]:
        subset = [a for a in all_alerts if time_bucket(a.time_et) == bucket]
        w, l, t, rate = wr(subset)
        cur = CURRENT_WEIGHTS.get(key, 0)
        sug = suggest_weight(rate, baseline)
        suggested_weights[key] = sug
        print(
            f"  {bucket:<30} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}"
        )

    # Factor 4: Tick rate
    print(f"\n  TICK RATE (trades/min in 3-min window):")
    tick_rates = sorted(a.tick_rate for a in all_alerts)
    q25 = tick_rates[len(tick_rates) // 4]
    q50 = tick_rates[len(tick_rates) // 2]
    q75 = tick_rates[3 * len(tick_rates) // 4]
    print(f"  Quartiles: Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}")

    print(
        f"  {'Bucket':<30} {'W':>5} {'L':>5} {'Total':>6} {'WR%':>6}  {'Current':>8}  {'Suggested':>10}"
    )
    print(
        f"  {'-' * 30} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}  {'-' * 8}  {'-' * 10}"
    )

    # Use current thresholds AND quartile-based thresholds
    tick_buckets = [
        (f"≥2000/min", lambda a: a.tick_rate >= 2000, "tick_high_score"),
        (f"1750-2000/min", lambda a: 1750 <= a.tick_rate < 2000, "tick_mid_score"),
        (f"1000-1750/min", lambda a: 1000 <= a.tick_rate < 1750, None),
        (f"<1000/min", lambda a: a.tick_rate < 1000, "tick_low_score"),
    ]
    for label, pred, key in tick_buckets:
        subset = [a for a in all_alerts if pred(a)]
        w, l, t, rate = wr(subset)
        cur = CURRENT_WEIGHTS.get(key, 0) if key else 0
        sug = suggest_weight(rate, baseline)
        if key:
            suggested_weights[key] = sug
        print(
            f"  {label:<30} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}"
        )

    # Also show quartile-based buckets
    print(f"\n  Tick rate by quartiles:")
    quartile_buckets = [
        (f"Q4 (≥{q75:.0f}/min)", lambda a: a.tick_rate >= q75),
        (f"Q3 ({q50:.0f}-{q75:.0f}/min)", lambda a: q50 <= a.tick_rate < q75),
        (f"Q2 ({q25:.0f}-{q50:.0f}/min)", lambda a: q25 <= a.tick_rate < q50),
        (f"Q1 (<{q25:.0f}/min)", lambda a: a.tick_rate < q25),
    ]
    for label, pred in quartile_buckets:
        subset = [a for a in all_alerts if pred(a)]
        w, l, t, rate = wr(subset)
        sug = suggest_weight(rate, baseline)
        print(f"  {label:<30} {w:>5} {l:>5} {t:>6} {rate:>5.1%}            {sug:>+10d}")

    # Factor 5: Test count
    print(f"\n  TEST COUNT:")
    print(
        f"  {'Bucket':<30} {'W':>5} {'L':>5} {'Total':>6} {'WR%':>6}  {'Current':>8}  {'Suggested':>10}"
    )
    print(
        f"  {'-' * 30} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}  {'-' * 8}  {'-' * 10}"
    )

    for tc, key in [
        (1, "test_1"),
        (2, "test_2"),
        (3, "test_3"),
        (4, "test_4"),
        (5, "test_5"),
    ]:
        subset = [a for a in all_alerts if a.entry_count == tc]
        w, l, t, rate = wr(subset)
        cur = CURRENT_WEIGHTS.get(key, 0)
        sug = suggest_weight(rate, baseline)
        suggested_weights[key] = sug
        print(
            f"  Test #{tc:<25} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}"
        )

    subset = [a for a in all_alerts if a.entry_count >= 6]
    w, l, t, rate = wr(subset)
    cur = CURRENT_WEIGHTS.get("test_6plus", 0)
    sug = suggest_weight(rate, baseline)
    suggested_weights["test_6plus"] = sug
    print(
        f"  {'Test #6+':<30} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}"
    )

    # Factor 6: Session move
    print(f"\n  SESSION MOVE (price - day open):")
    print(
        f"  {'Bucket':<30} {'W':>5} {'L':>5} {'Total':>6} {'WR%':>6}  {'Current':>8}  {'Suggested':>10}"
    )
    print(
        f"  {'-' * 30} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}  {'-' * 8}  {'-' * 10}"
    )

    session_buckets = [
        (
            "Strongly red (≤-50)",
            lambda a: a.session_move_pts <= -50,
            "session_strongly_red",
        ),
        (
            "Mildly red (-50 to 0)",
            lambda a: -50 < a.session_move_pts <= 0,
            "session_mildly_red",
        ),
        ("Mildly green (0 to 50)", lambda a: 0 < a.session_move_pts <= 50, None),
        (
            "Strongly green (>50)",
            lambda a: a.session_move_pts > 50,
            "session_strongly_green",
        ),
    ]
    for label, pred, key in session_buckets:
        subset = [a for a in all_alerts if pred(a)]
        w, l, t, rate = wr(subset)
        cur = CURRENT_WEIGHTS.get(key, 0) if key else 0
        sug = suggest_weight(rate, baseline)
        if key:
            suggested_weights[key] = sug
        print(
            f"  {label:<30} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}"
        )

    # Factor 7: Streak
    print(f"\n  STREAK:")
    print(
        f"  {'Bucket':<30} {'W':>5} {'L':>5} {'Total':>6} {'WR%':>6}  {'Current':>8}  {'Suggested':>10}"
    )
    print(
        f"  {'-' * 30} {'-' * 5} {'-' * 5} {'-' * 6} {'-' * 6}  {'-' * 8}  {'-' * 10}"
    )

    streak_buckets = [
        ("2+ consecutive wins", lambda a: a.consecutive_wins >= 2, "streak_wins"),
        ("2+ consecutive losses", lambda a: a.consecutive_losses >= 2, "streak_losses"),
        (
            "Neither/mixed",
            lambda a: a.consecutive_wins < 2 and a.consecutive_losses < 2,
            None,
        ),
    ]
    for label, pred, key in streak_buckets:
        subset = [a for a in all_alerts if pred(a)]
        w, l, t, rate = wr(subset)
        cur = CURRENT_WEIGHTS.get(key, 0) if key else 0
        sug = suggest_weight(rate, baseline)
        if key:
            suggested_weights[key] = sug
        print(
            f"  {label:<30} {w:>5} {l:>5} {t:>6} {rate:>5.1%}  {cur:>+8d}  {sug:>+10d}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Re-score with suggested weights and sweep thresholds
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 3: Threshold Sweep with Suggested Weights")

    print(f"\n  Suggested weights (changes from current marked with *):")
    for k in sorted(suggested_weights.keys()):
        cur = CURRENT_WEIGHTS.get(k, 0)
        sug = suggested_weights[k]
        changed = " *" if cur != sug else ""
        print(f"    {k:<40} current={cur:>+3d}  suggested={sug:>+3d}{changed}")

    # Score all alerts with suggested weights
    for a in all_alerts:
        a._suggested_score = composite_score(a, suggested_weights)

    print(
        f"\n  {'Weights':<12} {'Threshold':<12} {'W':>5}  {'L':>5}  {'Total':>6}  {'WR%':>6}  {'/day':>5}  {'Target':>7}"
    )
    print(
        f"  {'-' * 12} {'-' * 12} {'-' * 5}  {'-' * 5}  {'-' * 6}  {'-' * 6}  {'-' * 5}  {'-' * 7}"
    )

    for label, score_attr in [
        ("Current", "_current_score"),
        ("Suggested", "_suggested_score"),
    ]:
        for thr in range(-2, 10):
            filt = [a for a in all_alerts if getattr(a, score_attr) >= thr]
            w, l, t, rate = wr(filt)
            per_day = t / days_loaded
            target = (
                "★ YES"
                if rate >= 0.80 and t >= 600
                else ("close" if rate >= 0.78 and t >= 500 else "")
            )
            print(
                f"  {label:<12} ≥ {thr:<10} {w:>5}  {l:>5}  {t:>6}  {rate:>5.1%}  {per_day:>5.1f}  {target:>7}"
            )
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Stability check (split-half)
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 4: Split-Half Stability Check")

    mid = days_loaded // 2
    first_half = [a for a in all_alerts if a.day_index < mid]
    second_half = [a for a in all_alerts if a.day_index >= mid]

    print(f"\n  First half: {len(first_half)} alerts ({mid} days)")
    print(f"  Second half: {len(second_half)} alerts ({days_loaded - mid} days)")

    # Check best configurations from Step 3
    print(
        f"\n  {'Weights':<12} {'Thr':>4}  {'1st Half WR':>12} {'(n)':>6}  {'2nd Half WR':>12} {'(n)':>6}  {'Stable':>7}"
    )
    print(
        f"  {'-' * 12} {'-' * 4}  {'-' * 12} {'-' * 6}  {'-' * 12} {'-' * 6}  {'-' * 7}"
    )

    for label, score_attr in [
        ("Current", "_current_score"),
        ("Suggested", "_suggested_score"),
    ]:
        for thr in range(0, 8):
            f1 = [a for a in first_half if getattr(a, score_attr) >= thr]
            f2 = [a for a in second_half if getattr(a, score_attr) >= thr]
            _, _, t1, r1 = wr(f1)
            _, _, t2, r2 = wr(f2)
            stable = (
                "YES" if r1 >= 0.78 and r2 >= 0.78 and t1 >= 250 and t2 >= 250 else ""
            )
            if t1 > 0 and t2 > 0:
                print(
                    f"  {label:<12} ≥{thr:<3} {r1:>11.1%} {t1:>5}   {r2:>11.1%} {t2:>5}   {stable:>7}"
                )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Recommended scoring function
    # ══════════════════════════════════════════════════════════════════════════
    print_section("STEP 5: Recommended _composite_score Updates")

    # Find best threshold for suggested weights
    best_thr = None
    best_count = 0
    for thr in range(-2, 10):
        filt = [a for a in all_alerts if a._suggested_score >= thr]
        w, l, t, rate = wr(filt)
        if rate >= 0.80 and t >= 600 and t > best_count:
            best_thr = thr
            best_count = t

    if best_thr is not None:
        filt = [a for a in all_alerts if a._suggested_score >= best_thr]
        w, l, t, rate = wr(filt)
        print(
            f"\n  ★ RECOMMENDED: score ≥ {best_thr} → {rate:.1%} WR, {t} alerts ({t/days_loaded:.1f}/day)"
        )
    else:
        # Find closest to target
        print(f"\n  No configuration hit 80%/600+. Closest options:")
        results = []
        for thr in range(-2, 10):
            filt = [a for a in all_alerts if a._suggested_score >= thr]
            w, l, t, rate = wr(filt)
            if t > 0:
                results.append((thr, w, l, t, rate, t / days_loaded))
        for thr, w, l, t, rate, per_day in results:
            if rate >= 0.78 or t >= 500:
                print(f"    ≥{thr}: {rate:.1%} WR, {t} alerts ({per_day:.1f}/day)")

    # Print the suggested weight changes
    print(f"\n  Weight changes from current → suggested:")
    changes = []
    for k in sorted(suggested_weights.keys()):
        cur = CURRENT_WEIGHTS.get(k, 0)
        sug = suggested_weights[k]
        if cur != sug:
            changes.append((k, cur, sug))
    if changes:
        for k, cur, sug in changes:
            print(f"    {k:<40} {cur:>+3d} → {sug:>+3d}")
    else:
        print("    (no changes)")

    print(f"\n{'═' * 80}")
    print("  Done.")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
