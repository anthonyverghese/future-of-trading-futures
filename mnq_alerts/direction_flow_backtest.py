"""
direction_flow_backtest.py — Identify when breakout beats bounce.

The current system is bounce-based (price near support → BUY expecting reversion).
This backtest asks: are there identifiable conditions where breakout (continuation
through the line) is the right trade instead? If so, alerts could flag those setups.

Analyzes: approach speed, order flow, test count, time of day, level type, and
session context to find predictive features for breakout vs bounce.

Usage:
    python direction_flow_backtest.py
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

from targeted_backtest import (
    ALERT_THRESHOLD,
    EXIT_THRESHOLD,
    DayCache,
    _run_zone_numpy,
    compute_approach_speed,
    evaluate_outcome_np,
    load_cached_days,
    load_day,
    preprocess_day,
)
from scoring import composite_score as _composite_score

ET = pytz.timezone("America/New_York")


@dataclass
class DualAlert:
    """One alert with outcomes for both directions + contextual features."""

    date: datetime.date
    alert_time: datetime.datetime
    level: str
    line_price: float
    entry_price: float
    entry_count: int
    # Bounce = current system direction
    bounce_direction: str  # "up" if price > line, "down" if price < line
    bounce_outcome: str
    bounce_score: int
    # Breakout = opposite direction (continuation through the line)
    breakout_direction: str
    breakout_outcome: str
    breakout_score: int
    # Features for predicting breakout vs bounce
    approach_speed: float  # pts/min toward line (positive = approaching fast)
    buy_imbalance: float  # (buy_vol - sell_vol) / total_vol in 60s before alert
    buy_ratio: float
    large_imbalance: float
    session_move_pts: float  # distance from day open
    time_bucket: str  # "morning", "lunch", "afternoon", "power_hour"


def compute_order_flow(
    full_sides: np.ndarray | None,
    full_sizes: np.ndarray,
    full_ts_ns: np.ndarray,
    alert_idx: int,
    window_secs: int = 60,
) -> tuple[float, float, float]:
    """Compute order flow metrics in the window before an alert."""
    if full_sides is None:
        return 0.0, 0.5, 0.0

    alert_ns = full_ts_ns[alert_idx]
    start_ns = alert_ns - np.int64(window_secs * 1_000_000_000)
    start_idx = int(np.searchsorted(full_ts_ns, start_ns, side="left"))

    sides = full_sides[start_idx : alert_idx + 1]
    sizes = full_sizes[start_idx : alert_idx + 1]

    total_vol = int(np.sum(sizes))
    if total_vol == 0:
        return 0.0, 0.5, 0.0

    buy_vol = int(np.sum(sizes[sides == 1]))
    sell_vol = int(np.sum(sizes[sides == -1]))
    buy_imbalance = (buy_vol - sell_vol) / total_vol

    n_trades = len(sides)
    n_buys = int(np.sum(sides == 1))
    buy_ratio = n_buys / n_trades if n_trades > 0 else 0.5

    large_mask = sizes >= 3
    large_sides = sides[large_mask]
    large_sizes = sizes[large_mask]
    large_total = int(np.sum(large_sizes))
    if large_total > 0:
        lb = int(np.sum(large_sizes[large_sides == 1]))
        ls = int(np.sum(large_sizes[large_sides == -1]))
        large_imbalance = (lb - ls) / large_total
    else:
        large_imbalance = 0.0

    return buy_imbalance, buy_ratio, large_imbalance


def _time_bucket(t: datetime.time) -> str:
    mins = t.hour * 60 + t.minute
    if mins < 11 * 60 + 30:
        return "morning"
    elif mins < 13 * 60:
        return "lunch"
    elif mins < 15 * 60:
        return "afternoon"
    return "power_hour"


def simulate_day_dual(
    dc: DayCache,
    full_sides: np.ndarray | None,
    full_sizes: np.ndarray,
) -> list[DualAlert]:
    """Simulate one day, evaluating both directions for each alert."""
    prices = dc.post_ib_prices
    n = len(prices)

    all_levels = [
        ("IBH", np.full(n, dc.ibh), EXIT_THRESHOLD, False),
        ("IBL", np.full(n, dc.ibl), EXIT_THRESHOLD, False),
        ("VWAP", dc.post_ib_vwaps, EXIT_THRESHOLD, False),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), EXIT_THRESHOLD, False),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), EXIT_THRESHOLD, False),
    ]

    day_open = float(dc.full_prices[0])
    alerts: list[DualAlert] = []

    for level_name, level_arr, et, use_current in all_levels:
        entries = _run_zone_numpy(prices, level_arr, ALERT_THRESHOLD, et, use_current)

        for idx, entry_count, ref_price in entries:
            price = float(prices[idx])
            full_idx = dc.post_ib_start_idx + idx
            ts = dc.post_ib_timestamps[idx]

            bounce_dir = "up" if price > ref_price else "down"
            breakout_dir = "down" if price > ref_price else "up"

            bounce_outcome = evaluate_outcome_np(
                full_idx, ref_price, bounce_dir, dc.full_ts_ns, dc.full_prices
            )
            breakout_outcome = evaluate_outcome_np(
                full_idx, ref_price, breakout_dir, dc.full_ts_ns, dc.full_prices
            )

            buy_imb, buy_rat, large_imb = compute_order_flow(
                full_sides, full_sizes, dc.full_ts_ns, full_idx
            )

            t_et = ts.to_pydatetime(warn=False)
            if t_et.tzinfo is None:
                t_et = ET.localize(t_et)
            else:
                t_et = t_et.astimezone(ET)
            now_et = t_et.time()

            approach_speed = compute_approach_speed(
                dc.full_df, ts, ref_price, bounce_dir
            )

            bounce_score = _composite_score(
                level_name, entry_count, now_et, None, None, bounce_dir
            )
            breakout_score = _composite_score(
                level_name, entry_count, now_et, None, None, breakout_dir
            )

            session_move = price - day_open

            alerts.append(
                DualAlert(
                    date=dc.date,
                    alert_time=t_et,
                    level=level_name,
                    line_price=ref_price,
                    entry_price=price,
                    entry_count=entry_count,
                    bounce_direction=bounce_dir,
                    bounce_outcome=bounce_outcome,
                    bounce_score=bounce_score,
                    breakout_direction=breakout_dir,
                    breakout_outcome=breakout_outcome,
                    breakout_score=breakout_score,
                    approach_speed=approach_speed,
                    buy_imbalance=buy_imb,
                    buy_ratio=buy_rat,
                    large_imbalance=large_imb,
                    session_move_pts=session_move,
                    time_bucket=_time_bucket(now_et),
                )
            )

    return alerts


def print_section(title: str) -> None:
    print(f"\n{'═' * 75}")
    print(f"  {title}")
    print(f"{'═' * 75}")


def wr(w: int, l: int) -> str:
    t = w + l
    return f"{w / t:.1%}" if t > 0 else "N/A"


def main() -> None:
    days = load_cached_days()
    print(f"{'═' * 75}")
    print(f"  BREAKOUT PREDICTION BACKTEST")
    print(f"  {days[0]} → {days[-1]}  ({len(days)} days)")
    print(f"{'═' * 75}")

    all_alerts: list[DualAlert] = []
    days_loaded = 0

    for i, date in enumerate(days):
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is None:
                continue

            if "side" in df.columns:
                side_map = {"B": 1, "A": -1, "N": 0}
                full_sides = df["side"].map(side_map).fillna(0).values.astype(np.int8)
            else:
                full_sides = None
            full_sizes = df["size"].values.astype(np.int32)

            day_alerts = simulate_day_dual(dc, full_sides, full_sizes)
            all_alerts.extend(day_alerts)
            days_loaded += 1
        except Exception as e:
            print(f"  Error loading {date}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(days)} days loaded...", flush=True)

    print(f"  {days_loaded} days loaded, {len(all_alerts)} total alerts.")

    # Only alerts where both directions resolved
    both = [
        a
        for a in all_alerts
        if a.bounce_outcome in ("correct", "incorrect")
        and a.breakout_outcome in ("correct", "incorrect")
    ]
    print(f"  {len(both)} alerts with both directions decided.\n")

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE: How often does each outcome combination happen?
    # ══════════════════════════════════════════════════════════════════════════
    print_section("OUTCOME MATRIX: Bounce vs Breakout")
    print("  For each alert, what happened to both directions?\n")

    bb = sum(
        1
        for a in both
        if a.bounce_outcome == "correct" and a.breakout_outcome == "correct"
    )
    bf = sum(
        1
        for a in both
        if a.bounce_outcome == "correct" and a.breakout_outcome == "incorrect"
    )
    fb = sum(
        1
        for a in both
        if a.bounce_outcome == "incorrect" and a.breakout_outcome == "correct"
    )
    ff = sum(
        1
        for a in both
        if a.bounce_outcome == "incorrect" and a.breakout_outcome == "incorrect"
    )
    total = len(both)

    print(f"  {'':30} Breakout correct  Breakout incorrect")
    print(
        f"  {'Bounce correct':<30} {bb:>6} ({bb/total:.1%})      {bf:>6} ({bf/total:.1%})"
    )
    print(
        f"  {'Bounce incorrect':<30} {fb:>6} ({fb/total:.1%})      {ff:>6} ({ff/total:.1%})"
    )
    print()
    print(
        f"  Key insight: {fb} alerts ({fb/total:.1%}) where breakout won but bounce lost."
    )
    print(f"  If we could predict those, we'd rescue {fb} trades from the loss column.")
    print(
        f"  But {ff} alerts ({ff/total:.1%}) where BOTH lost — breakout can't help there."
    )

    # The population we care about: bounce failures where breakout succeeded
    # vs bounce failures where breakout also failed
    bounce_failures = [a for a in both if a.bounce_outcome == "incorrect"]
    bf_breakout_won = [a for a in bounce_failures if a.breakout_outcome == "correct"]
    bf_both_lost = [a for a in bounce_failures if a.breakout_outcome == "incorrect"]

    print(f"\n  When bounce fails ({len(bounce_failures)} alerts):")
    print(
        f"    Breakout would have won: {len(bf_breakout_won)} ({len(bf_breakout_won)/len(bounce_failures):.1%})"
    )
    print(
        f"    Both failed:             {len(bf_both_lost)} ({len(bf_both_lost)/len(bounce_failures):.1%})"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE ANALYSIS: What predicts "breakout wins, bounce loses"?
    # ══════════════════════════════════════════════════════════════════════════
    print_section("FEATURE ANALYSIS: What predicts breakout over bounce?")
    print("  Comparing features of 4 outcome groups.\n")

    groups = {
        "Bounce W, Break W": [
            a
            for a in both
            if a.bounce_outcome == "correct" and a.breakout_outcome == "correct"
        ],
        "Bounce W, Break L": [
            a
            for a in both
            if a.bounce_outcome == "correct" and a.breakout_outcome == "incorrect"
        ],
        "Bounce L, Break W": bf_breakout_won,
        "Both L": bf_both_lost,
    }

    # Feature averages by group
    print(
        f"  {'Group':<22} {'n':>5}  {'Approach':>9}  {'BuyImb':>7}  {'LrgImb':>7}  {'SessMove':>9}  {'TestCt':>7}"
    )
    print(f"  {'-'*22} {'-'*5}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}")

    for label, grp in groups.items():
        if not grp:
            continue
        avg_approach = sum(a.approach_speed for a in grp) / len(grp)
        avg_buy_imb = sum(a.buy_imbalance for a in grp) / len(grp)
        avg_large_imb = sum(a.large_imbalance for a in grp) / len(grp)
        avg_session = sum(a.session_move_pts for a in grp) / len(grp)
        avg_test = sum(a.entry_count for a in grp) / len(grp)
        print(
            f"  {label:<22} {len(grp):>5}  {avg_approach:>9.2f}  {avg_buy_imb:>7.3f}  "
            f"{avg_large_imb:>7.3f}  {avg_session:>9.1f}  {avg_test:>7.1f}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # APPROACH SPEED: Fast approach → breakout more likely?
    # ══════════════════════════════════════════════════════════════════════════
    print_section("APPROACH SPEED: Does fast approach predict breakout?")
    print("  approach_speed = pts/min toward line in 3 min before alert.")
    print("  Hypothesis: fast approach = momentum → breakout more likely.\n")

    speed_thresholds = [
        ("Slow (<2 pts/min)", lambda a: a.approach_speed < 2),
        ("Medium (2-5 pts/min)", lambda a: 2 <= a.approach_speed < 5),
        ("Fast (5-10 pts/min)", lambda a: 5 <= a.approach_speed < 10),
        ("Very fast (≥10 pts/min)", lambda a: a.approach_speed >= 10),
    ]

    print(
        f"  {'Speed bucket':<25} {'n':>5}  {'Bounce W%':>10}  {'Break W%':>10}  {'Break>Bounce':>13}"
    )
    print(f"  {'-'*25} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*13}")

    for label, pred in speed_thresholds:
        subset = [a for a in both if pred(a)]
        if not subset:
            continue
        bw = sum(1 for a in subset if a.bounce_outcome == "correct")
        kw = sum(1 for a in subset if a.breakout_outcome == "correct")
        n = len(subset)
        # How often did breakout beat bounce in this bucket?
        break_better = sum(
            1
            for a in subset
            if a.breakout_outcome == "correct" and a.bounce_outcome == "incorrect"
        )
        print(
            f"  {label:<25} {n:>5}  {bw/n:>9.1%}  {kw/n:>9.1%}  {break_better:>5} ({break_better/n:.1%})"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ORDER FLOW: Does flow direction predict breakout?
    # ══════════════════════════════════════════════════════════════════════════
    print_section("ORDER FLOW: Does flow predict breakout direction?")
    print("  'Flow favors breakout' = flow pushes in the breakout direction.")
    print(
        "  E.g., price above IBL, breakout=SELL, strong sell flow → favors breakout.\n"
    )

    flow_alerts = [a for a in both if a.buy_imbalance != 0.0 or a.buy_ratio != 0.5]

    if flow_alerts:
        # Flow aligned with breakout direction
        def flow_favors_breakout(a: DualAlert) -> bool | None:
            """True if flow pushes in breakout direction, False if bounce, None if neutral."""
            if a.breakout_direction == "up" and a.buy_imbalance > 0.1:
                return True
            if a.breakout_direction == "down" and a.buy_imbalance < -0.1:
                return True
            if a.breakout_direction == "up" and a.buy_imbalance < -0.1:
                return False
            if a.breakout_direction == "down" and a.buy_imbalance > 0.1:
                return False
            return None  # neutral

        favors_break = [a for a in flow_alerts if flow_favors_breakout(a) is True]
        favors_bounce = [a for a in flow_alerts if flow_favors_breakout(a) is False]
        neutral = [a for a in flow_alerts if flow_favors_breakout(a) is None]

        print(
            f"  {'Flow direction':<30} {'n':>5}  {'Bounce W%':>10}  {'Break W%':>10}  {'Break>Bounce':>13}"
        )
        print(f"  {'-'*30} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*13}")

        for label, subset in [
            ("Favors breakout", favors_break),
            ("Neutral", neutral),
            ("Favors bounce", favors_bounce),
        ]:
            if not subset:
                continue
            bw = sum(1 for a in subset if a.bounce_outcome == "correct")
            kw = sum(1 for a in subset if a.breakout_outcome == "correct")
            n = len(subset)
            break_better = sum(
                1
                for a in subset
                if a.breakout_outcome == "correct" and a.bounce_outcome == "incorrect"
            )
            print(
                f"  {label:<30} {n:>5}  {bw/n:>9.1%}  {kw/n:>9.1%}  {break_better:>5} ({break_better/n:.1%})"
            )

        # Stronger flow thresholds
        print(f"\n  Strong flow (|imbalance| > 0.3):")
        strong_break = [
            a
            for a in flow_alerts
            if (a.breakout_direction == "up" and a.buy_imbalance > 0.3)
            or (a.breakout_direction == "down" and a.buy_imbalance < -0.3)
        ]
        strong_bounce = [
            a
            for a in flow_alerts
            if (a.bounce_direction == "up" and a.buy_imbalance > 0.3)
            or (a.bounce_direction == "down" and a.buy_imbalance < -0.3)
        ]

        for label, subset, outcome_field in [
            ("Strong flow → breakout", strong_break, "breakout"),
            ("Strong flow → bounce", strong_bounce, "bounce"),
        ]:
            if not subset:
                continue
            if outcome_field == "breakout":
                w = sum(1 for a in subset if a.breakout_outcome == "correct")
            else:
                w = sum(1 for a in subset if a.bounce_outcome == "correct")
            n = len(subset)
            print(f"  {label:<30} {n:>5}  Win rate: {w/n:.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # LEVEL TYPE: Are some levels more breakout-prone?
    # ══════════════════════════════════════════════════════════════════════════
    print_section("LEVEL TYPE: Which levels see more breakout wins?")

    print(
        f"\n  {'Level':<25} {'n':>5}  {'Bounce W%':>10}  {'Break W%':>10}  {'Break>Bounce':>13}"
    )
    print(f"  {'-'*25} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*13}")

    for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        subset = [a for a in both if a.level == level]
        if not subset:
            continue
        bw = sum(1 for a in subset if a.bounce_outcome == "correct")
        kw = sum(1 for a in subset if a.breakout_outcome == "correct")
        n = len(subset)
        break_better = sum(
            1
            for a in subset
            if a.breakout_outcome == "correct" and a.bounce_outcome == "incorrect"
        )
        print(
            f"  {level:<25} {n:>5}  {bw/n:>9.1%}  {kw/n:>9.1%}  {break_better:>5} ({break_better/n:.1%})"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TEST COUNT: Do later retests break through more?
    # ══════════════════════════════════════════════════════════════════════════
    print_section("TEST COUNT: Do later retests favor breakout?")
    print("  Hypothesis: level weakens with each test → later tests = breakout.\n")

    print(
        f"  {'Test #':<15} {'n':>5}  {'Bounce W%':>10}  {'Break W%':>10}  {'Break>Bounce':>13}"
    )
    print(f"  {'-'*15} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*13}")

    for tc in [1, 2, 3, 4, 5]:
        if tc < 5:
            subset = [a for a in both if a.entry_count == tc]
            label = f"Test #{tc}"
        else:
            subset = [a for a in both if a.entry_count >= tc]
            label = f"Test #{tc}+"
        if not subset:
            continue
        bw = sum(1 for a in subset if a.bounce_outcome == "correct")
        kw = sum(1 for a in subset if a.breakout_outcome == "correct")
        n = len(subset)
        break_better = sum(
            1
            for a in subset
            if a.breakout_outcome == "correct" and a.bounce_outcome == "incorrect"
        )
        print(
            f"  {label:<15} {n:>5}  {bw/n:>9.1%}  {kw/n:>9.1%}  {break_better:>5} ({break_better/n:.1%})"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TIME OF DAY
    # ══════════════════════════════════════════════════════════════════════════
    print_section("TIME OF DAY: Does breakout likelihood vary by time?")

    print(
        f"\n  {'Time':<15} {'n':>5}  {'Bounce W%':>10}  {'Break W%':>10}  {'Break>Bounce':>13}"
    )
    print(f"  {'-'*15} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*13}")

    for bucket in ["morning", "lunch", "afternoon", "power_hour"]:
        subset = [a for a in both if a.time_bucket == bucket]
        if not subset:
            continue
        bw = sum(1 for a in subset if a.bounce_outcome == "correct")
        kw = sum(1 for a in subset if a.breakout_outcome == "correct")
        n = len(subset)
        break_better = sum(
            1
            for a in subset
            if a.breakout_outcome == "correct" and a.bounce_outcome == "incorrect"
        )
        print(
            f"  {bucket:<15} {n:>5}  {bw/n:>9.1%}  {kw/n:>9.1%}  {break_better:>5} ({break_better/n:.1%})"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINED: Best breakout predictor combos
    # ══════════════════════════════════════════════════════════════════════════
    print_section("COMBINED: Breakout predictor combinations")
    print("  Test multi-factor filters to find reliable breakout signals.\n")

    print(
        f"  {'Filter':<50} {'n':>5}  {'Break W%':>10}  {'Bounce W%':>10}  {'/day':>5}"
    )
    print(f"  {'-'*50} {'-'*5}  {'-'*10}  {'-'*10}  {'-'*5}")

    filters = [
        (
            "Fast approach (≥5) + test ≥3",
            lambda a: a.approach_speed >= 5 and a.entry_count >= 3,
        ),
        (
            "Fast approach (≥5) + test ≥4",
            lambda a: a.approach_speed >= 5 and a.entry_count >= 4,
        ),
        ("Very fast (≥10) + any test", lambda a: a.approach_speed >= 10),
        (
            "Very fast (≥10) + test ≥3",
            lambda a: a.approach_speed >= 10 and a.entry_count >= 3,
        ),
        (
            "Fast (≥5) + flow favors breakout",
            lambda a: a.approach_speed >= 5
            and (
                (a.breakout_direction == "up" and a.buy_imbalance > 0.1)
                or (a.breakout_direction == "down" and a.buy_imbalance < -0.1)
            ),
        ),
        (
            "Fast (≥5) + strong flow (>0.3) → breakout",
            lambda a: a.approach_speed >= 5
            and (
                (a.breakout_direction == "up" and a.buy_imbalance > 0.3)
                or (a.breakout_direction == "down" and a.buy_imbalance < -0.3)
            ),
        ),
        (
            "Test ≥4 + flow favors breakout",
            lambda a: a.entry_count >= 4
            and (
                (a.breakout_direction == "up" and a.buy_imbalance > 0.1)
                or (a.breakout_direction == "down" and a.buy_imbalance < -0.1)
            ),
        ),
        (
            "Fast (≥5) + afternoon/power_hour",
            lambda a: a.approach_speed >= 5
            and a.time_bucket in ("afternoon", "power_hour"),
        ),
        (
            "Fast (≥5) + test ≥3 + afternoon+",
            lambda a: a.approach_speed >= 5
            and a.entry_count >= 3
            and a.time_bucket in ("afternoon", "power_hour"),
        ),
        (
            "Session red (move < -20) + fast (≥5)",
            lambda a: a.session_move_pts < -20 and a.approach_speed >= 5,
        ),
    ]

    for label, pred in filters:
        subset = [a for a in both if pred(a)]
        if not subset:
            print(f"  {label:<50} {'0':>5}  {'N/A':>10}  {'N/A':>10}  {'0.0':>5}")
            continue
        kw = sum(1 for a in subset if a.breakout_outcome == "correct")
        bw = sum(1 for a in subset if a.bounce_outcome == "correct")
        n = len(subset)
        per_day = n / days_loaded if days_loaded > 0 else 0
        print(f"  {label:<50} {n:>5}  {kw/n:>9.1%}  {bw/n:>9.1%}  {per_day:>5.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # WHAT-IF: If we switched to breakout under certain conditions
    # ══════════════════════════════════════════════════════════════════════════
    print_section("WHAT-IF: Switch to breakout when conditions met, else bounce")
    print("  Baseline: always bounce with score≥3.\n")

    # Baseline
    baseline = [a for a in both if a.bounce_score >= 3]
    base_w = sum(1 for a in baseline if a.bounce_outcome == "correct")
    base_l = len(baseline) - base_w
    base_per_day = len(baseline) / days_loaded if days_loaded > 0 else 0
    print(
        f"  {'Always bounce (score≥3)':<50} W:{base_w} L:{base_l}  {wr(base_w, base_l)}  {base_per_day:.1f}/day"
    )
    print()

    # Strategies: use breakout when condition met, else bounce
    switch_rules = [
        ("Switch if approach ≥ 5 pts/min", lambda a: a.approach_speed >= 5),
        ("Switch if approach ≥ 10 pts/min", lambda a: a.approach_speed >= 10),
        (
            "Switch if approach ≥ 5 + test ≥ 3",
            lambda a: a.approach_speed >= 5 and a.entry_count >= 3,
        ),
        (
            "Switch if approach ≥ 5 + flow favors breakout",
            lambda a: a.approach_speed >= 5
            and (
                (a.breakout_direction == "up" and a.buy_imbalance > 0.1)
                or (a.breakout_direction == "down" and a.buy_imbalance < -0.1)
            ),
        ),
        ("Switch if test ≥ 4", lambda a: a.entry_count >= 4),
        (
            "Switch if flow favors breakout (>0.1)",
            lambda a: (a.breakout_direction == "up" and a.buy_imbalance > 0.1)
            or (a.breakout_direction == "down" and a.buy_imbalance < -0.1),
        ),
    ]

    print(
        f"  {'Strategy':<50} {'W':>5}  {'L':>5}  {'Win%':>6}  {'/day':>5}  {'vs base':>8}"
    )
    print(f"  {'-'*50} {'-'*5}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*8}")

    for label, use_breakout in switch_rules:
        w = 0
        l = 0
        for a in both:
            if use_breakout(a):
                # Use breakout direction
                score = a.breakout_score
                outcome = a.breakout_outcome
            else:
                # Use bounce direction (current system)
                score = a.bounce_score
                outcome = a.bounce_outcome

            if score < 3:
                continue
            if outcome == "correct":
                w += 1
            else:
                l += 1

        t = w + l
        per_day = t / days_loaded if days_loaded > 0 else 0
        if t > 0:
            delta = w / t - base_w / len(baseline) if len(baseline) > 0 else 0
            delta_str = f"{delta:+.1%}"
        else:
            delta_str = "N/A"
        print(
            f"  {label:<50} {w:>5}  {l:>5}  {wr(w, l):>6}  {per_day:>5.1f}  {delta_str:>8}"
        )

    print(f"\n{'═' * 75}")
    print("  Done.")
    print(f"{'═' * 75}")


if __name__ == "__main__":
    main()
