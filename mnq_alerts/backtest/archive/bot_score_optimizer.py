"""
bot_score_optimizer.py — Build a scoring system optimized for automated bot trading.

Key differences from human scoring (score_optimizer.py):
  - Optimizes for EV/day (profit), not just win rate
  - Accounts for round-trip fees ($0.54 = 0.27 pts per trade)
  - Entry at the line (1pt threshold), not 7pt early warning
  - Uses target=12, stop=25 (best EV/day from bot_backtest.py)

Usage:
    python bot_score_optimizer.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from bot_backtest import (
    BOT_ENTRY_THRESHOLD,
    BOT_EXIT_THRESHOLD,
    FEE_PTS,
    evaluate_bot_outcome,
    simulate_bot_day,
    run_backtest,
)
from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
)

# Best config from bot_backtest.py Test 2.
TARGET_PTS = 12.0
STOP_PTS = 25.0
WINDOW_SECS = 15 * 60

# Baseline WR for weight calculation.
BASELINE_WR = 72.3  # from bot_backtest Test 6 all-levels


def suggest_weight(bucket_wr: float, baseline: float = BASELINE_WR) -> int:
    """Map WR deviation from baseline to integer weight.

    +/- 2.5% WR → +/- 1 point. Cap at [-4, +4].
    """
    if bucket_wr == 0:
        return 0
    diff = bucket_wr - baseline
    weight = round(diff / 2.5)
    return max(-4, min(4, weight))


def main() -> None:
    print("=" * 75)
    print("  BOT SCORE OPTIMIZER — Data-driven weights for automated trading")
    print(
        f"  Config: target={TARGET_PTS}, stop={STOP_PTS}, entry={BOT_ENTRY_THRESHOLD}"
    )
    print(f"  Fee: {FEE_PTS} pts/trade")
    print("=" * 75)

    t0 = time.time()

    # Load all cached days.
    days = load_cached_days()
    print(f"\n  Loading {len(days)} cached days...")

    day_caches: dict[datetime.date, DayCache] = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except Exception:
            pass
    print(f"  Loaded {len(day_caches)} days in {time.time() - t0:.1f}s")

    valid_days = sorted(day_caches.keys())
    num_days = len(valid_days)

    # 75/25 train/test split.
    split_idx = int(len(valid_days) * 0.75)
    train_days = valid_days[:split_idx]
    test_days = valid_days[split_idx:]

    # Generate all trades with metadata.
    all_trades = run_backtest(
        valid_days,
        day_caches,
        target_pts=TARGET_PTS,
        stop_pts=STOP_PTS,
        window_secs=WINDOW_SECS,
    )
    decided = [t for t in all_trades if t["outcome"] in ("correct", "incorrect")]
    print(f"\n  Total trades: {len(all_trades)} ({len(decided)} decided)")
    w = sum(1 for t in decided if t["outcome"] == "correct")
    print(f"  Baseline: {w}/{len(decided)} = {w/len(decided)*100:.1f}% WR")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: FACTOR ANALYSIS — WR and EV per bucket
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  STEP 1: FACTOR ANALYSIS — WR and suggested weights")
    print(f"{'═' * 75}")

    def analyze_factor(
        name: str,
        buckets: list[tuple[str, list[dict]]],
    ) -> dict[str, int]:
        """Print factor analysis and return {bucket_label: suggested_weight}."""
        print(f"\n  Factor: {name}")
        print(
            f"  {'Bucket':<20}  {'W':>5}  {'L':>5}  {'Total':>6}  "
            f"{'WR%':>6}  {'EV/trade':>9}  {'Weight':>7}"
        )
        print(f"  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*6}  " f"{'-'*6}  {'-'*9}  {'-'*7}")
        weights = {}
        for label, bucket in buckets:
            w_count = sum(1 for t in bucket if t["outcome"] == "correct")
            l_count = sum(1 for t in bucket if t["outcome"] == "incorrect")
            total = w_count + l_count
            wr = w_count / total * 100 if total > 0 else 0
            ev = (
                wr / 100 * (TARGET_PTS - FEE_PTS)
                - (1 - wr / 100) * (STOP_PTS + FEE_PTS)
                if total > 0
                else 0
            )
            weight = suggest_weight(wr) if total >= 30 else 0
            weights[label] = weight
            print(
                f"  {label:<20}  {w_count:>5}  {l_count:>5}  {total:>6}  "
                f"{wr:>5.1f}%  {ev:>+8.2f}  {weight:>+5}"
            )
        return weights

    # Factor 1: Level
    level_buckets = []
    for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        bucket = [t for t in decided if t["level"] == level]
        level_buckets.append((level, bucket))
    level_weights = analyze_factor("Level", level_buckets)

    # Factor 2: Direction × Level combo
    combo_buckets = []
    for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        for direction in ["up", "down"]:
            action = "BUY" if direction == "up" else "SELL"
            bucket = [
                t
                for t in decided
                if t["level"] == level and t["direction"] == direction
            ]
            combo_buckets.append((f"{level}×{action}", bucket))
    combo_weights = analyze_factor("Direction × Level", combo_buckets)

    # Factor 3: Time of day
    time_buckets_config = [
        ("Late morning", datetime.time(10, 30), datetime.time(11, 30)),
        ("Lunch", datetime.time(11, 30), datetime.time(13, 0)),
        ("Afternoon", datetime.time(13, 0), datetime.time(15, 0)),
        ("Power hour", datetime.time(15, 0), datetime.time(16, 0)),
    ]
    time_buckets = []
    for label, start, end in time_buckets_config:
        bucket = [t for t in decided if start <= t["time"].time() < end]
        time_buckets.append((label, bucket))
    time_weights = analyze_factor("Time of day", time_buckets)

    # Factor 4: Tick rate
    tick_config = [
        ("<500", 0, 500),
        ("500-1000", 500, 1000),
        ("1000-1750", 1000, 1750),
        ("1750-2000", 1750, 2000),
        ("2000+", 2000, 99999),
    ]
    tick_buckets = []
    for label, lo, hi in tick_config:
        bucket = [t for t in decided if lo <= t["tick_rate"] < hi]
        tick_buckets.append((label, bucket))
    tick_weights = analyze_factor("Tick rate", tick_buckets)

    # Factor 5: Test count
    test_count_buckets = []
    for tc in [1, 2, 3, 4]:
        bucket = [t for t in decided if t["entry_count"] == tc]
        test_count_buckets.append((f"#{tc}", bucket))
    bucket_5plus = [t for t in decided if t["entry_count"] >= 5]
    test_count_buckets.append(("#5+", bucket_5plus))
    test_weights = analyze_factor("Test count", test_count_buckets)

    # Factor 6: Session move
    move_config = [
        ("<-100", -99999, -100),
        ("-100 to -30", -100, -30),
        ("-30 to +30", -30, 30),
        ("+30 to +100", 30, 100),
        (">+100", 100, 99999),
    ]
    move_buckets = []
    for label, lo, hi in move_config:
        bucket = [t for t in decided if lo <= t["session_move"] < hi]
        move_buckets.append((label, bucket))
    move_weights = analyze_factor("Session move", move_buckets)

    # Factor 7: Direction (global)
    dir_buckets = [
        ("BUY", [t for t in decided if t["direction"] == "up"]),
        ("SELL", [t for t in decided if t["direction"] == "down"]),
    ]
    dir_weights = analyze_factor("Direction", dir_buckets)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: SCORE ALL TRADES AND SWEEP THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  STEP 2: SCORE ALL TRADES AND SWEEP THRESHOLDS")
    print(f"{'═' * 75}")

    def score_trade(t: dict) -> int:
        """Score a single trade using the data-driven weights."""
        score = 0

        # Level weight.
        score += level_weights.get(t["level"], 0)

        # Direction × Level combo.
        action = "BUY" if t["direction"] == "up" else "SELL"
        combo_key = f"{t['level']}×{action}"
        score += combo_weights.get(combo_key, 0)

        # Time of day.
        trade_time = t["time"].time()
        for label, start, end in time_buckets_config:
            if start <= trade_time < end:
                score += time_weights.get(label, 0)
                break

        # Tick rate.
        for label, lo, hi in tick_config:
            if lo <= t["tick_rate"] < hi:
                score += tick_weights.get(label, 0)
                break

        # Test count.
        tc = t["entry_count"]
        if tc >= 5:
            score += test_weights.get("#5+", 0)
        else:
            score += test_weights.get(f"#{tc}", 0)

        # Session move.
        sm = t["session_move"]
        for label, lo, hi in move_config:
            if lo <= sm < hi:
                score += move_weights.get(label, 0)
                break

        # Direction.
        dir_label = "BUY" if t["direction"] == "up" else "SELL"
        score += dir_weights.get(dir_label, 0)

        return score

    # Score all trades.
    for t in all_trades:
        t["bot_score"] = score_trade(t)

    # Threshold sweep.
    scores = sorted(set(t["bot_score"] for t in all_trades))
    print(
        f"\n  {'Score≥':>8}  {'W':>5}  {'L':>5}  {'Inc':>5}  "
        f"{'Decided':>8}  {'WR%':>6}  {'/day':>5}  {'EV/trade':>9}  {'EV/day':>7}"
    )
    print(
        f"  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*5}  "
        f"{'-'*8}  {'-'*6}  {'-'*5}  {'-'*9}  {'-'*7}"
    )
    for min_score in range(min(scores), max(scores) + 1):
        filtered = [t for t in all_trades if t["bot_score"] >= min_score]
        decided_f = [t for t in filtered if t["outcome"] in ("correct", "incorrect")]
        w_count = sum(1 for t in decided_f if t["outcome"] == "correct")
        l_count = len(decided_f) - w_count
        inc = len(filtered) - len(decided_f)
        total = w_count + l_count
        wr = w_count / total * 100 if total > 0 else 0
        per_day = len(filtered) / num_days
        ev = (
            wr / 100 * (TARGET_PTS - FEE_PTS) - (1 - wr / 100) * (STOP_PTS + FEE_PTS)
            if total > 0
            else 0
        )
        ev_day = ev * per_day
        print(
            f"  {min_score:>8}  {w_count:>5}  {l_count:>5}  {inc:>5}  "
            f"{total:>8}  {wr:>5.1f}%  {per_day:>5.1f}  {ev:>+8.2f}  "
            f"{ev_day:>+6.2f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: TRAIN/TEST VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  STEP 3: TRAIN/TEST VALIDATION")
    print(f"{'═' * 75}")

    train_trades = [t for t in all_trades if t["date"] in set(train_days)]
    test_trades_list = [t for t in all_trades if t["date"] in set(test_days)]

    # Find the best threshold on train set (max EV/day with positive EV/trade).
    best_train_ev_day = -999
    best_threshold = 0
    for min_score in range(min(scores), max(scores) + 1):
        filtered = [t for t in train_trades if t["bot_score"] >= min_score]
        decided_f = [t for t in filtered if t["outcome"] in ("correct", "incorrect")]
        w_count = sum(1 for t in decided_f if t["outcome"] == "correct")
        total = len(decided_f)
        if total < 50:
            continue
        wr = w_count / total * 100
        per_day = len(filtered) / len(train_days)
        ev = wr / 100 * (TARGET_PTS - FEE_PTS) - (1 - wr / 100) * (STOP_PTS + FEE_PTS)
        ev_day = ev * per_day
        if ev_day > best_train_ev_day and ev > 0:
            best_train_ev_day = ev_day
            best_threshold = min_score

    print(f"\n  Best train threshold: score >= {best_threshold}")

    # Evaluate on train.
    for label, subset, n_days in [
        ("Train", train_trades, len(train_days)),
        ("Test", test_trades_list, len(test_days)),
    ]:
        print(f"\n  {label} ({n_days} days):")
        for ms in [
            best_threshold - 1,
            best_threshold,
            best_threshold + 1,
            best_threshold + 2,
        ]:
            filtered = [t for t in subset if t["bot_score"] >= ms]
            decided_f = [
                t for t in filtered if t["outcome"] in ("correct", "incorrect")
            ]
            w_count = sum(1 for t in decided_f if t["outcome"] == "correct")
            total = len(decided_f)
            wr = w_count / total * 100 if total > 0 else 0
            per_day = len(filtered) / n_days if n_days > 0 else 0
            ev = (
                wr / 100 * (TARGET_PTS - FEE_PTS)
                - (1 - wr / 100) * (STOP_PTS + FEE_PTS)
                if total > 0
                else 0
            )
            ev_day = ev * per_day
            marker = " ← best" if ms == best_threshold else ""
            print(
                f"    score >= {ms:>3}: {w_count}W/{total-w_count}L = "
                f"{wr:.1f}% WR, {per_day:.1f}/day, "
                f"EV/trade={ev:+.2f}, EV/day={ev_day:+.2f}{marker}"
            )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: STABILITY CHECK
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  STEP 4: STABILITY CHECK (train split in half)")
    print(f"{'═' * 75}")

    half = len(train_days) // 2
    h1_set = set(train_days[:half])
    h2_set = set(train_days[half:])

    for label, day_set, n_days in [
        ("First half", h1_set, half),
        ("Second half", h2_set, len(train_days) - half),
    ]:
        filtered = [
            t
            for t in all_trades
            if t["date"] in day_set and t["bot_score"] >= best_threshold
        ]
        decided_f = [t for t in filtered if t["outcome"] in ("correct", "incorrect")]
        w_count = sum(1 for t in decided_f if t["outcome"] == "correct")
        total = len(decided_f)
        wr = w_count / total * 100 if total > 0 else 0
        per_day = len(filtered) / n_days if n_days > 0 else 0
        ev = (
            wr / 100 * (TARGET_PTS - FEE_PTS) - (1 - wr / 100) * (STOP_PTS + FEE_PTS)
            if total > 0
            else 0
        )
        ev_day = ev * per_day
        print(
            f"  {label} ({n_days} days): {w_count}W/{total-w_count}L = "
            f"{wr:.1f}% WR, {per_day:.1f}/day, "
            f"EV/trade={ev:+.2f}, EV/day={ev_day:+.2f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: PRINT RECOMMENDED SCORING FUNCTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  STEP 5: RECOMMENDED BOT SCORING FUNCTION")
    print(f"{'═' * 75}")

    print(f"\n  MIN_BOT_SCORE = {best_threshold}")
    print(f"  TARGET_PTS = {TARGET_PTS}")
    print(f"  STOP_PTS = {STOP_PTS}")
    print(f"\n  Level weights: {level_weights}")
    print(f"  Combo weights: {combo_weights}")
    print(f"  Time weights: {time_weights}")
    print(f"  Tick rate weights: {tick_weights}")
    print(f"  Test count weights: {test_weights}")
    print(f"  Session move weights: {move_weights}")
    print(f"  Direction weights: {dir_weights}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
