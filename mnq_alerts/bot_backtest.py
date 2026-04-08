"""
bot_backtest.py — Backtest for automated IBKR bot trading.

Key difference from human trading (targeted_backtest.py):
  - Human trading: alert fires 7 pts BEFORE the level as early warning,
    then outcome is evaluated after price hits the line.
  - Bot trading: bot enters at the line itself (0-1 pt threshold),
    target/stop measured from line price, no hit-wait step.

This changes the entry model fundamentally — the bot gets filled at
the level, so every zone entry is a trade with a known fill price.

Usage:
    python bot_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    CACHE_DIR,
    IB_END,
    MARKET_CLOSE,
    MARKET_OPEN,
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    _run_zone_numpy,
)

ET = pytz.timezone("America/New_York")

# Bot entry: trade when price touches the line (within 1 pt).
BOT_ENTRY_THRESHOLD = 1.0

# Exit threshold: how far price must move away to reset the zone.
# Needs to be wide enough to avoid re-triggering on noise.
BOT_EXIT_THRESHOLD = 15.0

# Evaluation window (seconds). Bot can hold longer since it's automated.
BOT_WINDOW_SECS = 15 * 60

# MNQ fees: IBKR charges ~$0.62/side for MNQ, so ~$1.24 round-trip.
# MNQ multiplier is $2/point, so fee in points = $1.24 / $2 = 0.62 pts.
FEE_PTS = 0.62


def evaluate_bot_outcome(
    entry_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    target_pts: float,
    stop_pts: float,
    window_secs: int,
) -> str:
    """Evaluate outcome for a bot trade entered at the line.

    No hit-wait step — the bot is filled at (or very near) the line.
    Target/stop measured from line_price. Clock starts at entry.
    """
    entry_ns = ts_ns[entry_idx]
    window_ns = np.int64(window_secs * 1_000_000_000)
    eval_end_ns = entry_ns + window_ns

    target_idx = -1
    stop_idx = -1

    if direction == "up":
        target_price = line_price + target_pts
        stop_price = line_price - stop_pts
        for i in range(entry_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            if target_idx < 0 and prices[i] >= target_price:
                target_idx = i
            if stop_idx < 0 and prices[i] <= stop_price:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break
    else:
        target_price = line_price - target_pts
        stop_price = line_price + stop_pts
        for i in range(entry_idx + 1, len(prices)):
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


def simulate_bot_day(
    dc: DayCache,
    entry_threshold: float = BOT_ENTRY_THRESHOLD,
    exit_threshold: float = BOT_EXIT_THRESHOLD,
    target_pts: float = 8.0,
    stop_pts: float = 20.0,
    window_secs: int = BOT_WINDOW_SECS,
    levels_filter: set[str] | None = None,
) -> list[dict]:
    """Simulate one day of bot trading.

    Bot enters when price touches the line (within entry_threshold).
    Returns list of trade dicts with outcome and metadata.
    """
    prices = dc.post_ib_prices
    n = len(prices)

    all_levels = [
        ("IBH", np.full(n, dc.ibh), exit_threshold),
        ("IBL", np.full(n, dc.ibl), exit_threshold),
        ("VWAP", dc.post_ib_vwaps, exit_threshold),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), exit_threshold),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), exit_threshold),
    ]
    levels_config = [
        lc for lc in all_levels if levels_filter is None or lc[0] in levels_filter
    ]

    trades: list[dict] = []

    for level_name, level_arr, et in levels_config:
        entries = _run_zone_numpy(prices, level_arr, entry_threshold, et)
        for local_idx, entry_count, ref_price in entries:
            global_idx = dc.post_ib_start_idx + local_idx
            entry_price = float(dc.full_prices[global_idx])
            line_price = float(level_arr[local_idx])
            direction = "up" if entry_price > line_price else "down"

            outcome = evaluate_bot_outcome(
                global_idx,
                line_price,
                direction,
                dc.full_ts_ns,
                dc.full_prices,
                target_pts,
                stop_pts,
                window_secs,
            )

            trade_time = dc.post_ib_timestamps[local_idx]

            # Compute tick rate (trades in 3-min window / 3).
            window_start_ns = dc.full_ts_ns[global_idx] - np.int64(
                3 * 60 * 1_000_000_000
            )
            tick_start = int(
                np.searchsorted(dc.full_ts_ns, window_start_ns, side="left")
            )
            tick_rate = (global_idx - tick_start) / 3.0

            # Session move from day open.
            rth_start_idx = int(
                np.searchsorted(
                    dc.full_ts_ns,
                    dc.post_ib_timestamps[0].replace(hour=9, minute=30).value,
                    side="left",
                )
            )
            day_open = float(dc.full_prices[max(0, rth_start_idx)])
            session_move = entry_price - day_open

            trades.append(
                {
                    "date": dc.date,
                    "time": trade_time,
                    "level": level_name,
                    "line_price": line_price,
                    "entry_price": entry_price,
                    "direction": direction,
                    "entry_count": entry_count,
                    "tick_rate": tick_rate,
                    "session_move": session_move,
                    "outcome": outcome,
                }
            )

    return trades


def run_backtest(
    days: list[datetime.date],
    day_caches: dict[datetime.date, DayCache],
    entry_threshold: float = BOT_ENTRY_THRESHOLD,
    exit_threshold: float = BOT_EXIT_THRESHOLD,
    target_pts: float = 8.0,
    stop_pts: float = 20.0,
    window_secs: int = BOT_WINDOW_SECS,
    levels_filter: set[str] | None = None,
) -> list[dict]:
    """Run backtest across all days. Returns list of all trades."""
    all_trades = []
    for date in days:
        dc = day_caches.get(date)
        if dc is None:
            continue
        all_trades.extend(
            simulate_bot_day(
                dc,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                target_pts=target_pts,
                stop_pts=stop_pts,
                window_secs=window_secs,
                levels_filter=levels_filter,
            )
        )
    return all_trades


def print_summary(
    trades: list[dict],
    label: str = "",
    num_days: int = 0,
    target_pts: float = 8.0,
    stop_pts: float = 20.0,
) -> None:
    """Print win/loss summary for a list of trades."""
    decided = [t for t in trades if t["outcome"] in ("correct", "incorrect")]
    w = sum(1 for t in decided if t["outcome"] == "correct")
    l = sum(1 for t in decided if t["outcome"] == "incorrect")
    inc = len(trades) - len(decided)
    total = w + l
    wr = w / total * 100 if total > 0 else 0
    per_day = len(trades) / num_days if num_days > 0 else 0
    ev = (
        wr / 100 * (target_pts - FEE_PTS) - (1 - wr / 100) * (stop_pts + FEE_PTS)
        if total > 0
        else 0
    )
    ev_day = ev * per_day
    prefix = f"  {label}: " if label else "  "
    print(
        f"{prefix}{w}W / {l}L / {inc}inc = {wr:.1f}% WR "
        f"({len(trades)} total, {per_day:.1f}/day, "
        f"EV/trade={ev:+.2f}, EV/day={ev_day:+.2f})"
    )


def main() -> None:
    print("=" * 75)
    print("  BOT BACKTEST — Automated trading at the line")
    print("  Entry: price touches level (within 1 pt)")
    print("  No early warning threshold needed")
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
        except Exception as exc:
            print(f"  [skip] {date}: {exc}")
    print(f"  Loaded {len(day_caches)} days in {time.time() - t0:.1f}s")

    num_days = len(day_caches)
    valid_days = sorted(day_caches.keys())

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: BASELINE — Current human params applied to bot entry
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 1: BASELINE — Bot entry (1pt) with human eval params (8/20/15min)")
    print(f"{'═' * 75}")

    baseline = run_backtest(valid_days, day_caches)
    print_summary(baseline, "All levels", num_days, target_pts=8.0, stop_pts=20.0)

    # Per-level breakdown.
    for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        lvl_trades = [t for t in baseline if t["level"] == level]
        print_summary(lvl_trades, level, num_days, target_pts=8.0, stop_pts=20.0)

    # Per-direction breakdown.
    for direction in ["up", "down"]:
        dir_trades = [t for t in baseline if t["direction"] == direction]
        print_summary(
            dir_trades, f"  {direction}", num_days, target_pts=8.0, stop_pts=20.0
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: TARGET / STOP SWEEP (with fees)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 2: TARGET / STOP SWEEP (fee-adjusted)")
    print(f"  Fee: {FEE_PTS:.2f} pts/trade (${FEE_PTS * 2:.2f} round-trip)")
    print(f"  EV = WR × (target - fee) - (1 - WR) × (stop + fee)")
    print(f"  EV/day = EV × trades/day  [total daily profit in pts]")
    print(f"{'═' * 75}")

    targets = [2, 3, 4, 5, 6, 8, 10, 12, 16]
    stops = [4, 6, 8, 12, 16, 20, 25]

    print(
        f"\n  {'Target':>7}  {'Stop':>5}  {'W':>5}  {'L':>5}  {'Inc':>5}  "
        f"{'Decided':>8}  {'WR%':>6}  {'/day':>5}  {'EV':>7}  {'EV/day':>7}"
    )
    print(
        f"  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  "
        f"{'-'*8}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*7}"
    )

    best_ev_day = -999
    best_config = (0, 0)

    for target in targets:
        for stop in stops:
            trades = run_backtest(
                valid_days, day_caches, target_pts=target, stop_pts=stop
            )
            decided = [t for t in trades if t["outcome"] in ("correct", "incorrect")]
            w = sum(1 for t in decided if t["outcome"] == "correct")
            l = sum(1 for t in decided if t["outcome"] == "incorrect")
            inc = len(trades) - len(decided)
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            per_day = len(trades) / num_days if num_days > 0 else 0
            # Fee-adjusted EV per trade.
            ev = (
                wr / 100 * (target - FEE_PTS) - (1 - wr / 100) * (stop + FEE_PTS)
                if total > 0
                else 0
            )
            ev_day = ev * per_day
            marker = ""
            if target == 8 and stop == 20:
                marker = " ← human"
            if ev_day > best_ev_day and ev > 0:
                best_ev_day = ev_day
                best_config = (target, stop)
            print(
                f"  {target:>7}  {stop:>5}  {w:>5}  {l:>5}  {inc:>5}  "
                f"{total:>8}  {wr:>5.1f}%  {per_day:>5.1f}  {ev:>+6.2f}  "
                f"{ev_day:>+6.2f}{marker}"
            )
        print()

    print(
        f"  Best EV/day: target={best_config[0]}, stop={best_config[1]} "
        f"(EV/day={best_ev_day:+.2f} pts)"
    )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: ENTRY THRESHOLD SWEEP
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 3: ENTRY THRESHOLD SWEEP")
    print(f"  How tight should the bot's entry be?")
    print(f"{'═' * 75}")

    entry_thresholds = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]

    print(
        f"\n  {'Entry':>7}  {'W':>5}  {'L':>5}  {'Inc':>5}  "
        f"{'Decided':>8}  {'WR%':>6}  {'/day':>5}"
    )
    print(f"  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  " f"{'-'*8}  {'-'*6}  {'-'*5}")
    for et in entry_thresholds:
        trades = run_backtest(valid_days, day_caches, entry_threshold=et)
        decided = [t for t in trades if t["outcome"] in ("correct", "incorrect")]
        w = sum(1 for t in decided if t["outcome"] == "correct")
        l = sum(1 for t in decided if t["outcome"] == "incorrect")
        inc = len(trades) - len(decided)
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        per_day = len(trades) / num_days if num_days > 0 else 0
        marker = " ← bot default" if et == 1.0 else (" ← human" if et == 7.0 else "")
        print(
            f"  {et:>7.2f}  {w:>5}  {l:>5}  {inc:>5}  "
            f"{total:>8}  {wr:>5.1f}%  {per_day:>5.1f}{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: EXIT THRESHOLD SWEEP
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 4: EXIT THRESHOLD SWEEP (zone reset distance)")
    print(f"  How far must price leave before the bot can re-enter?")
    print(f"{'═' * 75}")

    exit_thresholds = [8, 10, 12, 15, 20, 25, 30]

    print(
        f"\n  {'Exit':>7}  {'W':>5}  {'L':>5}  {'Inc':>5}  "
        f"{'Decided':>8}  {'WR%':>6}  {'/day':>5}"
    )
    print(f"  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  " f"{'-'*8}  {'-'*6}  {'-'*5}")
    for exit_t in exit_thresholds:
        trades = run_backtest(valid_days, day_caches, exit_threshold=exit_t)
        decided = [t for t in trades if t["outcome"] in ("correct", "incorrect")]
        w = sum(1 for t in decided if t["outcome"] == "correct")
        l = sum(1 for t in decided if t["outcome"] == "incorrect")
        inc = len(trades) - len(decided)
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        per_day = len(trades) / num_days if num_days > 0 else 0
        marker = (
            " ← human" if exit_t == 20 else (" ← bot default" if exit_t == 15 else "")
        )
        print(
            f"  {exit_t:>7}  {w:>5}  {l:>5}  {inc:>5}  "
            f"{total:>8}  {wr:>5.1f}%  {per_day:>5.1f}{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 5: WINDOW DURATION SWEEP
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 5: EVALUATION WINDOW SWEEP")
    print(f"  How long should the bot hold before giving up?")
    print(f"{'═' * 75}")

    windows = [5 * 60, 10 * 60, 15 * 60, 20 * 60, 30 * 60, 45 * 60, 60 * 60]

    print(
        f"\n  {'Window':>8}  {'W':>5}  {'L':>5}  {'Inc':>5}  "
        f"{'Decided':>8}  {'WR%':>6}  {'/day':>5}"
    )
    print(f"  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*5}  " f"{'-'*8}  {'-'*6}  {'-'*5}")
    for win in windows:
        trades = run_backtest(valid_days, day_caches, window_secs=win)
        decided = [t for t in trades if t["outcome"] in ("correct", "incorrect")]
        w = sum(1 for t in decided if t["outcome"] == "correct")
        l = sum(1 for t in decided if t["outcome"] == "incorrect")
        inc = len(trades) - len(decided)
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        per_day = len(trades) / num_days if num_days > 0 else 0
        marker = " ← current" if win == 15 * 60 else ""
        print(
            f"  {win // 60:>5} min  {w:>5}  {l:>5}  {inc:>5}  "
            f"{total:>8}  {wr:>5.1f}%  {per_day:>5.1f}{marker}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 6: PER-LEVEL BREAKDOWN WITH BEST CONFIG
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print(
        f"  TEST 6: PER-LEVEL BREAKDOWN — best config "
        f"(target={best_config[0]}, stop={best_config[1]})"
    )
    print(f"{'═' * 75}")

    bt, bs = best_config
    best_trades = run_backtest(
        valid_days,
        day_caches,
        target_pts=bt,
        stop_pts=bs,
    )
    print_summary(best_trades, "All levels", num_days, target_pts=bt, stop_pts=bs)

    for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        lvl_trades = [t for t in best_trades if t["level"] == level]
        if not lvl_trades:
            continue
        print_summary(lvl_trades, level, num_days, target_pts=bt, stop_pts=bs)
        for direction in ["up", "down"]:
            dir_trades = [t for t in lvl_trades if t["direction"] == direction]
            if dir_trades:
                print_summary(
                    dir_trades, f"  {direction}", num_days, target_pts=bt, stop_pts=bs
                )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 7: TRAIN/TEST SPLIT VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 7: TRAIN/TEST SPLIT (75/25) — best config")
    print(f"{'═' * 75}")

    split_idx = int(len(valid_days) * 0.75)
    train_days = valid_days[:split_idx]
    test_days = valid_days[split_idx:]

    train_trades = run_backtest(
        train_days,
        day_caches,
        target_pts=best_config[0],
        stop_pts=best_config[1],
    )
    test_trades = run_backtest(
        test_days,
        day_caches,
        target_pts=best_config[0],
        stop_pts=best_config[1],
    )
    print(f"\n  Train ({len(train_days)} days, {train_days[0]} to {train_days[-1]}):")
    print_summary(train_trades, "Train", len(train_days), target_pts=bt, stop_pts=bs)
    print(f"\n  Test ({len(test_days)} days, {test_days[0]} to {test_days[-1]}):")
    print_summary(test_trades, "Test", len(test_days), target_pts=bt, stop_pts=bs)

    # Stability: split train in half.
    half = len(train_days) // 2
    h1_trades = run_backtest(
        train_days[:half],
        day_caches,
        target_pts=bt,
        stop_pts=bs,
    )
    h2_trades = run_backtest(
        train_days[half:],
        day_caches,
        target_pts=bt,
        stop_pts=bs,
    )
    print(f"\n  Stability check (train split in half):")
    print_summary(h1_trades, "First half", half, target_pts=bt, stop_pts=bs)
    print_summary(
        h2_trades, "Second half", len(train_days) - half, target_pts=bt, stop_pts=bs
    )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 8: SCORING FACTOR ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 75}")
    print("  TEST 8: SCORING FACTORS — Which matter for bot trading?")
    print(f"{'═' * 75}")

    best_all = run_backtest(
        valid_days,
        day_caches,
        target_pts=best_config[0],
        stop_pts=best_config[1],
    )
    decided = [t for t in best_all if t["outcome"] in ("correct", "incorrect")]

    # Test count.
    print("\n  Test count (zone re-entries):")
    for tc in [1, 2, 3, 4, 5]:
        label = f"#{tc}" if tc < 5 else "#5+"
        tc_trades = (
            [t for t in decided if t["entry_count"] == tc]
            if tc < 5
            else [t for t in decided if t["entry_count"] >= tc]
        )
        w = sum(1 for t in tc_trades if t["outcome"] == "correct")
        total = len(tc_trades)
        wr = w / total * 100 if total > 0 else 0
        print(f"    {label:<5}  {w}W / {total - w}L = {wr:.1f}% ({total} trades)")

    # Time of day.
    print("\n  Time of day:")
    time_buckets = [
        ("Late morning", datetime.time(10, 30), datetime.time(11, 30)),
        ("Lunch", datetime.time(11, 30), datetime.time(13, 0)),
        ("Afternoon", datetime.time(13, 0), datetime.time(15, 0)),
        ("Power hour", datetime.time(15, 0), datetime.time(16, 0)),
    ]
    for label, start, end in time_buckets:
        bucket = [t for t in decided if start <= t["time"].time() < end]
        w = sum(1 for t in bucket if t["outcome"] == "correct")
        total = len(bucket)
        wr = w / total * 100 if total > 0 else 0
        print(f"    {label:<15}  {w}W / {total - w}L = {wr:.1f}% ({total} trades)")

    # Tick rate.
    print("\n  Tick rate (trades/min in 3-min window):")
    tick_buckets = [
        ("<500", 0, 500),
        ("500-1000", 500, 1000),
        ("1000-1750", 1000, 1750),
        ("1750-2000", 1750, 2000),
        ("2000+", 2000, 99999),
    ]
    for label, lo, hi in tick_buckets:
        bucket = [t for t in decided if lo <= t["tick_rate"] < hi]
        w = sum(1 for t in bucket if t["outcome"] == "correct")
        total = len(bucket)
        wr = w / total * 100 if total > 0 else 0
        print(f"    {label:<12}  {w}W / {total - w}L = {wr:.1f}% ({total} trades)")

    # Session move.
    print("\n  Session move (pts from open):")
    move_buckets = [
        ("<-100", -99999, -100),
        ("-100 to -30", -100, -30),
        ("-30 to +30", -30, 30),
        ("+30 to +100", 30, 100),
        (">+100", 100, 99999),
    ]
    for label, lo, hi in move_buckets:
        bucket = [t for t in decided if lo <= t["session_move"] < hi]
        w = sum(1 for t in bucket if t["outcome"] == "correct")
        total = len(bucket)
        wr = w / total * 100 if total > 0 else 0
        print(f"    {label:<15}  {w}W / {total - w}L = {wr:.1f}% ({total} trades)")

    # Direction × Level combo.
    print("\n  Direction × Level:")
    for level in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
        for direction in ["up", "down"]:
            combo = [
                t
                for t in decided
                if t["level"] == level and t["direction"] == direction
            ]
            w = sum(1 for t in combo if t["outcome"] == "correct")
            total = len(combo)
            wr = w / total * 100 if total > 0 else 0
            if total > 0:
                action = "BUY" if direction == "up" else "SELL"
                print(
                    f"    {level} × {action:<5}  {w}W / {total - w}L = "
                    f"{wr:.1f}% ({total} trades)"
                )

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
