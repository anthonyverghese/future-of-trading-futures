"""
bot_improvement_backtest.py — Walk-forward backtests for bot trading improvements.

Tests four improvements independently using walk-forward validation:
  1. Score filter: skip low-quality entries using a simplified scoring system
  2. Tighter stops: sweep stop-loss from 15-25 pts (currently 25)
  3. Level filter: restrict trading to specific levels
  4. Time-of-day filter: only trade during certain hours

Usage:
    python -u bot_improvement_backtest.py
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

import databento as db
from backtest import get_trading_days, fetch_trades
from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    _run_zone_numpy,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, evaluate_bot_trade
from config import BOT_EOD_FLATTEN_BUFFER_MIN, DATABENTO_API_KEY
from walk_forward import (
    DayEntries,
    DayOutcomes,
    Trade,
    precompute_day_entries,
    precompute_outcomes,
    replay_with_risk,
    trade_stats,
    _eod_cutoff_ns,
    INITIAL_TRAIN_DAYS,
    STEP_DAYS,
)
from bot_backtest import BOT_ENTRY_THRESHOLD, BOT_EXIT_THRESHOLD, FEE_PTS

_ET = pytz.timezone("America/New_York")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

NUM_DAYS = 100


def ensure_data(n: int = NUM_DAYS) -> list[datetime.date]:
    """Fetch n trading days of data, caching to disk."""
    days = get_trading_days(n=n, offset=0)
    cached = set(load_cached_days())
    to_fetch = [d for d in days if d not in cached]
    if to_fetch:
        print(f"Fetching {len(to_fetch)} days from Databento...")
        client = db.Historical(key=DATABENTO_API_KEY)
        for i, date in enumerate(to_fetch):
            print(f"  [{i+1}/{len(to_fetch)}] {date}", flush=True)
            try:
                fetch_trades(client, date)
            except Exception as e:
                print(f"    ERROR: {e}")
    # Return only days that are actually cached
    cached = set(load_cached_days())
    return sorted(d for d in days if d in cached)


def load_all_days(days: list[datetime.date]) -> dict[datetime.date, DayCache]:
    """Load and preprocess all days."""
    caches = {}
    for d in days:
        try:
            df = load_day(d)
            dc = preprocess_day(df, d)
            if dc is not None and dc.ibh > dc.ibl:
                caches[d] = dc
        except Exception as e:
            print(f"  Skipping {d}: {e}")
    return caches


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY ENRICHMENT — Add features for scoring/filtering
# ══════════════════════════════════════════════════════════════════════════════


def enrich_entries(de: DayEntries, dc: DayCache) -> dict:
    """Compute per-entry features: time of day, entry count, level, session move."""
    n = len(de.global_idx)
    features = {
        "time_hour": np.zeros(n, dtype=np.float64),
        "entry_count": de.entry_count.copy(),
        "session_move": np.zeros(n, dtype=np.float64),
    }
    day_open = float(dc.full_prices[0])
    for i in range(n):
        ts_ns = int(de.entry_ns[i])
        # Convert to ET hour (fractional)
        ts_sec = ts_ns / 1e9
        dt = datetime.datetime.fromtimestamp(ts_sec, tz=_ET)
        features["time_hour"][i] = dt.hour + dt.minute / 60.0
        features["session_move"][i] = float(de.entry_price[i]) - day_open
    return features


# ══════════════════════════════════════════════════════════════════════════════
# FILTERED REPLAY — Apply entry filters before risk gates
# ══════════════════════════════════════════════════════════════════════════════


def replay_with_filter(
    days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    outcomes_by_date: dict[datetime.date, DayOutcomes],
    features_by_date: dict[datetime.date, dict],
    daily_loss_usd: float | None,
    max_consec_losses: int | None,
    # Filter params
    min_entry_count: int = 0,
    allowed_levels: set[str] | None = None,
    min_hour: float = 0.0,
    max_hour: float = 24.0,
    min_score: int | None = None,
) -> list[Trade]:
    """Replay with entry filters applied before risk gates."""
    trades: list[Trade] = []
    for date in days:
        de = entries_by_date.get(date)
        do = outcomes_by_date.get(date)
        feat = features_by_date.get(date)
        if de is None or do is None or feat is None or len(de.global_idx) == 0:
            continue
        eod_ns = _eod_cutoff_ns(date)
        position_exit_ns = 0
        daily_pnl = 0.0
        consec = 0
        stopped = False
        for i in range(len(de.global_idx)):
            if stopped:
                break
            if int(de.entry_ns[i]) >= eod_ns:
                break
            if int(de.entry_ns[i]) < position_exit_ns:
                continue

            # Apply filters
            if min_entry_count > 0 and int(feat["entry_count"][i]) < min_entry_count:
                continue
            if allowed_levels is not None and de.level[i] not in allowed_levels:
                continue
            hour = float(feat["time_hour"][i])
            if hour < min_hour or hour >= max_hour:
                continue

            # Simple score (if requested)
            if min_score is not None:
                score = _simple_score(
                    de.level[i],
                    de.direction[i],
                    int(feat["entry_count"][i]),
                    hour,
                )
                if score < min_score:
                    continue

            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            position_exit_ns = int(do.exit_ns[i])
            trades.append(
                Trade(date=date, level=de.level[i], outcome=outcome, pnl_usd=pnl)
            )
            daily_pnl += pnl
            if pnl < 0:
                consec += 1
            else:
                consec = 0
            if daily_loss_usd is not None and daily_pnl <= -daily_loss_usd:
                stopped = True
            if max_consec_losses is not None and consec >= max_consec_losses:
                stopped = True
    return trades


def _simple_score(level: str, direction: str, entry_count: int, hour: float) -> int:
    """Simplified scoring for bot entries (lighter than human composite_score)."""
    score = 0
    # Level quality
    if level == "IBL":
        score += 2
    elif level in ("FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"):
        score += 1
    elif level == "IBH":
        score -= 1
    elif level == "VWAP":
        score += 0

    # Direction + level combos
    if level == "IBL" and direction == "up":
        score += 1
    if level == "IBH" and direction == "down":
        score += 1
    if level == "IBH" and direction == "up":
        score -= 1

    # Entry count (retest)
    if entry_count == 1:
        score -= 2
    elif entry_count >= 3:
        score += 1

    # Time of day
    if hour >= 15.0:  # 3pm+ ET (power hour)
        score += 2
    elif 13.0 <= hour < 15.0:  # afternoon
        score += 1
    elif 10.5 <= hour < 11.5:  # post-IB
        score -= 1

    return score


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD WITH FILTERS
# ══════════════════════════════════════════════════════════════════════════════


def walk_forward_filtered(
    valid_days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    outcomes_by_date: dict[datetime.date, DayOutcomes],
    features_by_date: dict[datetime.date, dict],
    configs: list[dict],
    label: str,
) -> None:
    """Run walk-forward for a list of filter configs.

    Each config is a dict of kwargs for replay_with_filter (excluding days/entries/outcomes).
    """
    n = len(valid_days)
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  {len(valid_days)} days, walk-forward {INITIAL_TRAIN_DAYS}+{STEP_DAYS}")
    print(f"{'='*80}")

    for cfg_idx, cfg in enumerate(configs):
        cfg_label = cfg.pop("label", f"Config {cfg_idx}")
        all_test_trades: list[Trade] = []
        k = INITIAL_TRAIN_DAYS
        windows = 0

        while k < n:
            train_days = valid_days[:k]
            test_days = valid_days[k : k + STEP_DAYS]
            if not test_days:
                break

            # Train: find best risk params for this config
            best_risk = (150.0, 4)
            best_pnl = -float("inf")
            for dloss, mcons in [
                (100.0, 3),
                (150.0, 3),
                (150.0, 4),
                (200.0, 4),
                (None, None),
            ]:
                trades = replay_with_filter(
                    train_days,
                    entries_by_date,
                    outcomes_by_date,
                    features_by_date,
                    dloss,
                    mcons,
                    **cfg,
                )
                total = sum(t.pnl_usd for t in trades)
                per_day = total / len(train_days) if train_days else 0
                if per_day > best_pnl:
                    best_pnl = per_day
                    best_risk = (dloss, mcons)

            # Test with best risk
            test_trades = replay_with_filter(
                test_days,
                entries_by_date,
                outcomes_by_date,
                features_by_date,
                best_risk[0],
                best_risk[1],
                **cfg,
            )
            all_test_trades.extend(test_trades)
            windows += 1
            k += STEP_DAYS

        # Report
        if all_test_trades:
            total_pnl, wins, losses, wr, max_dd = trade_stats(all_test_trades)
            decided = wins + losses
            trades_per_day = len(all_test_trades) / max(1, n - INITIAL_TRAIN_DAYS)
            ev = total_pnl / decided if decided else 0
            print(
                f"  {cfg_label:<40} | "
                f"{decided:>4} trades ({trades_per_day:.1f}/day) | "
                f"WR {wr:5.1f}% | "
                f"P&L ${total_pnl:>+8.2f} | "
                f"EV ${ev:>+6.2f} | "
                f"MaxDD ${max_dd:>7.2f}"
            )
        else:
            print(f"  {cfg_label:<40} | NO TRADES")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Run all four improvement tests
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    # Step 1: Ensure data
    print("Step 1: Loading data...")
    days = ensure_data(NUM_DAYS)
    print(f"  {len(days)} trading days available")

    # Step 2: Preprocess
    print("Step 2: Preprocessing...")
    day_caches = load_all_days(days)
    valid_days = sorted(day_caches.keys())
    print(f"  {len(valid_days)} valid days after preprocessing")

    # Step 3: Precompute entries
    print("Step 3: Precomputing entries...")
    entries_by_date = {}
    features_by_date = {}
    for d in valid_days:
        de = precompute_day_entries(day_caches[d])
        entries_by_date[d] = de
        features_by_date[d] = enrich_entries(de, day_caches[d])

    # Step 4: Precompute outcomes for multiple T/S combos
    print("Step 4: Precomputing outcomes...")
    ts_combos = [
        (12.0, 25.0),  # current
        (12.0, 20.0),
        (12.0, 18.0),
        (12.0, 15.0),
        (10.0, 20.0),
        (10.0, 15.0),
        (8.0, 16.0),
        (8.0, 12.0),
    ]
    outcomes_by_ts: dict[tuple[float, float], dict[datetime.date, DayOutcomes]] = {}
    for target, stop in ts_combos:
        outcomes_by_ts[(target, stop)] = {}
        for d in valid_days:
            outcomes_by_ts[(target, stop)][d] = precompute_outcomes(
                entries_by_date[d], day_caches[d], target, stop
            )
        print(f"  T/S={target}/{stop} done")

    # ──────────────────────────────────────────────────────────────────────────
    # BASELINE: Current config (no filters, T12/S25, risk 150/4)
    # ──────────────────────────────────────────────────────────────────────────
    baseline_outcomes = outcomes_by_ts[(12.0, 25.0)]
    walk_forward_filtered(
        valid_days,
        entries_by_date,
        baseline_outcomes,
        features_by_date,
        [{"label": "BASELINE (T12/S25, no filter)"}],
        "BASELINE",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1: Score filter (skip low-quality entries)
    # ──────────────────────────────────────────────────────────────────────────
    configs_score = []
    for min_sc in [-1, 0, 1, 2, 3]:
        configs_score.append(
            {
                "label": f"Score >= {min_sc}",
                "min_score": min_sc,
            }
        )
    walk_forward_filtered(
        valid_days,
        entries_by_date,
        baseline_outcomes,
        features_by_date,
        configs_score,
        "TEST 1: SCORE FILTER (T12/S25)",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2: Tighter stops (with current T12)
    # ──────────────────────────────────────────────────────────────────────────
    configs_ts = []
    for target, stop in ts_combos:
        configs_ts.append(
            {
                "label": f"T{target}/S{stop}",
            }
        )
    # Need separate outcomes for each T/S
    print(f"\n{'='*80}")
    print(f"  TEST 2: TARGET/STOP SWEEP")
    print(f"  {len(valid_days)} days, walk-forward {INITIAL_TRAIN_DAYS}+{STEP_DAYS}")
    print(f"{'='*80}")
    for target, stop in ts_combos:
        outs = outcomes_by_ts[(target, stop)]
        all_test_trades: list[Trade] = []
        k = INITIAL_TRAIN_DAYS
        while k < len(valid_days):
            train_days = valid_days[:k]
            test_days = valid_days[k : k + STEP_DAYS]
            if not test_days:
                break
            best_risk = (150.0, 4)
            best_pnl = -float("inf")
            for dloss, mcons in [
                (100.0, 3),
                (150.0, 3),
                (150.0, 4),
                (200.0, 4),
                (None, None),
            ]:
                trades = replay_with_risk(
                    train_days,
                    entries_by_date,
                    outs,
                    dloss,
                    mcons,
                )
                total = sum(t.pnl_usd for t in trades)
                per_day = total / len(train_days) if train_days else 0
                if per_day > best_pnl:
                    best_pnl = per_day
                    best_risk = (dloss, mcons)
            test_trades = replay_with_risk(
                test_days,
                entries_by_date,
                outs,
                best_risk[0],
                best_risk[1],
            )
            all_test_trades.extend(test_trades)
            k += STEP_DAYS
        if all_test_trades:
            total_pnl, wins, losses, wr, max_dd = trade_stats(all_test_trades)
            decided = wins + losses
            trades_per_day = len(all_test_trades) / max(
                1, len(valid_days) - INITIAL_TRAIN_DAYS
            )
            ev = total_pnl / decided if decided else 0
            print(
                f"  T{target:>4.0f}/S{stop:<4.0f}                              | "
                f"{decided:>4} trades ({trades_per_day:.1f}/day) | "
                f"WR {wr:5.1f}% | "
                f"P&L ${total_pnl:>+8.2f} | "
                f"EV ${ev:>+6.2f} | "
                f"MaxDD ${max_dd:>7.2f}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 3: Level filter (restrict to specific levels)
    # ──────────────────────────────────────────────────────────────────────────
    all_levels = {"IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"}
    configs_level = [
        {"label": "All levels", "allowed_levels": None},
        {"label": "IBH + IBL only", "allowed_levels": {"IBH", "IBL"}},
        {
            "label": "IBH + IBL + Fib",
            "allowed_levels": {"IBH", "IBL", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"},
        },
        {"label": "VWAP only", "allowed_levels": {"VWAP"}},
        {"label": "No VWAP", "allowed_levels": all_levels - {"VWAP"}},
        {"label": "IBL + FIB_LO only", "allowed_levels": {"IBL", "FIB_EXT_LO_1.272"}},
        {"label": "IBH + FIB_HI only", "allowed_levels": {"IBH", "FIB_EXT_HI_1.272"}},
    ]
    walk_forward_filtered(
        valid_days,
        entries_by_date,
        baseline_outcomes,
        features_by_date,
        configs_level,
        "TEST 3: LEVEL FILTER (T12/S25)",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 5: Time-of-day filter
    # ──────────────────────────────────────────────────────────────────────────
    configs_time = [
        {"label": "All hours (10:30-16:00)", "min_hour": 10.5, "max_hour": 16.0},
        {"label": "11:30+ (skip post-IB)", "min_hour": 11.5, "max_hour": 16.0},
        {"label": "13:00+ (afternoon only)", "min_hour": 13.0, "max_hour": 16.0},
        {"label": "14:00+ (late afternoon)", "min_hour": 14.0, "max_hour": 16.0},
        {"label": "15:00+ (power hour)", "min_hour": 15.0, "max_hour": 16.0},
        {"label": "10:30-13:00 (morning)", "min_hour": 10.5, "max_hour": 13.0},
        {"label": "11:30-15:00 (midday)", "min_hour": 11.5, "max_hour": 15.0},
    ]
    walk_forward_filtered(
        valid_days,
        entries_by_date,
        baseline_outcomes,
        features_by_date,
        configs_time,
        "TEST 5: TIME-OF-DAY FILTER (T12/S25)",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # COMBINED: Best filters from each test
    # ──────────────────────────────────────────────────────────────────────────
    # Run after reviewing individual results — use top configs from each
    configs_combined = [
        {
            "label": "Score>=1 + skip post-IB",
            "min_score": 1,
            "min_hour": 11.5,
        },
        {
            "label": "Score>=2 + afternoon",
            "min_score": 2,
            "min_hour": 13.0,
        },
        {
            "label": "Score>=1 + no VWAP",
            "min_score": 1,
            "allowed_levels": all_levels - {"VWAP"},
        },
        {
            "label": "Retest>=2 + skip post-IB",
            "min_entry_count": 2,
            "min_hour": 11.5,
        },
        {
            "label": "Retest>=2 + afternoon + no VWAP",
            "min_entry_count": 2,
            "min_hour": 13.0,
            "allowed_levels": all_levels - {"VWAP"},
        },
        {
            "label": "Score>=1 + afternoon + IBH/IBL/Fib",
            "min_score": 1,
            "min_hour": 13.0,
            "allowed_levels": {"IBH", "IBL", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"},
        },
    ]
    walk_forward_filtered(
        valid_days,
        entries_by_date,
        baseline_outcomes,
        features_by_date,
        configs_combined,
        "COMBINED FILTERS (T12/S25)",
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
