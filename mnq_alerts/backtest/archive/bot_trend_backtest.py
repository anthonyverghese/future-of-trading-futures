"""
bot_trend_backtest.py — Walk-forward backtest for bot trend/volatility filters.

Tests adding trend and volatility awareness to the bot's entry scoring:
  1. Counter-trend filter: penalize entries against strong recent price movement
  2. Volatility filter: penalize entries during extreme price swings
  3. Max entries per level per day: cap repeated trades at a failing level
  4. Combined filters

Uses corrected VWAP zone, no VWAP trading, T12/S25, $1.24 fee.
Timeouts with negative P&L counted as losses in win rate.

Usage:
    python -u bot_trend_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    _run_zone_numpy,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, evaluate_bot_trade
from config import BOT_EOD_FLATTEN_BUFFER_MIN
from walk_forward import (
    DayEntries,
    DayOutcomes,
    Trade,
    precompute_day_entries,
    precompute_outcomes,
    trade_stats,
    _eod_cutoff_ns,
    INITIAL_TRAIN_DAYS,
    STEP_DAYS,
)
from bot_backtest import BOT_ENTRY_THRESHOLD, BOT_EXIT_THRESHOLD, FEE_PTS

_ET = pytz.timezone("America/New_York")

# Levels the bot actually trades (no VWAP)
BOT_LEVELS = {"IBH", "IBL", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════


def enrich_entries(de: DayEntries, dc: DayCache) -> dict:
    """Compute per-entry features: time, entry count, trend, volatility."""
    n = len(de.global_idx)
    features = {
        "time_hour": np.zeros(n, dtype=np.float64),
        "entry_count": de.entry_count.copy(),
        "trend_30m": np.zeros(n, dtype=np.float64),
        "trend_60m": np.zeros(n, dtype=np.float64),
        "range_30m": np.zeros(n, dtype=np.float64),
        "range_60m": np.zeros(n, dtype=np.float64),
    }

    prices = dc.full_prices
    ts_ns = dc.full_ts_ns
    NS_PER_MIN = 60_000_000_000

    for i in range(n):
        idx = int(de.global_idx[i])
        entry_ns = int(ts_ns[idx])

        # Time of day
        ts_sec = entry_ns / 1e9
        dt = datetime.datetime.fromtimestamp(ts_sec, tz=_ET)
        features["time_hour"][i] = dt.hour + dt.minute / 60.0

        # Trend and range lookbacks
        for minutes, trend_key, range_key in [
            (30, "trend_30m", "range_30m"),
            (60, "trend_60m", "range_60m"),
        ]:
            window_start_ns = entry_ns - minutes * NS_PER_MIN
            # Find start index (binary search)
            start_idx = np.searchsorted(ts_ns, window_start_ns, side="left")
            if start_idx < idx:
                window_prices = prices[start_idx : idx + 1]
                features[trend_key][i] = float(window_prices[-1] - window_prices[0])
                features[range_key][i] = float(
                    np.max(window_prices) - np.min(window_prices)
                )

    return features


# ══════════════════════════════════════════════════════════════════════════════
# BOT SCORING (matches bot_trader.py bot_entry_score)
# ══════════════════════════════════════════════════════════════════════════════


def bot_entry_score(level: str, direction: str, entry_count: int, hour: float) -> int:
    """Score a bot zone entry (same as live bot_trader.py)."""
    score = 0
    if level == "IBL":
        score += 2
    elif level in ("FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"):
        score += 1
    elif level == "IBH":
        score -= 1

    if level == "IBL" and direction == "up":
        score += 1
    if level == "IBH" and direction == "down":
        score += 1
    if level == "IBH" and direction == "up":
        score -= 1

    if entry_count == 1:
        score -= 2
    elif entry_count >= 3:
        score += 1

    if hour >= 15.0:
        score += 2
    elif 13.0 <= hour < 15.0:
        score += 1
    elif 10.5 <= hour < 11.5:
        score -= 1

    return score


# ══════════════════════════════════════════════════════════════════════════════
# FILTERED REPLAY
# ══════════════════════════════════════════════════════════════════════════════


def replay_with_trend_filter(
    days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    outcomes_by_date: dict[datetime.date, DayOutcomes],
    features_by_date: dict[datetime.date, dict],
    daily_loss_usd: float | None,
    max_consec_losses: int | None,
    # Bot score filter
    min_score: int = 1,
    # Trend filter
    trend_lookback: str = "30m",
    trend_threshold: float = 50.0,
    trend_penalty: int = 0,
    # Volatility filter
    vol_lookback: str = "30m",
    vol_threshold: float = 100.0,
    vol_penalty: int = 0,
    # Max entries per level per day
    max_entries_per_level: int | None = None,
) -> list[Trade]:
    """Replay bot trades with trend/vol filters and per-level caps."""
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
        level_entry_counts: dict[str, int] = {}

        for i in range(len(de.global_idx)):
            if stopped:
                break
            if int(de.entry_ns[i]) >= eod_ns:
                break
            if int(de.entry_ns[i]) < position_exit_ns:
                continue

            level = de.level[i]

            # Only trade bot levels (no VWAP)
            if level not in BOT_LEVELS:
                continue

            # Per-level entry cap
            if max_entries_per_level is not None:
                count = level_entry_counts.get(level, 0)
                if count >= max_entries_per_level:
                    continue

            # Compute score with trend/vol adjustments
            hour = float(feat["time_hour"][i])
            direction = de.direction[i]
            entry_count = int(feat["entry_count"][i])
            score = bot_entry_score(level, direction, entry_count, hour)

            # Trend penalty
            if trend_penalty != 0:
                trend_val = float(feat[f"trend_{trend_lookback}"][i])
                if direction == "up" and trend_val < -trend_threshold:
                    score += trend_penalty
                elif direction == "down" and trend_val > trend_threshold:
                    score += trend_penalty

            # Volatility penalty
            if vol_penalty != 0:
                vol_val = float(feat[f"range_{vol_lookback}"][i])
                if vol_val > vol_threshold:
                    score += vol_penalty

            if score < min_score:
                continue

            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            position_exit_ns = int(do.exit_ns[i])
            trades.append(Trade(date=date, level=level, outcome=outcome, pnl_usd=pnl))
            level_entry_counts[level] = level_entry_counts.get(level, 0) + 1
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


def bot_trade_stats(trades: list[Trade]) -> tuple[float, int, int, float, float]:
    """Like trade_stats but counts negative-P&L timeouts as losses."""
    wins = sum(
        1
        for t in trades
        if t.outcome == "win" or (t.outcome == "timeout" and t.pnl_usd >= 0)
    )
    losses = sum(
        1
        for t in trades
        if t.outcome == "loss" or (t.outcome == "timeout" and t.pnl_usd < 0)
    )
    decided = wins + losses
    wr = wins / decided * 100 if decided else 0
    total = sum(t.pnl_usd for t in trades)
    eq = STARTING_BALANCE
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    return total, wins, losses, wr, max_dd


def print_result(
    label: str, trades: list[Trade], test_days: int, baseline_wr: float
) -> None:
    """Print formatted result line."""
    if not trades:
        print(f"  {label:<55} NO TRADES")
        return
    total_pnl, wins, losses, wr, max_dd = bot_trade_stats(trades)
    timeouts = sum(1 for t in trades if t.outcome == "timeout")
    tpd = len(trades) / max(1, test_days)
    ev = total_pnl / len(trades)
    pnl_dd = total_pnl / max_dd if max_dd > 0 else float("inf")
    delta = wr - baseline_wr
    print(
        f"  {label:<55} | "
        f"{len(trades):>4} ({tpd:.1f}/d) "
        f"[{wins}W/{losses}L] | "
        f"WR {wr:>5.1f}% ({delta:>+4.1f}) | "
        f"P&L ${total_pnl:>+8.2f} | "
        f"EV ${ev:>+5.2f} | "
        f"MaxDD ${max_dd:>7.2f} | "
        f"P/DD {pnl_dd:.1f}x"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("Step 1: Loading data...")
    all_days = sorted(load_cached_days())
    print(f"  {len(all_days)} cached trading days")

    print("Step 2: Preprocessing...")
    day_caches: dict[datetime.date, DayCache] = {}
    for d in all_days:
        try:
            df = load_day(d)
            dc = preprocess_day(df, d)
            if dc is not None and dc.ibh > dc.ibl:
                day_caches[d] = dc
        except Exception:
            pass
    valid_days = sorted(day_caches.keys())
    print(f"  {len(valid_days)} valid days")

    print("Step 3: Precomputing entries + features...")
    entries_by_date: dict[datetime.date, DayEntries] = {}
    features_by_date: dict[datetime.date, dict] = {}
    for d in valid_days:
        de = precompute_day_entries(day_caches[d])
        entries_by_date[d] = de
        features_by_date[d] = enrich_entries(de, day_caches[d])

    print("Step 4: Precomputing outcomes (T12/S25)...")
    outcomes_by_date: dict[datetime.date, DayOutcomes] = {}
    for d in valid_days:
        outcomes_by_date[d] = precompute_outcomes(
            entries_by_date[d], day_caches[d], 12.0, 25.0
        )
    print(f"  Done in {time.time()-t0:.0f}s")

    n = len(valid_days)
    test_day_count = n - INITIAL_TRAIN_DAYS

    # ──────────────────────────────────────────────────────────────────────
    # Walk-forward: sweep configs
    # ──────────────────────────────────────────────────────────────────────
    configs = []

    # Baseline: current live config (Score>=1, T12/S25, no VWAP)
    configs.append(
        {
            "label": "BASELINE (Score>=1, no trend/vol)",
        }
    )

    # Counter-trend filter
    for lb in ["30m", "60m"]:
        for thresh in [30, 40, 50, 75]:
            for pen in [-2, -3, -4]:
                configs.append(
                    {
                        "label": f"Trend {lb} >{thresh}pts pen={pen}",
                        "trend_lookback": lb,
                        "trend_threshold": thresh,
                        "trend_penalty": pen,
                    }
                )

    # Volatility filter
    for lb in ["30m", "60m"]:
        for thresh in [50, 75, 100, 125]:
            for pen in [-2, -3]:
                configs.append(
                    {
                        "label": f"Vol {lb} >{thresh}pts pen={pen}",
                        "vol_lookback": lb,
                        "vol_threshold": thresh,
                        "vol_penalty": pen,
                    }
                )

    # Max entries per level per day
    for max_e in [3, 5, 7, 10]:
        configs.append(
            {
                "label": f"Max {max_e} entries/level/day",
                "max_entries_per_level": max_e,
            }
        )

    # Combined: trend + vol
    for t_lb, t_thresh, t_pen in [("30m", 40, -3), ("30m", 50, -3), ("60m", 50, -3)]:
        for v_lb, v_thresh, v_pen in [("30m", 75, -2), ("60m", 100, -2)]:
            configs.append(
                {
                    "label": f"T:{t_lb}>{t_thresh}({t_pen})+V:{v_lb}>{v_thresh}({v_pen})",
                    "trend_lookback": t_lb,
                    "trend_threshold": t_thresh,
                    "trend_penalty": t_pen,
                    "vol_lookback": v_lb,
                    "vol_threshold": v_thresh,
                    "vol_penalty": v_pen,
                }
            )

    # Combined: trend + max entries
    for t_lb, t_thresh, t_pen in [("30m", 40, -3), ("60m", 50, -3)]:
        for max_e in [5, 7]:
            configs.append(
                {
                    "label": f"T:{t_lb}>{t_thresh}({t_pen})+Max {max_e}/level",
                    "trend_lookback": t_lb,
                    "trend_threshold": t_thresh,
                    "trend_penalty": t_pen,
                    "max_entries_per_level": max_e,
                }
            )

    # Combined: vol + max entries
    for v_lb, v_thresh, v_pen in [("30m", 75, -2), ("60m", 100, -2)]:
        for max_e in [5, 7]:
            configs.append(
                {
                    "label": f"V:{v_lb}>{v_thresh}({v_pen})+Max {max_e}/level",
                    "vol_lookback": v_lb,
                    "vol_threshold": v_thresh,
                    "vol_penalty": v_pen,
                    "max_entries_per_level": max_e,
                }
            )

    # Triple: trend + vol + max entries
    for t_lb, t_thresh, t_pen in [("30m", 40, -3), ("60m", 50, -3)]:
        for v_lb, v_thresh, v_pen in [("30m", 75, -2)]:
            for max_e in [5, 7]:
                configs.append(
                    {
                        "label": f"T:{t_lb}>{t_thresh}({t_pen})+V:{v_lb}>{v_thresh}({v_pen})+Max{max_e}",
                        "trend_lookback": t_lb,
                        "trend_threshold": t_thresh,
                        "trend_penalty": t_pen,
                        "vol_lookback": v_lb,
                        "vol_threshold": v_thresh,
                        "vol_penalty": v_pen,
                        "max_entries_per_level": max_e,
                    }
                )

    print(f"\n{'='*130}")
    print(
        f"  BOT TREND/VOL WALK-FORWARD — {n} days, {INITIAL_TRAIN_DAYS}+{STEP_DAYS}, T12/S25, no VWAP"
    )
    print(f"  {len(configs)} configs to test")
    print(f"{'='*130}")

    # Compute baseline first for delta column
    baseline_trades: list[Trade] = []
    k = INITIAL_TRAIN_DAYS
    while k < n:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days:
            break
        # Find best risk params on train
        best_risk = (150.0, 5)
        best_pnl = -float("inf")
        for dloss, mcons in [
            (100.0, 3),
            (150.0, 3),
            (150.0, 4),
            (150.0, 5),
            (200.0, 4),
            (None, None),
        ]:
            tr = replay_with_trend_filter(
                train_days,
                entries_by_date,
                outcomes_by_date,
                features_by_date,
                dloss,
                mcons,
            )
            total = sum(t.pnl_usd for t in tr)
            per_day = total / len(train_days) if train_days else 0
            if per_day > best_pnl:
                best_pnl = per_day
                best_risk = (dloss, mcons)
        baseline_trades.extend(
            replay_with_trend_filter(
                test_days,
                entries_by_date,
                outcomes_by_date,
                features_by_date,
                best_risk[0],
                best_risk[1],
            )
        )
        k += STEP_DAYS

    _, _, _, baseline_wr, _ = bot_trade_stats(baseline_trades)
    print(
        f"\n  {'Config':<55} | {'Trades':>11} | {'WR':>13} | {'P&L':>11} | {'EV':>8} | {'MaxDD':>10} | P/DD"
    )
    print(f"  {'-'*55}-+-{'-'*11}-+-{'-'*13}-+-{'-'*11}-+-{'-'*8}-+-{'-'*10}-+------")
    print_result(
        "BASELINE (Score>=1, no trend/vol)",
        baseline_trades,
        test_day_count,
        baseline_wr,
    )

    # Test each config
    for cfg in configs:
        label = cfg.pop("label")
        all_test_trades: list[Trade] = []
        k = INITIAL_TRAIN_DAYS
        while k < n:
            train_days = valid_days[:k]
            test_days = valid_days[k : k + STEP_DAYS]
            if not test_days:
                break
            best_risk = (150.0, 5)
            best_pnl = -float("inf")
            for dloss, mcons in [
                (100.0, 3),
                (150.0, 3),
                (150.0, 4),
                (150.0, 5),
                (200.0, 4),
                (None, None),
            ]:
                tr = replay_with_trend_filter(
                    train_days,
                    entries_by_date,
                    outcomes_by_date,
                    features_by_date,
                    dloss,
                    mcons,
                    **cfg,
                )
                total = sum(t.pnl_usd for t in tr)
                per_day = total / len(train_days) if train_days else 0
                if per_day > best_pnl:
                    best_pnl = per_day
                    best_risk = (dloss, mcons)
            all_test_trades.extend(
                replay_with_trend_filter(
                    test_days,
                    entries_by_date,
                    outcomes_by_date,
                    features_by_date,
                    best_risk[0],
                    best_risk[1],
                    **cfg,
                )
            )
            k += STEP_DAYS

        print_result(label, all_test_trades, test_day_count, baseline_wr)
        cfg["label"] = label  # restore

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
