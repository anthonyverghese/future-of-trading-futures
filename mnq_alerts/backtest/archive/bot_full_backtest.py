"""
bot_full_backtest.py — Comprehensive bot backtest with full scoring factors.

Tests the impact of using the full human scoring system for bot entries,
sweeping score thresholds, max entries per level, and IB range-normalized
T/S configs. Uses walk-forward methodology to prevent overfitting.

Motivation: the bot's simple scoring (level + direction + trend) missed
factors that the human scoring correctly uses (volatility, session move,
streak, tick rate). On 2026-04-16, the human app suppressed entries that
the bot took and lost on.

Usage:
    python -u bot_full_backtest.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    _run_zone_numpy,
)
from bot_backtest import BOT_ENTRY_THRESHOLD, BOT_EXIT_THRESHOLD, FEE_PTS
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, evaluate_bot_trade
from config import BOT_EOD_FLATTEN_BUFFER_MIN
from score_optimizer import (
    EnrichedAlert,
    Weights,
    compute_tick_rate,
    score_alert,
    suggest_weight,
)
from walk_forward import (
    DayEntries,
    DayOutcomes,
    precompute_day_entries,
    precompute_outcomes,
    _eod_cutoff_ns,
    INITIAL_TRAIN_DAYS,
    STEP_DAYS,
)

_ET = pytz.timezone("America/New_York")

# ══════════════════════════════════════════════════════════════════════════════
# ENRICHED BOT ENTRIES: compute all 8 human scoring factors per bot entry
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EnrichedBotEntry:
    """Bot zone entry with all scoring factors computed."""

    idx: int  # index into DayEntries arrays
    level: str
    direction: str
    entry_count: int
    entry_price: float
    line_price: float
    entry_ns: int

    # Scoring factors
    now_et: datetime.time | None = None
    tick_rate: float | None = None
    session_move_pts: float | None = None
    range_30m: float | None = None


def enrich_day_entries(
    de: DayEntries, dc: DayCache
) -> list[EnrichedBotEntry]:
    """Compute all scoring factors for each bot zone entry in a day."""
    first_price = float(dc.post_ib_prices[0])
    enriched: list[EnrichedBotEntry] = []

    for i in range(len(de.global_idx)):
        gidx = int(de.global_idx[i])
        entry_ns = int(de.entry_ns[i])
        entry_price = float(de.entry_price[i])

        # Time of day (ET)
        ts_pd = pd.Timestamp(entry_ns, unit="ns", tz="UTC").tz_convert(_ET)
        now_et = ts_pd.time()

        # Tick rate: trades in 3-min window / 3
        tick_rate = compute_tick_rate(dc.full_df, ts_pd)

        # Session move: entry price - first post-IB price
        session_move = entry_price - first_price

        # 30-min range: high - low in last 30 minutes
        window_ns = np.int64(30 * 60 * 1_000_000_000)
        window_start_ns = dc.full_ts_ns[gidx] - window_ns
        win_start_idx = int(
            np.searchsorted(dc.full_ts_ns, window_start_ns, side="left")
        )
        if win_start_idx < gidx:
            window_prices = dc.full_prices[win_start_idx : gidx + 1]
            range_30m = float(np.max(window_prices) - np.min(window_prices))
        else:
            range_30m = None

        enriched.append(
            EnrichedBotEntry(
                idx=i,
                level=de.level[i],
                direction=de.direction[i],
                entry_count=int(de.entry_count[i]),
                entry_price=entry_price,
                line_price=float(de.ref_price[i]),
                entry_ns=entry_ns,
                now_et=now_et,
                tick_rate=tick_rate,
                session_move_pts=session_move,
                range_30m=range_30m,
            )
        )

    return enriched


def score_bot_entry(e: EnrichedBotEntry, w: Weights) -> int:
    """Score a bot entry using the full human scoring system."""
    # Build an EnrichedAlert to reuse the human scoring function.
    ea = EnrichedAlert(
        date=datetime.date.today(),  # placeholder
        level=e.level,
        direction=e.direction,
        entry_count=e.entry_count,
        outcome="correct",  # placeholder
        entry_price=e.entry_price,
        line_price=e.line_price,
        alert_time=datetime.datetime.now(),  # placeholder
        now_et=e.now_et,
        tick_rate=e.tick_rate,
        session_move_pts=e.session_move_pts,
        consecutive_wins=0,  # streaks tracked during replay
        consecutive_losses=0,
    )
    # Score without streak (streak is tracked during replay).
    # Also add volatility penalty if range_30m is available.
    base = score_alert(ea, w)
    if e.range_30m is not None and e.range_30m > 75.0:
        base -= 2
    return base


# ══════════════════════════════════════════════════════════════════════════════
# SCORE-FILTERED REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BotTrade:
    date: datetime.date
    level: str
    direction: str
    outcome: str
    pnl_usd: float
    score: int
    entry_count: int


def replay_scored(
    days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    enriched_by_date: dict[datetime.date, list[EnrichedBotEntry]],
    outcomes_by_date: dict[datetime.date, DayOutcomes],
    weights: Weights,
    min_score: int,
    max_entries_per_level: int,
    daily_loss_usd: float | None,
    max_consec_losses: int | None,
) -> list[BotTrade]:
    """Replay with full scoring, per-level caps, and risk gates."""
    trades: list[BotTrade] = []
    # Track streaks across days (like the live bot).
    consec_wins = 0
    consec_losses = 0

    for date in days:
        de = entries_by_date.get(date)
        do = outcomes_by_date.get(date)
        enriched = enriched_by_date.get(date)
        if de is None or do is None or enriched is None or len(de.global_idx) == 0:
            continue

        eod_ns = _eod_cutoff_ns(date)
        position_exit_ns = 0
        daily_pnl = 0.0
        daily_consec = 0
        stopped = False
        level_counts: dict[str, int] = {}

        for i, eb in enumerate(enriched):
            if stopped:
                break
            if int(eb.entry_ns) >= eod_ns:
                break
            if int(eb.entry_ns) < position_exit_ns:
                continue

            # Per-level cap.
            lc = level_counts.get(eb.level, 0)
            if lc >= max_entries_per_level:
                continue

            # Score with streak context.
            ea = EnrichedAlert(
                date=date,
                level=eb.level,
                direction=eb.direction,
                entry_count=eb.entry_count,
                outcome="correct",
                entry_price=eb.entry_price,
                line_price=eb.line_price,
                alert_time=datetime.datetime.now(),
                now_et=eb.now_et,
                tick_rate=eb.tick_rate,
                session_move_pts=eb.session_move_pts,
                consecutive_wins=consec_wins,
                consecutive_losses=consec_losses,
            )
            score = score_alert(ea, weights)
            if eb.range_30m is not None and eb.range_30m > 75.0:
                score -= 2

            if score < min_score:
                continue

            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            position_exit_ns = int(do.exit_ns[i])
            level_counts[eb.level] = lc + 1

            trades.append(
                BotTrade(
                    date=date,
                    level=eb.level,
                    direction=eb.direction,
                    outcome=outcome,
                    pnl_usd=pnl,
                    score=score,
                    entry_count=eb.entry_count,
                )
            )
            daily_pnl += pnl

            if pnl < 0:
                daily_consec += 1
                consec_wins = 0
                consec_losses += 1
            else:
                daily_consec = 0
                consec_losses = 0
                consec_wins += 1

            if daily_loss_usd is not None and daily_pnl <= -daily_loss_usd:
                stopped = True
            if max_consec_losses is not None and daily_consec >= max_consec_losses:
                stopped = True

    return trades


def trade_summary(
    trades: list[BotTrade], num_days: int, label: str = ""
) -> dict:
    """Compute summary stats for a set of trades."""
    if not trades:
        return {"label": label, "trades": 0}
    wins = sum(1 for t in trades if t.outcome == "win")
    losses = sum(1 for t in trades if t.outcome == "loss")
    timeouts = sum(1 for t in trades if t.outcome == "timeout")
    decided = wins + losses
    wr = wins / decided * 100 if decided else 0
    total_pnl = sum(t.pnl_usd for t in trades)
    per_day = total_pnl / num_days if num_days else 0

    # Max drawdown.
    eq = STARTING_BALANCE
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    return {
        "label": label,
        "trades": len(trades),
        "per_day": len(trades) / num_days if num_days else 0,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "wr": wr,
        "pnl": total_pnl,
        "pnl_per_day": per_day,
        "max_dd": max_dd,
    }


def print_summary(s: dict) -> None:
    if s["trades"] == 0:
        print(f"  {s['label']}: no trades")
        return
    print(
        f"  {s['label']:40s} "
        f"{s['trades']:>4} trades ({s['per_day']:.1f}/day) "
        f"{s['wins']}W/{s['losses']}L/{s['timeouts']}T "
        f"= {s['wr']:.1f}% WR  "
        f"P&L ${s['pnl']:>+8,.0f} (${s['pnl_per_day']:>+6.1f}/day)  "
        f"MaxDD ${s['max_dd']:>6,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# T/S CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

FIXED_TS_GRID = [
    (6.0, 12.0),
    (8.0, 16.0),
    (8.0, 20.0),
    (10.0, 20.0),
    (10.0, 25.0),
    (12.0, 25.0),
]

# IB-range-normalized T/S: target = range * k, stop = range * s
IB_NORM_GRID = [
    (0.05, 0.10),
    (0.05, 0.15),
    (0.07, 0.15),
    (0.07, 0.20),
    (0.10, 0.15),
    (0.10, 0.20),
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    print("=" * 78, flush=True)
    print("  BOT FULL BACKTEST — Comprehensive scoring + parameter sweep", flush=True)
    print("=" * 78, flush=True)

    # Load all days.
    days = load_cached_days()
    print(f"\n  Loading {len(days)} cached days...", flush=True)
    day_caches: dict[datetime.date, DayCache] = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except Exception:
            pass
    valid_days = sorted(day_caches.keys())
    num_days = len(valid_days)
    print(f"  Loaded {num_days} valid days in {time.time()-t0:.1f}s", flush=True)
    print(f"  Range: {valid_days[0]} → {valid_days[-1]}", flush=True)

    # Stage 1: precompute entries.
    t1 = time.time()
    print(f"\n  Stage 1: precomputing zone entries...", flush=True)
    entries_by_date: dict[datetime.date, DayEntries] = {}
    total_entries = 0
    for date in valid_days:
        de = precompute_day_entries(day_caches[date])
        entries_by_date[date] = de
        total_entries += len(de.global_idx)
    print(f"  {total_entries} entries in {time.time()-t1:.1f}s", flush=True)

    # Stage 2: enrich entries with all scoring factors.
    t2 = time.time()
    print(f"\n  Stage 2: computing scoring factors for all entries...", flush=True)
    enriched_by_date: dict[datetime.date, list[EnrichedBotEntry]] = {}
    for i, date in enumerate(valid_days):
        enriched_by_date[date] = enrich_day_entries(
            entries_by_date[date], day_caches[date]
        )
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{num_days} days enriched...", flush=True)
    print(f"  Done in {time.time()-t2:.1f}s", flush=True)

    # Stage 3: precompute outcomes for fixed T/S configs.
    t3 = time.time()
    print(f"\n  Stage 3: precomputing outcomes for {len(FIXED_TS_GRID)} T/S configs...", flush=True)
    outcomes_by_ts: dict[tuple[float, float], dict[datetime.date, DayOutcomes]] = {}
    for ts in FIXED_TS_GRID:
        per_day: dict[datetime.date, DayOutcomes] = {}
        for date in valid_days:
            per_day[date] = precompute_outcomes(
                entries_by_date[date], day_caches[date], ts[0], ts[1]
            )
        outcomes_by_ts[ts] = per_day
        print(f"    T/S={int(ts[0])}/{int(ts[1])} done", flush=True)
    print(f"  Done in {time.time()-t3:.1f}s", flush=True)

    # Stage 4: precompute IB-range-normalized outcomes.
    t4 = time.time()
    print(
        f"\n  Stage 4: precomputing IB-range-normalized outcomes "
        f"for {len(IB_NORM_GRID)} configs...",
        flush=True,
    )
    outcomes_by_norm: dict[
        tuple[float, float], dict[datetime.date, DayOutcomes]
    ] = {}
    for norm in IB_NORM_GRID:
        per_day: dict[datetime.date, DayOutcomes] = {}
        for date in valid_days:
            dc = day_caches[date]
            ib_range = dc.ibh - dc.ibl
            if ib_range < 5:
                # Degenerate IB range — use minimum T/S.
                t_pts, s_pts = 4.0, 8.0
            else:
                t_pts = max(4.0, round(ib_range * norm[0] * 4) / 4)
                s_pts = max(8.0, round(ib_range * norm[1] * 4) / 4)
            per_day[date] = precompute_outcomes(
                entries_by_date[date], dc, t_pts, s_pts
            )
        outcomes_by_norm[norm] = per_day
        print(f"    norm T={norm[0]}/S={norm[1]} done", flush=True)
    print(f"  Done in {time.time()-t4:.1f}s", flush=True)

    # Use default human weights (walk-forward validated).
    weights = Weights()

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Score threshold sweep (fixed 10/25, $150/3, max 5/level)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  TEST 1: Score threshold sweep (10/25, $150/3, max 5/level)", flush=True)
    print("=" * 78, flush=True)

    outcomes_10_25 = outcomes_by_ts[(10.0, 25.0)]
    for min_score in range(-4, 7):
        trades = replay_scored(
            valid_days,
            entries_by_date,
            enriched_by_date,
            outcomes_10_25,
            weights,
            min_score=min_score,
            max_entries_per_level=5,
            daily_loss_usd=150.0,
            max_consec_losses=3,
        )
        s = trade_summary(trades, num_days, f"score >= {min_score}")
        print_summary(s)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: Max entries per level sweep (10/25, $150/3, score >= best)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  TEST 2: Max entries per level sweep (10/25, $150/3)", flush=True)
    print("=" * 78, flush=True)

    for min_score in [0, 1, 2]:
        for max_entries in [1, 2, 3, 5]:
            trades = replay_scored(
                valid_days,
                entries_by_date,
                enriched_by_date,
                outcomes_10_25,
                weights,
                min_score=min_score,
                max_entries_per_level=max_entries,
                daily_loss_usd=150.0,
                max_consec_losses=3,
            )
            s = trade_summary(
                trades, num_days, f"score>={min_score}, max={max_entries}/level"
            )
            print_summary(s)
        print()

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: T/S config sweep (best score, best max_entries, $150/3)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  TEST 3: Fixed T/S sweep (score >= 1, max 3/level, $150/3)", flush=True)
    print("=" * 78, flush=True)

    for ts in FIXED_TS_GRID:
        trades = replay_scored(
            valid_days,
            entries_by_date,
            enriched_by_date,
            outcomes_by_ts[ts],
            weights,
            min_score=1,
            max_entries_per_level=3,
            daily_loss_usd=150.0,
            max_consec_losses=3,
        )
        s = trade_summary(trades, num_days, f"T/S={int(ts[0])}/{int(ts[1])}")
        print_summary(s)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: IB-range-normalized T/S
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print(
        "  TEST 4: IB-range-normalized T/S (score >= 1, max 3/level, $150/3)",
        flush=True,
    )
    print("=" * 78, flush=True)

    for norm in IB_NORM_GRID:
        trades = replay_scored(
            valid_days,
            entries_by_date,
            enriched_by_date,
            outcomes_by_norm[norm],
            weights,
            min_score=1,
            max_entries_per_level=3,
            daily_loss_usd=150.0,
            max_consec_losses=3,
        )
        s = trade_summary(
            trades,
            num_days,
            f"T={norm[0]:.2f}×IB / S={norm[1]:.2f}×IB",
        )
        print_summary(s)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 5: Per-level × direction breakdown (best config)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  TEST 5: Per-level × direction breakdown (10/25, score>=1, max 3)", flush=True)
    print("=" * 78, flush=True)

    all_trades = replay_scored(
        valid_days,
        entries_by_date,
        enriched_by_date,
        outcomes_10_25,
        weights,
        min_score=1,
        max_entries_per_level=3,
        daily_loss_usd=None,
        max_consec_losses=None,
    )
    combos: dict[str, list[BotTrade]] = {}
    for t in all_trades:
        key = f"{t.level} × {t.direction}"
        combos.setdefault(key, []).append(t)
    for key in sorted(combos.keys()):
        s = trade_summary(combos[key], num_days, key)
        print_summary(s)

    # ══════════════════════════════════════════════════════════════════════
    # TEST 6: Recent regime analysis (last 60 days)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  TEST 6: Recent regime (last 60 days)", flush=True)
    print("=" * 78, flush=True)

    recent_days = valid_days[-60:]
    recent_n = len(recent_days)
    print(f"  Period: {recent_days[0]} → {recent_days[-1]} ({recent_n} days)\n")

    for min_score in [-2, 0, 1, 2, 3]:
        for max_entries in [2, 3, 5]:
            trades = replay_scored(
                recent_days,
                entries_by_date,
                enriched_by_date,
                outcomes_10_25,
                weights,
                min_score=min_score,
                max_entries_per_level=max_entries,
                daily_loss_usd=150.0,
                max_consec_losses=3,
            )
            s = trade_summary(
                trades, recent_n, f"score>={min_score}, max={max_entries}/level"
            )
            print_summary(s)
        print()

    # ══════════════════════════════════════════════════════════════════════
    # TEST 7: Walk-forward with full scoring
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  TEST 7: Walk-forward (retrain weights, sweep score + max_entries)", flush=True)
    print("=" * 78, flush=True)

    from walk_forward import fit_weights, load_alerts_for_days

    # Load human alerts for weight training.
    t7 = time.time()
    print("  Loading alerts for weight training...", flush=True)
    all_alerts = load_alerts_for_days(valid_days, day_caches)
    alerts_by_date: dict[datetime.date, list[EnrichedAlert]] = {}
    for a in all_alerts:
        alerts_by_date.setdefault(a.date, []).append(a)
    print(f"  {len(all_alerts)} alerts loaded in {time.time()-t7:.1f}s", flush=True)

    SCORE_GRID = [-1, 0, 1, 2, 3]
    MAX_ENTRIES_GRID = [2, 3, 5]
    RISK_GRID = [(100.0, 3), (150.0, 3), (150.0, 4), (200.0, 3)]

    # Walk-forward: retrain weights on training window, test on next window.
    print(
        f"\n  Walk-forward: train={INITIAL_TRAIN_DAYS}d, step={STEP_DAYS}d",
        flush=True,
    )

    # Collect OOS trades per config.
    oos_by_config: dict[tuple, list[BotTrade]] = {}
    oos_days_total = 0

    k = INITIAL_TRAIN_DAYS
    n = len(valid_days)
    window_count = 0
    while k < n:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days:
            break
        window_count += 1
        oos_days_total += len(test_days)

        # Retrain weights on training alerts.
        train_alerts = [a for d in train_days for a in alerts_by_date.get(d, [])]
        w = fit_weights(train_alerts)

        # Sweep configs on test window.
        for min_s in SCORE_GRID:
            for max_e in MAX_ENTRIES_GRID:
                for risk in RISK_GRID:
                    cfg = (min_s, max_e, risk[0], risk[1])
                    trades = replay_scored(
                        test_days,
                        entries_by_date,
                        enriched_by_date,
                        outcomes_10_25,
                        w,
                        min_score=min_s,
                        max_entries_per_level=max_e,
                        daily_loss_usd=risk[0],
                        max_consec_losses=risk[1],
                    )
                    oos_by_config.setdefault(cfg, []).extend(trades)

        k += STEP_DAYS

    print(f"\n  {window_count} windows, {oos_days_total} OOS days\n", flush=True)

    # Report top configs by P&L/day.
    results = []
    for cfg, trades in oos_by_config.items():
        s = trade_summary(trades, oos_days_total)
        results.append((cfg, s))
    results.sort(key=lambda x: x[1].get("pnl_per_day", 0), reverse=True)

    print(
        f"  {'Score':>5} {'Max/Lv':>6} {'Risk':>8} "
        f"{'Trades':>6} {'/day':>5} {'WR%':>6} "
        f"{'P&L':>9} {'$/day':>7} {'MaxDD':>7}",
        flush=True,
    )
    print(f"  {'-'*5} {'-'*6} {'-'*8} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*7} {'-'*7}", flush=True)
    for cfg, s in results[:25]:
        if s["trades"] == 0:
            continue
        risk_str = f"${int(cfg[2])}/{cfg[3]}"
        print(
            f"  {'>=' + str(cfg[0]):>5} {cfg[1]:>6} {risk_str:>8} "
            f"{s['trades']:>6} {s['per_day']:>5.1f} {s['wr']:>5.1f}% "
            f"${s['pnl']:>+8,.0f} ${s['pnl_per_day']:>+6.1f} ${s['max_dd']:>6,.0f}",
            flush=True,
        )

    # Also show the current live config for comparison.
    current_cfg = (1, 5, 150.0, 3)
    if current_cfg in oos_by_config:
        s = trade_summary(oos_by_config[current_cfg], oos_days_total)
        print(f"\n  Current live config (score>=1, max=5, $150/3):", flush=True)
        print(
            f"  {s['trades']} trades, {s['wr']:.1f}% WR, "
            f"P&L ${s['pnl']:+,.0f} (${s['pnl_per_day']:+.1f}/day), "
            f"MaxDD ${s['max_dd']:,.0f}",
            flush=True,
        )

    print(f"\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
