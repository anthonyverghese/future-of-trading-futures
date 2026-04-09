"""
walk_forward.py — Walk-forward validation for both apps.

Unlike a single 75/25 split, walk-forward re-tunes parameters at every step
using only past data, then tests on the next unseen window. Gives honest
OOS estimates across the full history and catches regime drift.

Optimization: zone entries per day are T/S-independent. Outcomes per (T/S)
are risk-gate-independent. So we:
  1) Precompute entries once per day
  2) Precompute outcomes once per (T/S) per day
  3) Sweep risk gates as cheap replays of precomputed outcomes

This cuts the bot sweep from ~60K full day-sims to ~2K per-entry evaluations.

Usage:
    python -u walk_forward.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytz

_ET = pytz.timezone("America/New_York")

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache,
    load_cached_days,
    load_day,
    preprocess_day,
    simulate_day,
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

# Walk-forward params.
INITIAL_TRAIN_DAYS = 60
STEP_DAYS = 20

# EOD flatten time (hh:mm ET). Mirrors live bot: no new entries after this,
# and any open position is closed at this cutoff.
_eod_hhmm = 16 * 60 - BOT_EOD_FLATTEN_BUFFER_MIN
_EOD_FLATTEN_TIME = datetime.time(_eod_hhmm // 60, _eod_hhmm % 60)


def _eod_cutoff_ns(date: datetime.date) -> int:
    """Compute the EOD flatten cutoff for a day as ns since epoch.

    full_ts_ns is asi8 from a tz-aware ET index, so the cutoff must also
    be converted through ET localization → UTC epoch ns to be comparable.
    """
    dt = _ET.localize(datetime.datetime.combine(date, _EOD_FLATTEN_TIME))
    return int(dt.timestamp() * 1_000_000_000)


def _eod_cutoff_ns_for_day(dc) -> int:
    return _eod_cutoff_ns(dc.date)


TS_GRID = [
    (4.0, 8.0),
    (6.0, 12.0),
    (8.0, 16.0),
    (8.0, 20.0),
    (10.0, 20.0),
    (10.0, 25.0),
    (12.0, 20.0),
    (12.0, 25.0),
    (16.0, 25.0),
]
RISK_GRID: list[tuple[float | None, int | None]] = [
    (100.0, 3),
    (150.0, 3),
    (150.0, 4),
    (200.0, 3),
    (200.0, 4),
    (None, None),
]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Precompute entries per day (T/S-independent)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DayEntries:
    date: datetime.date
    # Per-entry arrays (chronologically sorted)
    global_idx: np.ndarray  # int
    level: list[str]
    entry_count: np.ndarray  # int
    ref_price: np.ndarray  # float
    entry_price: np.ndarray  # float
    direction: list[str]
    entry_ns: np.ndarray  # int64


def precompute_day_entries(dc: DayCache) -> DayEntries:
    prices = dc.post_ib_prices
    n = len(prices)
    all_entries: list[tuple[int, str, int, float]] = []
    for level_name, level_arr in [
        ("IBH", np.full(n, dc.ibh)),
        ("IBL", np.full(n, dc.ibl)),
        ("VWAP", dc.post_ib_vwaps),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo)),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi)),
    ]:
        use_current = level_name == "VWAP"
        entries = _run_zone_numpy(
            prices,
            level_arr,
            BOT_ENTRY_THRESHOLD,
            BOT_EXIT_THRESHOLD,
            use_current_exit=use_current,
        )
        for local_idx, ec, rp in entries:
            all_entries.append((dc.post_ib_start_idx + local_idx, level_name, ec, rp))
    all_entries.sort(key=lambda x: x[0])
    if not all_entries:
        return DayEntries(
            date=dc.date,
            global_idx=np.array([], dtype=np.int64),
            level=[],
            entry_count=np.array([], dtype=np.int64),
            ref_price=np.array([], dtype=np.float64),
            entry_price=np.array([], dtype=np.float64),
            direction=[],
            entry_ns=np.array([], dtype=np.int64),
        )
    gidx = np.array([e[0] for e in all_entries], dtype=np.int64)
    levels = [e[1] for e in all_entries]
    counts = np.array([e[2] for e in all_entries], dtype=np.int64)
    refs = np.array([e[3] for e in all_entries], dtype=np.float64)
    eprices = dc.full_prices[gidx]
    directions = ["up" if eprices[i] > refs[i] else "down" for i in range(len(gidx))]
    ens = dc.full_ts_ns[gidx]
    return DayEntries(
        date=dc.date,
        global_idx=gidx,
        level=levels,
        entry_count=counts,
        ref_price=refs,
        entry_price=eprices,
        direction=directions,
        entry_ns=ens,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Precompute outcomes per (T/S) per day
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DayOutcomes:
    """Per-entry outcomes for a specific (T/S) on one day."""

    date: datetime.date
    outcome: list[str]  # "win"/"loss"/"timeout"
    exit_ns: np.ndarray  # int64
    pnl_pts: np.ndarray  # float
    pnl_usd: np.ndarray  # float


def precompute_outcomes(
    de: DayEntries,
    dc: DayCache,
    target_pts: float,
    stop_pts: float,
    window_secs: int = 15 * 60,
) -> DayOutcomes:
    n = len(de.global_idx)
    outcomes: list[str] = []
    exit_ns = np.zeros(n, dtype=np.int64)
    pnl_pts = np.zeros(n, dtype=np.float64)
    # EOD cutoff: clip the 15-min window so stalled trades close before 4pm ET,
    # matching the live bot's pre-close flatten.
    eod_cutoff_ns = _eod_cutoff_ns_for_day(dc)
    for i in range(n):
        out, exit_idx, pnl = evaluate_bot_trade(
            int(de.global_idx[i]),
            float(de.ref_price[i]),
            de.direction[i],
            dc.full_ts_ns,
            dc.full_prices,
            target_pts,
            stop_pts,
            window_secs,
            eod_cutoff_ns,
        )
        outcomes.append(out)
        exit_ns[i] = int(dc.full_ts_ns[exit_idx])
        pnl_pts[i] = pnl
    return DayOutcomes(
        date=de.date,
        outcome=outcomes,
        exit_ns=exit_ns,
        pnl_pts=pnl_pts,
        pnl_usd=pnl_pts * MULTIPLIER,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Cheap replay with risk gates
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    outcome: str
    pnl_usd: float


def replay_with_risk(
    days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    outcomes_by_date: dict[datetime.date, DayOutcomes],
    daily_loss_usd: float | None,
    max_consec_losses: int | None,
) -> list[Trade]:
    trades: list[Trade] = []
    for date in days:
        de = entries_by_date.get(date)
        do = outcomes_by_date.get(date)
        if de is None or do is None or len(de.global_idx) == 0:
            continue
        # No new entries after EOD cutoff — matches live eod_flatten() which
        # sets _stopped_for_day=True at 15:58 ET.
        eod_ns = _eod_cutoff_ns(date)
        position_exit_ns = 0
        daily_pnl = 0.0
        consec = 0
        stopped = False
        for i in range(len(de.global_idx)):
            if stopped:
                break
            if int(de.entry_ns[i]) >= eod_ns:
                break  # entries are chronologically sorted
            if int(de.entry_ns[i]) < position_exit_ns:
                continue
            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            position_exit_ns = int(do.exit_ns[i])
            trades.append(
                Trade(date=date, level=de.level[i], outcome=outcome, pnl_usd=pnl)
            )
            daily_pnl += pnl
            # Match live broker: consecutive-loss counter is by $ sign, not label.
            # A timeout that closed at a loss should count toward the consec cap.
            if pnl < 0:
                consec += 1
            else:
                consec = 0
            if daily_loss_usd is not None and daily_pnl <= -daily_loss_usd:
                stopped = True
            if max_consec_losses is not None and consec >= max_consec_losses:
                stopped = True
    return trades


def trade_stats(trades: list[Trade]) -> tuple[float, int, int, float, float]:
    wins = sum(1 for t in trades if t.outcome == "win")
    losses = sum(1 for t in trades if t.outcome == "loss")
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


def timeout_stats(trades: list[Trade]) -> tuple[int, float, float]:
    """Return (count, total_pnl, avg_pnl) for timeout trades."""
    tos = [t for t in trades if t.outcome == "timeout"]
    if not tos:
        return 0, 0.0, 0.0
    total = sum(t.pnl_usd for t in tos)
    return len(tos), total, total / len(tos)


# ══════════════════════════════════════════════════════════════════════════════
# BOT WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WFWindow:
    train_range: tuple[datetime.date, datetime.date]
    test_range: tuple[datetime.date, datetime.date]
    train_days: int
    test_days: int
    chosen_ts: tuple[float, float]
    chosen_risk: tuple[float | None, int | None]
    train_pnl_per_day: float
    test_trades: list[Trade]


def walk_forward_bot(
    valid_days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    outcomes_by_ts: dict[tuple[float, float], dict[datetime.date, DayOutcomes]],
) -> list[WFWindow]:
    windows: list[WFWindow] = []
    n = len(valid_days)
    k = INITIAL_TRAIN_DAYS
    while k < n:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days:
            break
        best_ts = None
        best_risk: tuple[float | None, int | None] = (None, None)
        best_per_day = -float("inf")
        for ts in TS_GRID:
            obd = outcomes_by_ts[ts]
            for risk in RISK_GRID:
                trades = replay_with_risk(
                    train_days, entries_by_date, obd, risk[0], risk[1]
                )
                if not trades:
                    continue
                total = sum(t.pnl_usd for t in trades)
                per_day = total / len(train_days)
                if per_day > best_per_day:
                    best_per_day = per_day
                    best_ts = ts
                    best_risk = risk
        if best_ts is None:
            break
        test_trades = replay_with_risk(
            test_days,
            entries_by_date,
            outcomes_by_ts[best_ts],
            best_risk[0],
            best_risk[1],
        )
        windows.append(
            WFWindow(
                train_range=(train_days[0], train_days[-1]),
                test_range=(test_days[0], test_days[-1]),
                train_days=len(train_days),
                test_days=len(test_days),
                chosen_ts=best_ts,
                chosen_risk=best_risk,
                train_pnl_per_day=best_per_day,
                test_trades=test_trades,
            )
        )
        k += STEP_DAYS
    return windows


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════


def load_alerts_for_days(
    valid_days: list[datetime.date],
    day_caches: dict[datetime.date, DayCache],
) -> list[EnrichedAlert]:
    out: list[EnrichedAlert] = []
    cw = 0
    cl = 0
    for date in sorted(valid_days):
        dc = day_caches.get(date)
        if dc is None:
            continue
        day_alerts = simulate_day(dc)
        first_price = float(dc.post_ib_prices[0])
        day_alerts.sort(key=lambda a: a.alert_time)
        for a in day_alerts:
            if a.outcome not in ("correct", "incorrect"):
                continue
            if hasattr(a.alert_time, "astimezone") and a.alert_time.tzinfo:
                now_et = a.alert_time.astimezone(
                    datetime.timezone(datetime.timedelta(hours=-4))
                ).time()
            else:
                now_et = None
            tr = compute_tick_rate(dc.full_df, pd.Timestamp(a.alert_time))
            sm = a.entry_price - first_price
            out.append(
                EnrichedAlert(
                    date=date,
                    level=a.level,
                    direction=a.direction,
                    entry_count=a.level_test_count,
                    outcome=a.outcome,
                    entry_price=a.entry_price,
                    line_price=a.line_price,
                    alert_time=a.alert_time,
                    now_et=now_et,
                    tick_rate=tr,
                    session_move_pts=sm,
                    consecutive_wins=cw,
                    consecutive_losses=cl,
                )
            )
            if a.outcome == "correct":
                cw += 1
                cl = 0
            else:
                cl += 1
                cw = 0
    return out


def fit_weights(train_alerts: list[EnrichedAlert]) -> Weights:
    if not train_alerts:
        return Weights()
    w_count = sum(1 for a in train_alerts if a.outcome == "correct")
    baseline = w_count / len(train_alerts) * 100

    def wr(fn) -> float:
        b = [a for a in train_alerts if fn(a)]
        if len(b) < 30:
            return baseline
        return sum(1 for a in b if a.outcome == "correct") / len(b) * 100

    opt = Weights()
    opt.level_fib_hi = suggest_weight(
        wr(lambda a: a.level == "FIB_EXT_HI_1.272"), baseline
    )
    opt.level_ibl = suggest_weight(wr(lambda a: a.level == "IBL"), baseline)
    opt.level_fib_lo = suggest_weight(
        wr(lambda a: a.level == "FIB_EXT_LO_1.272"), baseline
    )
    opt.level_vwap = suggest_weight(wr(lambda a: a.level == "VWAP"), baseline)
    opt.level_ibh = suggest_weight(wr(lambda a: a.level == "IBH"), baseline)
    opt.combo_fib_hi_up = suggest_weight(
        wr(lambda a: a.level == "FIB_EXT_HI_1.272" and a.direction == "up"), baseline
    )
    opt.combo_fib_lo_down = suggest_weight(
        wr(lambda a: a.level == "FIB_EXT_LO_1.272" and a.direction == "down"), baseline
    )
    opt.combo_ibl_down = suggest_weight(
        wr(lambda a: a.level == "IBL" and a.direction == "down"), baseline
    )
    opt.combo_vwap_up = suggest_weight(
        wr(lambda a: a.level == "VWAP" and a.direction == "up"), baseline
    )
    opt.combo_ibh_up = suggest_weight(
        wr(lambda a: a.level == "IBH" and a.direction == "up"), baseline
    )
    opt.combo_ibl_up = suggest_weight(
        wr(lambda a: a.level == "IBL" and a.direction == "up"), baseline
    )
    opt.combo_fib_lo_up = suggest_weight(
        wr(lambda a: a.level == "FIB_EXT_LO_1.272" and a.direction == "up"), baseline
    )
    opt.combo_fib_hi_down = suggest_weight(
        wr(lambda a: a.level == "FIB_EXT_HI_1.272" and a.direction == "down"), baseline
    )
    opt.combo_vwap_down = suggest_weight(
        wr(lambda a: a.level == "VWAP" and a.direction == "down"), baseline
    )
    opt.time_power_hour = suggest_weight(
        wr(lambda a: a.now_et and a.now_et.hour * 60 + a.now_et.minute >= 15 * 60),
        baseline,
    )
    opt.tick_sweet_spot = suggest_weight(
        wr(lambda a: a.tick_rate is not None and 1750 <= a.tick_rate < 2000), baseline
    )
    opt.test_1 = suggest_weight(wr(lambda a: a.entry_count == 1), baseline)
    opt.test_2 = suggest_weight(wr(lambda a: a.entry_count == 2), baseline)
    opt.test_3 = suggest_weight(wr(lambda a: a.entry_count == 3), baseline)
    opt.test_5 = suggest_weight(wr(lambda a: a.entry_count == 5), baseline)
    opt.move_sweet_green = suggest_weight(
        wr(lambda a: a.session_move_pts is not None and 10 < a.session_move_pts <= 20),
        baseline,
    )
    opt.move_sweet_red = suggest_weight(
        wr(
            lambda a: a.session_move_pts is not None and -20 < a.session_move_pts <= -10
        ),
        baseline,
    )
    opt.move_strong_red = suggest_weight(
        wr(lambda a: a.session_move_pts is not None and a.session_move_pts <= -50),
        baseline,
    )
    opt.move_near_zero_green = suggest_weight(
        wr(lambda a: a.session_move_pts is not None and 0 < a.session_move_pts <= 10),
        baseline,
    )
    opt.move_strong_green = suggest_weight(
        wr(lambda a: a.session_move_pts is not None and a.session_move_pts > 50),
        baseline,
    )
    opt.streak_win_bonus = suggest_weight(
        wr(lambda a: a.consecutive_wins >= 2), baseline
    )
    opt.streak_loss_penalty = suggest_weight(
        wr(lambda a: a.consecutive_losses >= 2), baseline
    )
    return opt


@dataclass
class HumanWFWindow:
    train_range: tuple[datetime.date, datetime.date]
    test_range: tuple[datetime.date, datetime.date]
    train_days: int
    test_days: int
    scored: list[tuple[EnrichedAlert, int]]


def walk_forward_human(
    all_alerts: list[EnrichedAlert],
    valid_days: list[datetime.date],
) -> list[HumanWFWindow]:
    by_date: dict[datetime.date, list[EnrichedAlert]] = {}
    for a in all_alerts:
        by_date.setdefault(a.date, []).append(a)
    out: list[HumanWFWindow] = []
    n = len(valid_days)
    k = INITIAL_TRAIN_DAYS
    while k < n:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days:
            break
        train_alerts = [a for d in train_days for a in by_date.get(d, [])]
        test_alerts = [a for d in test_days for a in by_date.get(d, [])]
        w = fit_weights(train_alerts)
        scored = [(a, score_alert(a, w)) for a in test_alerts]
        out.append(
            HumanWFWindow(
                train_range=(train_days[0], train_days[-1]),
                test_range=(test_days[0], test_days[-1]),
                train_days=len(train_days),
                test_days=len(test_days),
                scored=scored,
            )
        )
        k += STEP_DAYS
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    print("=" * 78, flush=True)
    print("  WALK-FORWARD VALIDATION", flush=True)
    print(
        f"  Initial train: {INITIAL_TRAIN_DAYS} days, step: {STEP_DAYS} days",
        flush=True,
    )
    print("=" * 78, flush=True)

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

    # Stage 1: precompute entries per day
    t1 = time.time()
    print(f"\n  Stage 1: precomputing zone entries for {num_days} days...", flush=True)
    entries_by_date: dict[datetime.date, DayEntries] = {}
    total_entries = 0
    for date in valid_days:
        de = precompute_day_entries(day_caches[date])
        entries_by_date[date] = de
        total_entries += len(de.global_idx)
    print(f"  {total_entries} total zone entries in {time.time()-t1:.1f}s", flush=True)

    # Stage 2: precompute outcomes per (T/S) per day
    t2 = time.time()
    print(
        f"\n  Stage 2: precomputing outcomes for {len(TS_GRID)} T/S configs...",
        flush=True,
    )
    outcomes_by_ts: dict[tuple[float, float], dict[datetime.date, DayOutcomes]] = {}
    for ts in TS_GRID:
        per_day: dict[datetime.date, DayOutcomes] = {}
        for date in valid_days:
            per_day[date] = precompute_outcomes(
                entries_by_date[date], day_caches[date], ts[0], ts[1]
            )
        outcomes_by_ts[ts] = per_day
        print(f"    T/S={int(ts[0])}/{int(ts[1])} done", flush=True)
    print(f"  Stage 2 done in {time.time()-t2:.1f}s", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # BOT WALK-FORWARD
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  BOT: walk-forward T/S + risk-limit selection", flush=True)
    print("=" * 78, flush=True)
    t3 = time.time()
    wf = walk_forward_bot(valid_days, entries_by_date, outcomes_by_ts)
    print(f"  Ran {len(wf)} windows in {time.time()-t3:.1f}s\n", flush=True)

    print(
        f"  {'Train end':<12} {'Test range':<24} {'T/S':>6} {'Risk':>11} {'Tr':>4} {'WR%':>6} {'P&L':>8} {'$/day':>7}",
        flush=True,
    )
    print(
        f"  {'-'*12} {'-'*24} {'-'*6} {'-'*11} {'-'*4} {'-'*6} {'-'*8} {'-'*7}",
        flush=True,
    )
    ts_counts: dict[tuple, int] = {}
    risk_counts: dict[tuple, int] = {}
    for w in wf:
        total, wins, losses, wr, _ = trade_stats(w.test_trades)
        per_day = total / w.test_days
        ts_str = f"{int(w.chosen_ts[0])}/{int(w.chosen_ts[1])}"
        r_str = (
            f"${int(w.chosen_risk[0])}/{w.chosen_risk[1]}"
            if w.chosen_risk[0]
            else "unrestr"
        )
        ts_counts[w.chosen_ts] = ts_counts.get(w.chosen_ts, 0) + 1
        risk_counts[w.chosen_risk] = risk_counts.get(w.chosen_risk, 0) + 1
        print(
            f"  {str(w.train_range[1]):<12} {str(w.test_range[0])}→{str(w.test_range[1])} "
            f"{ts_str:>6} {r_str:>11} {len(w.test_trades):>4} "
            f"{wr:>5.1f}% ${total:>+6,.0f} ${per_day:>+5.1f}",
            flush=True,
        )

    all_test = [t for w in wf for t in w.test_trades]
    total_days = sum(w.test_days for w in wf)
    total, wins, losses, wr, dd = trade_stats(all_test)
    print(f"\n  AGGREGATE OOS (all test windows combined):", flush=True)
    print(f"    Days:    {total_days}", flush=True)
    print(
        f"    Trades:  {len(all_test)} ({len(all_test)/total_days:.1f}/day)", flush=True
    )
    print(f"    Record:  {wins}W / {losses}L = {wr:.1f}% WR", flush=True)
    print(f"    P&L:     ${total:+,.0f} (${total/total_days:+.2f}/day)", flush=True)
    print(f"    Max DD:  ${dd:,.0f}", flush=True)
    to_n, to_total, to_avg = timeout_stats(all_test)
    if to_n:
        to_pct = to_n / len(all_test) * 100
        print(
            f"    Timeouts: {to_n} ({to_pct:.1f}% of trades), "
            f"avg ${to_avg:+.2f}, total ${to_total:+,.0f}",
            flush=True,
        )

    print(f"\n  Chosen T/S distribution:", flush=True)
    for ts, cnt in sorted(ts_counts.items(), key=lambda x: -x[1]):
        print(
            f"    {int(ts[0]):>2}/{int(ts[1]):<2}  {cnt:>2} times ({cnt/len(wf)*100:>3.0f}%)",
            flush=True,
        )
    print(f"\n  Chosen risk distribution:", flush=True)
    for r, cnt in sorted(risk_counts.items(), key=lambda x: -x[1]):
        label = f"${int(r[0])}/{r[1]}" if r[0] else "unrestricted"
        print(f"    {label:<15}  {cnt:>2} times ({cnt/len(wf)*100:>3.0f}%)", flush=True)

    # COMPARISON: fixed 12/25 + $150/3 (current live bot) over same OOS windows
    print(
        f"\n  COMPARISON — Fixed current live params (12/25, $150/3) on same OOS windows:",
        flush=True,
    )
    fixed_all: list[Trade] = []
    for w in wf:
        test_days = valid_days[
            valid_days.index(w.test_range[0]) : valid_days.index(w.test_range[1]) + 1
        ]
        fixed_all.extend(
            replay_with_risk(
                test_days, entries_by_date, outcomes_by_ts[(12.0, 25.0)], 150.0, 3
            )
        )
    total_f, wf_w, wf_l, wr_f, dd_f = trade_stats(fixed_all)
    print(
        f"    Trades:  {len(fixed_all)} ({len(fixed_all)/total_days:.1f}/day)",
        flush=True,
    )
    print(f"    Record:  {wf_w}W / {wf_l}L = {wr_f:.1f}% WR", flush=True)
    print(f"    P&L:     ${total_f:+,.0f} (${total_f/total_days:+.2f}/day)", flush=True)
    print(f"    Max DD:  ${dd_f:,.0f}", flush=True)
    to_n_f, to_tot_f, to_avg_f = timeout_stats(fixed_all)
    if to_n_f:
        print(
            f"    Timeouts: {to_n_f} ({to_n_f/len(fixed_all)*100:.1f}% of trades), "
            f"avg ${to_avg_f:+.2f}, total ${to_tot_f:+,.0f}",
            flush=True,
        )

    # Also test each fixed T/S config on full OOS period (no walk-forward)
    print(
        f"\n  FIXED CONFIG GRID (fixed T/S, $150/3, full OOS period, no retraining):",
        flush=True,
    )
    print(
        f"  {'T/S':>6} {'Trades':>7} {'WR%':>7} {'P&L':>8} {'$/day':>7} {'MaxDD':>8}",
        flush=True,
    )
    oos_days = [
        d
        for w in wf
        for d in valid_days[
            valid_days.index(w.test_range[0]) : valid_days.index(w.test_range[1]) + 1
        ]
    ]
    for ts in TS_GRID:
        tr = replay_with_risk(oos_days, entries_by_date, outcomes_by_ts[ts], 150.0, 3)
        t_total, t_w, t_l, t_wr, t_dd = trade_stats(tr)
        marker = " <- current" if ts == (12.0, 25.0) else ""
        print(
            f"  {int(ts[0]):>2}/{int(ts[1]):<2}  {len(tr):>7} {t_wr:>6.1f}% ${t_total:>+6,.0f} ${t_total/total_days:>+5.1f}  ${t_dd:>6,.0f}{marker}",
            flush=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # HUMAN WALK-FORWARD
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  HUMAN: walk-forward scoring-weight tuning", flush=True)
    print("=" * 78, flush=True)
    t4 = time.time()
    print("  Loading alerts with factors...", flush=True)
    all_alerts = load_alerts_for_days(valid_days, day_caches)
    print(
        f"  Loaded {len(all_alerts)} decided alerts in {time.time()-t4:.1f}s",
        flush=True,
    )

    t5 = time.time()
    hwf = walk_forward_human(all_alerts, valid_days)
    print(f"  Ran {len(hwf)} windows in {time.time()-t5:.1f}s", flush=True)

    all_scored = [(a, s) for w in hwf for a, s in w.scored]
    oos_days_h = sum(w.test_days for w in hwf)
    oos_alerts = [a for a, _ in all_scored]
    oos_w = sum(1 for a in oos_alerts if a.outcome == "correct")
    print(
        f"\n  OOS alerts: {len(oos_alerts)} ({oos_w} correct, baseline {oos_w/len(oos_alerts)*100:.1f}% WR)",
        flush=True,
    )

    print(
        f"\n  Walk-forward OOS sweep (weights retrained every {STEP_DAYS} days):",
        flush=True,
    )
    print(
        f"  {'Thr':>4} {'W':>5} {'L':>5} {'Tot':>6} {'WR%':>7} {'/day':>6}", flush=True
    )
    if all_scored:
        min_s = min(s for _, s in all_scored)
        max_s = max(s for _, s in all_scored)
        for thr in range(min_s, max_s + 1):
            p = [(a, s) for a, s in all_scored if s >= thr]
            if not p:
                continue
            w = sum(1 for a, _ in p if a.outcome == "correct")
            marker = (
                " <- current (>=5)"
                if thr == 5
                else (" <- MIN_SCORE (>=4)" if thr == 4 else "")
            )
            print(
                f"  {thr:>4} {w:>5} {len(p)-w:>5} {len(p):>6} {w/len(p)*100:>6.1f}% {len(p)/oos_days_h:>5.1f}{marker}",
                flush=True,
            )

    # Compare current fixed weights on same OOS alerts
    cur = Weights()
    fixed = [(a, score_alert(a, cur)) for a in oos_alerts]
    print(f"\n  COMPARISON — current FIXED weights on same OOS alerts:", flush=True)
    print(
        f"  {'Thr':>4} {'W':>5} {'L':>5} {'Tot':>6} {'WR%':>7} {'/day':>6}", flush=True
    )
    min_s = min(s for _, s in fixed)
    max_s = max(s for _, s in fixed)
    for thr in range(min_s, max_s + 1):
        p = [(a, s) for a, s in fixed if s >= thr]
        if not p:
            continue
        w = sum(1 for a, _ in p if a.outcome == "correct")
        marker = (
            " <- current (>=5)"
            if thr == 5
            else (" <- MIN_SCORE (>=4)" if thr == 4 else "")
        )
        print(
            f"  {thr:>4} {w:>5} {len(p)-w:>5} {len(p):>6} {w/len(p)*100:>6.1f}% {len(p)/oos_days_h:>5.1f}{marker}",
            flush=True,
        )

    print(f"\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
