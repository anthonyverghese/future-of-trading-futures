"""
bot_pct_backtest.py — Percentage-based bot backtest with factor analysis.

All thresholds are expressed as percentages of price rather than fixed
points, so results aren't biased by the ~23% price increase across the
319-day backtest period (MNQ 21K → 26K).

Three stages:
  1. Factor analysis: win rates per scoring factor bucket at 1-pt bot
     entries, to derive bot-specific weights (not borrowed from human).
  2. Full walk-forward with retrained weights, sweeping %-based T/S,
     score thresholds, max entries per level.
  3. Recent regime (last 60 days) analysis.

Usage:
    python -u bot_pct_backtest.py
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
    precompute_day_entries,
    _eod_cutoff_ns,
    INITIAL_TRAIN_DAYS,
    STEP_DAYS,
)

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# PERCENTAGE-BASED OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PctOutcomes:
    """Per-entry outcomes for a %-based T/S on one day."""

    date: datetime.date
    outcome: list[str]
    exit_ns: np.ndarray
    pnl_pts: np.ndarray
    pnl_usd: np.ndarray


def precompute_pct_outcomes(
    de: DayEntries,
    dc: DayCache,
    target_pct: float,
    stop_pct: float,
    window_secs: int = 15 * 60,
) -> PctOutcomes:
    """Evaluate bot entries with %-of-price T/S."""
    n = len(de.global_idx)
    outcomes: list[str] = []
    exit_ns = np.zeros(n, dtype=np.int64)
    pnl_pts = np.zeros(n, dtype=np.float64)
    eod_cutoff_ns = _eod_cutoff_ns(dc.date)

    for i in range(n):
        gidx = int(de.global_idx[i])
        line_price = float(de.ref_price[i])
        direction = de.direction[i]

        # Convert % to points based on line price.
        t_pts = line_price * target_pct / 100.0
        s_pts = line_price * stop_pct / 100.0

        out, exit_idx, pnl = evaluate_bot_trade(
            gidx,
            line_price,
            direction,
            dc.full_ts_ns,
            dc.full_prices,
            t_pts,
            s_pts,
            window_secs,
            eod_cutoff_ns,
        )
        outcomes.append(out)
        exit_ns[i] = int(dc.full_ts_ns[exit_idx])
        pnl_pts[i] = pnl

    return PctOutcomes(
        date=de.date,
        outcome=outcomes,
        exit_ns=exit_ns,
        pnl_pts=pnl_pts,
        pnl_usd=pnl_pts * MULTIPLIER,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENRICHED ENTRIES (same as bot_full_backtest but included here for
# self-containment)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BotEntry:
    idx: int
    level: str
    direction: str
    entry_count: int
    entry_price: float
    line_price: float
    entry_ns: int
    now_et: datetime.time | None = None
    tick_rate: float | None = None
    session_move_pts: float | None = None
    range_30m: float | None = None
    # Percentage-based session move (for %-based scoring).
    session_move_pct: float | None = None


def enrich_entries(de: DayEntries, dc: DayCache) -> list[BotEntry]:
    first_price = float(dc.post_ib_prices[0])
    enriched: list[BotEntry] = []

    for i in range(len(de.global_idx)):
        gidx = int(de.global_idx[i])
        entry_ns = int(de.entry_ns[i])
        entry_price = float(de.entry_price[i])

        ts_pd = pd.Timestamp(entry_ns, unit="ns", tz="UTC").tz_convert(_ET)
        now_et = ts_pd.time()
        tick_rate = compute_tick_rate(dc.full_df, ts_pd)
        session_move = entry_price - first_price

        # 30-min range.
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

        # %-based session move.
        session_move_pct = (
            session_move / first_price * 100 if first_price > 0 else None
        )

        enriched.append(
            BotEntry(
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
                session_move_pct=session_move_pct,
            )
        )

    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR ANALYSIS: win rates per bucket using bot entries
# ══════════════════════════════════════════════════════════════════════════════


def factor_analysis(
    all_entries: list[tuple[BotEntry, str]],  # (entry, outcome)
) -> None:
    """Print win rates per factor bucket for bot entries."""
    total = len(all_entries)
    if total == 0:
        print("  No entries for factor analysis.")
        return

    baseline_wins = sum(1 for _, o in all_entries if o == "win")
    baseline_wr = baseline_wins / total * 100
    print(f"\n  Baseline: {baseline_wins}W / {total} = {baseline_wr:.1f}% WR\n")

    def wr_line(label, fn):
        subset = [(e, o) for e, o in all_entries if fn(e)]
        if len(subset) < 30:
            return
        w = sum(1 for _, o in subset if o == "win")
        wr = w / len(subset) * 100
        delta = wr - baseline_wr
        wt = suggest_weight(wr, baseline_wr)
        print(
            f"  {label:<45s} {w:>5}W / {len(subset):>5} "
            f"= {wr:>5.1f}% ({delta:>+5.1f}pp)  wt={wt:>+d}"
        )

    # Level quality.
    print("  --- Level quality ---")
    for lv in ["IBH", "IBL", "VWAP", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]:
        wr_line(lv, lambda e, l=lv: e.level == l)

    # Direction × level.
    print("\n  --- Direction × level ---")
    for lv in ["IBH", "IBL", "VWAP", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]:
        for d in ["up", "down"]:
            wr_line(
                f"{lv} × {d}",
                lambda e, l=lv, dr=d: e.level == l and e.direction == dr,
            )

    # Time of day.
    print("\n  --- Time of day ---")
    wr_line(
        "Post-IB (10:31-11:30)",
        lambda e: e.now_et and 10 * 60 + 31 <= e.now_et.hour * 60 + e.now_et.minute < 11 * 60 + 30,
    )
    wr_line(
        "Late morning (11:30-13:00)",
        lambda e: e.now_et and 11 * 60 + 30 <= e.now_et.hour * 60 + e.now_et.minute < 13 * 60,
    )
    wr_line(
        "Afternoon (13:00-15:00)",
        lambda e: e.now_et and 13 * 60 <= e.now_et.hour * 60 + e.now_et.minute < 15 * 60,
    )
    wr_line(
        "Power hour (15:00-16:00)",
        lambda e: e.now_et and e.now_et.hour * 60 + e.now_et.minute >= 15 * 60,
    )

    # Entry count.
    print("\n  --- Entry count (test #) ---")
    for ec in [1, 2, 3, 4, 5]:
        wr_line(f"Test #{ec}", lambda e, c=ec: e.entry_count == c)
    wr_line("Test #6+", lambda e: e.entry_count >= 6)

    # Tick rate.
    print("\n  --- Tick rate (trades/min in 3-min window) ---")
    for lo, hi, label in [
        (0, 500, "<500"),
        (500, 1000, "500-1000"),
        (1000, 1500, "1000-1500"),
        (1500, 2000, "1500-2000"),
        (2000, 2500, "2000-2500"),
        (2500, 99999, "2500+"),
    ]:
        wr_line(
            f"Tick rate {label}",
            lambda e, l=lo, h=hi: e.tick_rate is not None and l <= e.tick_rate < h,
        )

    # Session move (%-based).
    print("\n  --- Session move (% of price) ---")
    for lo, hi, label in [
        (-999, -0.20, "< -0.20%"),
        (-0.20, -0.05, "-0.20% to -0.05%"),
        (-0.05, 0.00, "-0.05% to 0%"),
        (0.00, 0.05, "0% to +0.05%"),
        (0.05, 0.20, "+0.05% to +0.20%"),
        (0.20, 999, "> +0.20%"),
    ]:
        wr_line(
            f"Session move {label}",
            lambda e, l=lo, h=hi: e.session_move_pct is not None and l <= e.session_move_pct < h,
        )

    # 30-min range (%-based: range / entry_price * 100).
    print("\n  --- 30-min range (% of price) ---")
    for lo, hi, label in [
        (0.0, 0.15, "<0.15%"),
        (0.15, 0.25, "0.15-0.25%"),
        (0.25, 0.35, "0.25-0.35%"),
        (0.35, 0.50, "0.35-0.50%"),
        (0.50, 999, ">0.50%"),
    ]:
        wr_line(
            f"30m range {label}",
            lambda e, l=lo, h=hi: (
                e.range_30m is not None
                and e.entry_price > 0
                and l <= e.range_30m / e.entry_price * 100 < h
            ),
        )

    # IB range (%-based: ib_range / entry_price * 100).
    print("\n  --- IB range (% of price) ---")
    # We don't have IB range directly on entries, but can derive from level
    # prices for fib entries.
    print("  (captured via T/S normalization in walk-forward)")


# ══════════════════════════════════════════════════════════════════════════════
# BOT-SPECIFIC WEIGHT FITTING
# ══════════════════════════════════════════════════════════════════════════════


def fit_bot_weights(
    entries_with_outcomes: list[tuple[BotEntry, str]],
) -> Weights:
    """Derive scoring weights from bot entry outcomes (not human alerts)."""
    if not entries_with_outcomes:
        return Weights()

    total = len(entries_with_outcomes)
    w_count = sum(1 for _, o in entries_with_outcomes if o == "win")
    baseline = w_count / total * 100

    def wr(fn) -> float:
        b = [(e, o) for e, o in entries_with_outcomes if fn(e)]
        if len(b) < 30:
            return baseline
        return sum(1 for _, o in b if o == "win") / len(b) * 100

    opt = Weights()
    # Level quality.
    opt.level_fib_hi = suggest_weight(
        wr(lambda e: e.level == "FIB_EXT_HI_1.272"), baseline
    )
    opt.level_ibl = suggest_weight(wr(lambda e: e.level == "IBL"), baseline)
    opt.level_fib_lo = suggest_weight(
        wr(lambda e: e.level == "FIB_EXT_LO_1.272"), baseline
    )
    opt.level_vwap = suggest_weight(wr(lambda e: e.level == "VWAP"), baseline)
    opt.level_ibh = suggest_weight(wr(lambda e: e.level == "IBH"), baseline)

    # Direction × level combos.
    opt.combo_fib_hi_up = suggest_weight(
        wr(lambda e: e.level == "FIB_EXT_HI_1.272" and e.direction == "up"),
        baseline,
    )
    opt.combo_fib_lo_down = suggest_weight(
        wr(lambda e: e.level == "FIB_EXT_LO_1.272" and e.direction == "down"),
        baseline,
    )
    opt.combo_ibl_down = suggest_weight(
        wr(lambda e: e.level == "IBL" and e.direction == "down"), baseline
    )
    opt.combo_vwap_up = suggest_weight(
        wr(lambda e: e.level == "VWAP" and e.direction == "up"), baseline
    )
    opt.combo_ibh_up = suggest_weight(
        wr(lambda e: e.level == "IBH" and e.direction == "up"), baseline
    )
    opt.combo_ibl_up = suggest_weight(
        wr(lambda e: e.level == "IBL" and e.direction == "up"), baseline
    )
    opt.combo_fib_lo_up = suggest_weight(
        wr(lambda e: e.level == "FIB_EXT_LO_1.272" and e.direction == "up"),
        baseline,
    )
    opt.combo_fib_hi_down = suggest_weight(
        wr(lambda e: e.level == "FIB_EXT_HI_1.272" and e.direction == "down"),
        baseline,
    )
    opt.combo_vwap_down = suggest_weight(
        wr(lambda e: e.level == "VWAP" and e.direction == "down"), baseline
    )

    # Time of day.
    opt.time_power_hour = suggest_weight(
        wr(lambda e: e.now_et and e.now_et.hour * 60 + e.now_et.minute >= 15 * 60),
        baseline,
    )

    # Tick rate (find best bucket).
    opt.tick_sweet_spot = suggest_weight(
        wr(lambda e: e.tick_rate is not None and 1500 <= e.tick_rate < 2000),
        baseline,
    )

    # Entry count.
    opt.test_1 = suggest_weight(wr(lambda e: e.entry_count == 1), baseline)
    opt.test_2 = suggest_weight(wr(lambda e: e.entry_count == 2), baseline)
    opt.test_3 = suggest_weight(wr(lambda e: e.entry_count == 3), baseline)
    opt.test_5 = suggest_weight(wr(lambda e: e.entry_count >= 5), baseline)

    # Session move (%-based buckets mapped to the existing weight fields).
    # Use %-based thresholds that roughly correspond to the point-based ones
    # at the median price (~23K): 10pts ≈ 0.04%, 20pts ≈ 0.09%, 50pts ≈ 0.22%
    opt.move_sweet_green = suggest_weight(
        wr(lambda e: e.session_move_pct is not None and 0.04 < e.session_move_pct <= 0.09),
        baseline,
    )
    opt.move_sweet_red = suggest_weight(
        wr(lambda e: e.session_move_pct is not None and -0.09 < e.session_move_pct <= -0.04),
        baseline,
    )
    opt.move_strong_red = suggest_weight(
        wr(lambda e: e.session_move_pct is not None and e.session_move_pct <= -0.22),
        baseline,
    )
    opt.move_near_zero_green = suggest_weight(
        wr(lambda e: e.session_move_pct is not None and 0 < e.session_move_pct <= 0.04),
        baseline,
    )
    opt.move_strong_green = suggest_weight(
        wr(lambda e: e.session_move_pct is not None and e.session_move_pct > 0.22),
        baseline,
    )

    # Streak: we can't compute from a flat list (needs replay order), so
    # keep default weights.

    return opt


def score_bot_entry(e: BotEntry, w: Weights, consec_wins: int = 0, consec_losses: int = 0) -> int:
    """Score a bot entry using bot-derived weights."""
    # Use %-based session move for scoring. Map to the point-based fields
    # that score_alert expects by using the %-based thresholds.
    # We build an EnrichedAlert but override session_move_pts with a
    # synthetic value that maps our %-based buckets to the existing
    # scoring function's point-based thresholds.
    #
    # This is a bit hacky — ideally we'd refactor score_alert to accept
    # %-based inputs, but for now we map:
    #   0.04% → 10 pts, 0.09% → 20 pts, 0.22% → 50 pts
    if e.session_move_pct is not None:
        # Map %-based move to synthetic pts that hit the same score_alert buckets.
        pct = e.session_move_pct
        if pct <= -0.22:
            synthetic_move = -55.0
        elif -0.09 < pct <= -0.04:
            synthetic_move = -15.0
        elif 0 < pct <= 0.04:
            synthetic_move = 5.0
        elif 0.04 < pct <= 0.09:
            synthetic_move = 15.0
        elif pct > 0.22:
            synthetic_move = 55.0
        else:
            synthetic_move = 0.0
    else:
        synthetic_move = None

    ea = EnrichedAlert(
        date=datetime.date.today(),
        level=e.level,
        direction=e.direction,
        entry_count=e.entry_count,
        outcome="correct",
        entry_price=e.entry_price,
        line_price=e.line_price,
        alert_time=datetime.datetime.now(),
        now_et=e.now_et,
        tick_rate=e.tick_rate,
        session_move_pts=synthetic_move,
        consecutive_wins=consec_wins,
        consecutive_losses=consec_losses,
    )
    score = score_alert(ea, w)

    # Volatility penalty (%-based: 30m range > 0.30% of price).
    if (
        e.range_30m is not None
        and e.entry_price > 0
        and e.range_30m / e.entry_price * 100 > 0.30
    ):
        score -= 2

    return score


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BotTrade:
    date: datetime.date
    level: str
    direction: str
    outcome: str
    pnl_usd: float
    score: int


def replay(
    days: list[datetime.date],
    entries_by_date: dict[datetime.date, DayEntries],
    enriched_by_date: dict[datetime.date, list[BotEntry]],
    outcomes_by_date: dict[datetime.date, PctOutcomes],
    weights: Weights,
    min_score: int,
    max_entries_per_level: int,
    daily_loss_usd: float | None,
    max_consec_losses: int | None,
    excluded_combos: set[tuple[str, str]] | None = None,
) -> list[BotTrade]:
    trades: list[BotTrade] = []
    cw, cl = 0, 0

    for date in days:
        de = entries_by_date.get(date)
        do = outcomes_by_date.get(date)
        enriched = enriched_by_date.get(date)
        if de is None or do is None or enriched is None or len(de.global_idx) == 0:
            continue

        eod_ns = _eod_cutoff_ns(date)
        pos_exit_ns = 0
        daily_pnl = 0.0
        daily_consec = 0
        stopped = False
        lc: dict[str, int] = {}

        for eb in enriched:
            if stopped:
                break
            if eb.entry_ns >= eod_ns:
                break
            if eb.entry_ns < pos_exit_ns:
                continue
            if excluded_combos and (eb.level, eb.direction) in excluded_combos:
                continue

            lv_count = lc.get(eb.level, 0)
            if lv_count >= max_entries_per_level:
                continue

            score = score_bot_entry(eb, weights, cw, cl)
            if score < min_score:
                continue

            i = eb.idx
            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            pos_exit_ns = int(do.exit_ns[i])
            lc[eb.level] = lv_count + 1

            trades.append(
                BotTrade(
                    date=date,
                    level=eb.level,
                    direction=eb.direction,
                    outcome=outcome,
                    pnl_usd=pnl,
                    score=score,
                )
            )
            daily_pnl += pnl

            if pnl < 0:
                daily_consec += 1
                cw = 0
                cl += 1
            else:
                daily_consec = 0
                cl = 0
                cw += 1

            if daily_loss_usd is not None and daily_pnl <= -daily_loss_usd:
                stopped = True
            if max_consec_losses is not None and daily_consec >= max_consec_losses:
                stopped = True

    return trades


def summarize(trades: list[BotTrade], n_days: int, label: str = "") -> str:
    if not trades:
        return f"  {label:45s} no trades"
    w = sum(1 for t in trades if t.outcome == "win")
    l = sum(1 for t in trades if t.outcome == "loss")
    to = sum(1 for t in trades if t.outcome == "timeout")
    d = w + l
    wr = w / d * 100 if d else 0
    pnl = sum(t.pnl_usd for t in trades)
    ppd = pnl / n_days if n_days else 0
    eq = STARTING_BALANCE
    peak = eq
    dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return (
        f"  {label:45s} "
        f"{len(trades):>4} ({len(trades)/n_days:.1f}/d) "
        f"{w}W/{l}L/{to}T = {wr:>5.1f}%  "
        f"${pnl:>+8,.0f} (${ppd:>+6.1f}/d)  "
        f"DD ${dd:>6,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# T/S GRIDS
# ══════════════════════════════════════════════════════════════════════════════

# Percentage-based T/S (% of entry price).
PCT_TS_GRID = [
    (0.030, 0.075),  # ~7.5/19 at 25K
    (0.035, 0.090),  # ~9/22 at 25K
    (0.040, 0.100),  # ~10/25 at 25K (≈ current)
    (0.045, 0.100),  # ~11/25
    (0.040, 0.120),  # ~10/30
    (0.050, 0.100),  # ~12.5/25
    (0.050, 0.120),  # ~12.5/30
]

# IB-range-normalized T/S.
IB_NORM_GRID = [
    (0.05, 0.15),
    (0.07, 0.15),
    (0.07, 0.20),
    (0.10, 0.20),
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    t0 = time.time()
    print("=" * 78, flush=True)
    print("  BOT %-BASED BACKTEST — Factor analysis + walk-forward", flush=True)
    print("=" * 78, flush=True)

    # Load days.
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
    N = len(valid_days)
    print(f"  {N} days in {time.time()-t0:.1f}s ({valid_days[0]} → {valid_days[-1]})", flush=True)

    # Precompute entries.
    t1 = time.time()
    print(f"\n  Precomputing entries + factors...", flush=True)
    entries_by_date: dict[datetime.date, DayEntries] = {}
    enriched_by_date: dict[datetime.date, list[BotEntry]] = {}
    n_entries = 0
    for i, date in enumerate(valid_days):
        de = precompute_day_entries(day_caches[date])
        entries_by_date[date] = de
        enriched_by_date[date] = enrich_entries(de, day_caches[date])
        n_entries += len(de.global_idx)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N}...", flush=True)
    print(f"  {n_entries} entries in {time.time()-t1:.1f}s", flush=True)

    # Precompute outcomes for %-based T/S.
    t2 = time.time()
    print(f"\n  Precomputing %-based outcomes ({len(PCT_TS_GRID)} configs)...", flush=True)
    outcomes_by_pct: dict[tuple[float, float], dict[datetime.date, PctOutcomes]] = {}
    for ts in PCT_TS_GRID:
        per_day: dict[datetime.date, PctOutcomes] = {}
        for date in valid_days:
            per_day[date] = precompute_pct_outcomes(
                entries_by_date[date], day_caches[date], ts[0], ts[1]
            )
        outcomes_by_pct[ts] = per_day
        print(f"    T={ts[0]:.3f}% / S={ts[1]:.3f}% done", flush=True)
    print(f"  Done in {time.time()-t2:.1f}s", flush=True)

    # Precompute IB-norm outcomes.
    t3 = time.time()
    print(f"\n  Precomputing IB-norm outcomes ({len(IB_NORM_GRID)} configs)...", flush=True)
    outcomes_by_ib: dict[tuple[float, float], dict[datetime.date, PctOutcomes]] = {}
    for norm in IB_NORM_GRID:
        per_day: dict[datetime.date, PctOutcomes] = {}
        for date in valid_days:
            dc = day_caches[date]
            ib_range = dc.ibh - dc.ibl
            if ib_range < 5:
                t_pts, s_pts = 4.0, 8.0
            else:
                t_pts = max(4.0, round(ib_range * norm[0] * 4) / 4)
                s_pts = max(8.0, round(ib_range * norm[1] * 4) / 4)
            # Convert to % for consistency (even though it varies by day).
            entry_price_approx = float(dc.post_ib_prices[0])
            t_pct = t_pts / entry_price_approx * 100 if entry_price_approx > 0 else 0.04
            s_pct = s_pts / entry_price_approx * 100 if entry_price_approx > 0 else 0.10
            per_day[date] = precompute_pct_outcomes(
                entries_by_date[date], dc, t_pct, s_pct
            )
        outcomes_by_ib[norm] = per_day
        print(f"    IB T={norm[0]}/S={norm[1]} done", flush=True)
    print(f"  Done in {time.time()-t3:.1f}s", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Factor analysis (full dataset, %-based T/S ≈ current 10/25)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  STAGE 1: Factor analysis (0.040%/0.100% T/S ≈ 10/25 at 25K)", flush=True)
    print("=" * 78, flush=True)

    ref_ts = (0.040, 0.100)
    ref_outcomes = outcomes_by_pct[ref_ts]
    all_eo: list[tuple[BotEntry, str]] = []
    for date in valid_days:
        enriched = enriched_by_date[date]
        do = ref_outcomes[date]
        for eb in enriched:
            all_eo.append((eb, do.outcome[eb.idx]))

    factor_analysis(all_eo)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward with bot-specific weights
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  STAGE 2: Walk-forward (bot-retrained weights, %-based T/S)", flush=True)
    print("=" * 78, flush=True)

    SCORE_GRID = [-1, 0, 1, 2, 3]
    MAX_E_GRID = [2, 3, 5]
    RISK_GRID = [(100.0, 3), (150.0, 3), (200.0, 3)]

    oos_by_config: dict[tuple, list[BotTrade]] = {}
    oos_days = 0
    k = INITIAL_TRAIN_DAYS
    windows = 0

    while k < N:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days:
            break
        windows += 1
        oos_days += len(test_days)

        # Train bot-specific weights on training data.
        train_eo: list[tuple[BotEntry, str]] = []
        for d in train_days:
            enriched = enriched_by_date[d]
            do = ref_outcomes[d]
            for eb in enriched:
                train_eo.append((eb, do.outcome[eb.idx]))
        w = fit_bot_weights(train_eo)

        # Sweep on test window.
        for ts in PCT_TS_GRID:
            for min_s in SCORE_GRID:
                for max_e in MAX_E_GRID:
                    for risk in RISK_GRID:
                        cfg = (ts, min_s, max_e, risk[0], risk[1])
                        trades = replay(
                            test_days,
                            entries_by_date,
                            enriched_by_date,
                            outcomes_by_pct[ts],
                            w,
                            min_score=min_s,
                            max_entries_per_level=max_e,
                            daily_loss_usd=risk[0],
                            max_consec_losses=risk[1],
                        )
                        oos_by_config.setdefault(cfg, []).extend(trades)

        # Also sweep IB-norm configs.
        for norm in IB_NORM_GRID:
            for min_s in SCORE_GRID:
                for max_e in MAX_E_GRID:
                    for risk in RISK_GRID:
                        cfg = (f"IB_{norm[0]}_{norm[1]}", min_s, max_e, risk[0], risk[1])
                        trades = replay(
                            test_days,
                            entries_by_date,
                            enriched_by_date,
                            outcomes_by_ib[norm],
                            w,
                            min_score=min_s,
                            max_entries_per_level=max_e,
                            daily_loss_usd=risk[0],
                            max_consec_losses=risk[1],
                        )
                        oos_by_config.setdefault(cfg, []).extend(trades)

        k += STEP_DAYS

    print(f"\n  {windows} windows, {oos_days} OOS days", flush=True)

    # Top 30 configs by P&L/day.
    results = []
    for cfg, trades in oos_by_config.items():
        if not trades:
            continue
        s = summarize(trades, oos_days, "")
        w_count = sum(1 for t in trades if t.outcome == "win")
        l_count = sum(1 for t in trades if t.outcome == "loss")
        d = w_count + l_count
        wr = w_count / d * 100 if d else 0
        pnl = sum(t.pnl_usd for t in trades)
        ppd = pnl / oos_days
        eq = STARTING_BALANCE
        peak = eq
        dd = 0.0
        for t in trades:
            eq += t.pnl_usd
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        results.append((cfg, len(trades), wr, pnl, ppd, dd))
    results.sort(key=lambda x: x[4], reverse=True)

    print(
        f"\n  {'T/S':>18s} {'Scr':>4} {'Max':>3} {'Risk':>7} "
        f"{'N':>5} {'WR%':>6} {'P&L':>9} {'$/d':>6} {'DD':>7}",
        flush=True,
    )
    print(f"  {'-'*18} {'-'*4} {'-'*3} {'-'*7} {'-'*5} {'-'*6} {'-'*9} {'-'*6} {'-'*7}", flush=True)
    for cfg, n, wr, pnl, ppd, dd in results[:30]:
        ts_str = str(cfg[0]) if isinstance(cfg[0], str) else f"{cfg[0][0]:.3f}/{cfg[0][1]:.3f}"
        risk_str = f"${int(cfg[3])}/{cfg[4]}"
        print(
            f"  {ts_str:>18s} {'>=' + str(cfg[1]):>4} {cfg[2]:>3} {risk_str:>7} "
            f"{n:>5} {wr:>5.1f}% ${pnl:>+8,.0f} ${ppd:>+5.1f} ${dd:>6,.0f}",
            flush=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Recent regime (last 60 days)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78, flush=True)
    print("  STAGE 3: Recent regime (last 60 days) — top configs", flush=True)
    print("=" * 78, flush=True)

    recent = valid_days[-60:]
    rn = len(recent)
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n", flush=True)

    # Train weights on everything before the recent period.
    pre_recent = [d for d in valid_days if d < recent[0]]
    pre_eo: list[tuple[BotEntry, str]] = []
    for d in pre_recent:
        enriched = enriched_by_date[d]
        do = ref_outcomes[d]
        for eb in enriched:
            pre_eo.append((eb, do.outcome[eb.idx]))
    w_recent = fit_bot_weights(pre_eo)

    # Test top configs on recent period.
    for ts in [(0.035, 0.090), (0.040, 0.100), (0.045, 0.100)]:
        print(f"  --- T/S = {ts[0]:.3f}% / {ts[1]:.3f}% ---")
        for min_s in [-1, 0, 1, 2, 3]:
            for max_e in [2, 3]:
                trades = replay(
                    recent,
                    entries_by_date,
                    enriched_by_date,
                    outcomes_by_pct[ts],
                    w_recent,
                    min_score=min_s,
                    max_entries_per_level=max_e,
                    daily_loss_usd=150.0,
                    max_consec_losses=3,
                )
                print(summarize(trades, rn, f"score>={min_s}, max={max_e}"))
        print()

    # IB-norm on recent.
    for norm in IB_NORM_GRID:
        print(f"  --- IB-norm T={norm[0]}/S={norm[1]} ---")
        for min_s in [-1, 0, 1, 2]:
            for max_e in [2, 3]:
                trades = replay(
                    recent,
                    entries_by_date,
                    enriched_by_date,
                    outcomes_by_ib[norm],
                    w_recent,
                    min_score=min_s,
                    max_entries_per_level=max_e,
                    daily_loss_usd=150.0,
                    max_consec_losses=3,
                )
                print(summarize(trades, rn, f"score>={min_s}, max={max_e}"))
        print()

    print(f"\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
