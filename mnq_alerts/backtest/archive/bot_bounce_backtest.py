"""
bot_bounce_backtest.py — Bounce-based bot backtest.

Analyzes actual bounce behavior at levels (how far, how fast) and tests
strategies designed around tight stops and realistic targets.

Stage 1: Bounce analysis — MFE (max favorable excursion) and MAE (max
         adverse excursion) per level/direction, to find where to set
         targets and stops based on what actually happens at the line.

Stage 2: Strategy sweep — tight stops, time-based exits, trailing stops.

Stage 3: Walk-forward validation of best strategies.

Usage:
    python -u bot_bounce_backtest.py
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
)
from bot_backtest import BOT_ENTRY_THRESHOLD, BOT_EXIT_THRESHOLD, FEE_PTS
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE
from config import BOT_EOD_FLATTEN_BUFFER_MIN
from walk_forward import (
    DayEntries,
    precompute_day_entries,
    _eod_cutoff_ns,
    INITIAL_TRAIN_DAYS,
    STEP_DAYS,
)
from bot_pct_backtest import (
    BotEntry,
    enrich_entries,
    fit_bot_weights,
    score_bot_entry,
    PctOutcomes,
    precompute_pct_outcomes,
)

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Bounce analysis — MFE/MAE/time-to-move
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BounceStats:
    """Stats for a single entry's price action after zone entry."""

    level: str
    direction: str
    entry_price: float
    line_price: float
    mfe_pts: float  # max favorable excursion (from line price)
    mae_pts: float  # max adverse excursion (from line price)
    mfe_secs: float  # seconds until MFE was reached
    pnl_at_60s: float  # pts favorable at 60 seconds (from line)
    pnl_at_120s: float  # pts favorable at 120 seconds
    pnl_at_300s: float  # pts favorable at 5 minutes


def analyze_bounces(
    de: DayEntries,
    dc: DayCache,
    window_secs: int = 15 * 60,
) -> list[BounceStats]:
    """Compute MFE/MAE for each entry in a day."""
    results = []
    eod_ns = _eod_cutoff_ns(dc.date)

    for i in range(len(de.global_idx)):
        gidx = int(de.global_idx[i])
        entry_ns = int(de.entry_ns[i])
        if entry_ns >= eod_ns:
            continue
        line_price = float(de.ref_price[i])
        direction = de.direction[i]
        entry_price = float(de.entry_price[i])

        window_ns = np.int64(window_secs * 1_000_000_000)
        end_ns = min(entry_ns + window_ns, np.int64(eod_ns))

        mfe = 0.0
        mae = 0.0
        mfe_secs = 0.0
        pnl_60 = 0.0
        pnl_120 = 0.0
        pnl_300 = 0.0

        for j in range(gidx + 1, len(dc.full_prices)):
            if dc.full_ts_ns[j] > end_ns:
                break
            p = float(dc.full_prices[j])
            elapsed_secs = (dc.full_ts_ns[j] - entry_ns) / 1e9

            if direction == "up":
                fav = p - line_price
                adv = line_price - p
            else:
                fav = line_price - p
                adv = p - line_price

            if fav > mfe:
                mfe = fav
                mfe_secs = elapsed_secs
            if adv > mae:
                mae = adv

            if elapsed_secs <= 60 and fav > pnl_60:
                pnl_60 = fav
            elif elapsed_secs <= 60 and direction == "up":
                pnl_60 = min(pnl_60, fav) if pnl_60 != 0 else fav
            if 59 < elapsed_secs <= 61:
                pnl_60 = fav
            if 119 < elapsed_secs <= 121:
                pnl_120 = fav
            if 299 < elapsed_secs <= 301:
                pnl_300 = fav

        results.append(
            BounceStats(
                level=de.level[i],
                direction=direction,
                entry_price=entry_price,
                line_price=line_price,
                mfe_pts=mfe,
                mae_pts=mae,
                mfe_secs=mfe_secs,
                pnl_at_60s=pnl_60,
                pnl_at_120s=pnl_120,
                pnl_at_300s=pnl_300,
            )
        )

    return results


def print_bounce_analysis(all_bounces: list[BounceStats]) -> None:
    """Print MFE/MAE statistics by level×direction."""
    print(f"\n  Total entries: {len(all_bounces)}\n")

    combos: dict[str, list[BounceStats]] = {}
    for b in all_bounces:
        key = f"{b.level} × {b.direction}"
        combos.setdefault(key, []).append(b)
    combos["ALL"] = all_bounces

    print(
        f"  {'Level × Dir':>30s}    N  "
        f"{'MFE p25':>7} {'MFE p50':>7} {'MFE p75':>7} "
        f"{'MAE p25':>7} {'MAE p50':>7} {'MAE p75':>7} "
        f"{'MFE sec':>7}",
    )
    print(f"  {'-'*30}  ---  {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for key in sorted(combos.keys()):
        bs = combos[key]
        if len(bs) < 20:
            continue
        mfes = sorted(b.mfe_pts for b in bs)
        maes = sorted(b.mae_pts for b in bs)
        mfe_secs = sorted(b.mfe_secs for b in bs)
        n = len(bs)

        print(
            f"  {key:>30s}  {n:>3}  "
            f"{mfes[n//4]:>7.1f} {mfes[n//2]:>7.1f} {mfes[3*n//4]:>7.1f} "
            f"{maes[n//4]:>7.1f} {maes[n//2]:>7.1f} {maes[3*n//4]:>7.1f} "
            f"{mfe_secs[n//2]:>7.0f}s",
        )


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM TRADE EVALUATION: tight stop + time exit + trailing stop
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_bounce_trade(
    entry_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    target_pts: float,
    stop_pts: float,
    max_hold_secs: int = 900,
    time_exit_secs: int | None = None,
    trail_activate_pts: float | None = None,
    trail_offset_pts: float | None = None,
    eod_cutoff_ns: int | None = None,
) -> tuple[str, int, float]:
    """Evaluate a bounce trade with optional time exit and trailing stop.

    Args:
        target_pts: limit target from line price
        stop_pts: initial stop from line price
        max_hold_secs: max hold time (hard timeout)
        time_exit_secs: if set, exit at market after this many seconds
                        if NOT profitable (favorable excursion < 1 pt)
        trail_activate_pts: once favorable by this much, activate trailing stop
        trail_offset_pts: trailing stop distance from peak favorable price

    Returns: (outcome, exit_idx, pnl_pts)
    """
    entry_ns = ts_ns[entry_idx]
    max_ns = entry_ns + np.int64(max_hold_secs * 1_000_000_000)
    if eod_cutoff_ns is not None and eod_cutoff_ns < max_ns:
        max_ns = np.int64(eod_cutoff_ns)

    time_exit_ns = (
        entry_ns + np.int64(time_exit_secs * 1_000_000_000)
        if time_exit_secs
        else None
    )

    if direction == "up":
        target_price = line_price + target_pts
        stop_price = line_price - stop_pts
    else:
        target_price = line_price - target_pts
        stop_price = line_price + stop_pts

    best_favorable = 0.0
    trailing_stop_active = False
    last_idx = entry_idx

    for j in range(entry_idx + 1, len(prices)):
        if ts_ns[j] > max_ns:
            break
        last_idx = j
        p = float(prices[j])

        # Check target (limit order — fills at target price).
        if direction == "up" and p >= target_price:
            return "win", j, target_pts - FEE_PTS
        if direction == "down" and p <= target_price:
            return "win", j, target_pts - FEE_PTS

        # Check stop.
        if direction == "up" and p <= stop_price:
            return "loss", j, -(stop_pts + FEE_PTS)
        if direction == "down" and p >= stop_price:
            return "loss", j, -(stop_pts + FEE_PTS)

        # Track favorable excursion.
        if direction == "up":
            fav = p - line_price
        else:
            fav = line_price - p
        best_favorable = max(best_favorable, fav)

        # Trailing stop: once price moves trail_activate_pts in our favor,
        # set a trailing stop at trail_offset_pts behind peak.
        if (
            trail_activate_pts is not None
            and trail_offset_pts is not None
            and best_favorable >= trail_activate_pts
        ):
            trailing_stop_active = True
            if direction == "up":
                trail_stop = (line_price + best_favorable) - trail_offset_pts
                if p <= trail_stop:
                    pnl = p - line_price - FEE_PTS
                    return "trail", j, pnl
            else:
                trail_stop = (line_price - best_favorable) + trail_offset_pts
                if p >= trail_stop:
                    pnl = line_price - p - FEE_PTS
                    return "trail", j, pnl

        # Time-based exit: if not profitable after N seconds, cut.
        if time_exit_ns is not None and ts_ns[j] >= time_exit_ns:
            if best_favorable < 1.0:
                # Exit at current price.
                if direction == "up":
                    pnl = p - line_price - FEE_PTS
                else:
                    pnl = line_price - p - FEE_PTS
                return "time_exit", j, pnl

    # Timeout — close at last price.
    exit_price = float(prices[last_idx])
    if direction == "up":
        pnl = exit_price - line_price - FEE_PTS
    else:
        pnl = line_price - exit_price - FEE_PTS
    return "timeout", last_idx, pnl


# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE BOUNCE OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BounceOutcomes:
    date: datetime.date
    outcome: list[str]
    exit_ns: np.ndarray
    pnl_pts: np.ndarray
    pnl_usd: np.ndarray


def precompute_bounce_outcomes(
    de: DayEntries,
    dc: DayCache,
    target_pts: float,
    stop_pts: float,
    max_hold_secs: int = 900,
    time_exit_secs: int | None = None,
    trail_activate_pts: float | None = None,
    trail_offset_pts: float | None = None,
) -> BounceOutcomes:
    n = len(de.global_idx)
    outcomes: list[str] = []
    exit_ns = np.zeros(n, dtype=np.int64)
    pnl_pts = np.zeros(n, dtype=np.float64)
    eod_ns = _eod_cutoff_ns(dc.date)

    for i in range(n):
        out, exit_idx, pnl = evaluate_bounce_trade(
            int(de.global_idx[i]),
            float(de.ref_price[i]),
            de.direction[i],
            dc.full_ts_ns,
            dc.full_prices,
            target_pts,
            stop_pts,
            max_hold_secs,
            time_exit_secs,
            trail_activate_pts,
            trail_offset_pts,
            eod_ns,
        )
        outcomes.append(out)
        exit_ns[i] = int(dc.full_ts_ns[exit_idx])
        pnl_pts[i] = pnl

    return BounceOutcomes(
        date=de.date,
        outcome=outcomes,
        exit_ns=exit_ns,
        pnl_pts=pnl_pts,
        pnl_usd=pnl_pts * MULTIPLIER,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    direction: str
    outcome: str
    pnl_usd: float


def replay(
    days, entries_by_date, enriched_by_date, outcomes_by_date,
    weights, min_score, max_entries_per_level,
    daily_loss_usd, max_consec_losses,
):
    trades = []
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
        lc = {}
        for eb in enriched:
            if stopped:
                break
            if eb.entry_ns >= eod_ns:
                break
            if eb.entry_ns < pos_exit_ns:
                continue
            lv_count = lc.get(eb.level, 0)
            if lv_count >= max_entries_per_level:
                continue
            # Vol filter
            range_30m_pct = (
                eb.range_30m / eb.entry_price * 100
                if eb.range_30m is not None and eb.entry_price > 0
                else None
            )
            if range_30m_pct is not None and range_30m_pct < 0.15:
                continue
            score = score_bot_entry(eb, weights, cw, cl)
            if score < min_score:
                continue
            i = eb.idx
            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            pos_exit_ns = int(do.exit_ns[i])
            lc[eb.level] = lv_count + 1
            trades.append(Trade(date, eb.level, eb.direction, outcome, pnl))
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


def summarize(trades, n_days, label=""):
    if not trades:
        return f"  {label:>50s}  no trades"
    w = sum(1 for t in trades if t.outcome == "win")
    l = sum(1 for t in trades if t.outcome in ("loss",))
    other = len(trades) - w - l
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
    avg_w = sum(t.pnl_usd for t in trades if t.outcome == "win") / w if w else 0
    avg_l = sum(t.pnl_usd for t in trades if t.outcome in ("loss",)) / l if l else 0
    return (
        f"  {label:>50s}  {len(trades):>4} ({len(trades)/n_days:.1f}/d) "
        f"{w}W/{l}L/{other}O {wr:>5.1f}%  "
        f"W${avg_w:>+5.1f} L${avg_l:>+5.1f}  "
        f"${pnl:>+7,.0f} (${ppd:>+5.1f}/d)  DD${dd:>5,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 90)
    print("  BOUNCE BACKTEST — Tight stops, time exits, trailing stops")
    print("=" * 90)

    days = load_cached_days()
    day_caches = {}
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
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s ({valid_days[0]} → {valid_days[-1]})")

    # Precompute entries + enrichment.
    entries_by_date = {}
    enriched_by_date = {}
    for date in valid_days:
        de = precompute_day_entries(day_caches[date])
        entries_by_date[date] = de
        enriched_by_date[date] = enrich_entries(de, day_caches[date])
    print(f"  Entries enriched in {time.time()-t0:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: Bounce analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  STAGE 1: Bounce analysis — MFE/MAE per level×direction")
    print("=" * 90)

    all_bounces = []
    for date in valid_days:
        all_bounces.extend(analyze_bounces(entries_by_date[date], day_caches[date]))
    print_bounce_analysis(all_bounces)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Strategy sweep (full dataset, fixed weights)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  STAGE 2: Strategy sweep (full dataset)")
    print("=" * 90)

    # Train weights on full dataset for initial sweep.
    ref_outcomes = {}
    for date in valid_days:
        ref_outcomes[date] = precompute_pct_outcomes(
            entries_by_date[date], day_caches[date], 0.040, 0.100
        )
    all_eo = [
        (eb, ref_outcomes[date].outcome[eb.idx])
        for date in valid_days
        for eb in enriched_by_date[date]
    ]
    w = fit_bot_weights(all_eo)
    print(f"  Weights trained in {time.time()-t0:.0f}s")

    # A) Tight stop sweep with various targets.
    print(f"\n  --- A) Target/Stop sweep (score>=-1, max 3/level, $150/3) ---")
    strategies = [
        # (target, stop, time_exit, trail_act, trail_off, label)
        (8, 5, None, None, None, "T8/S5"),
        (10, 5, None, None, None, "T10/S5"),
        (12, 5, None, None, None, "T12/S5"),
        (15, 5, None, None, None, "T15/S5"),
        (8, 8, None, None, None, "T8/S8"),
        (10, 8, None, None, None, "T10/S8"),
        (12, 8, None, None, None, "T12/S8"),
        (15, 8, None, None, None, "T15/S8"),
        (10, 10, None, None, None, "T10/S10"),
        (12, 10, None, None, None, "T12/S10"),
        (15, 10, None, None, None, "T15/S10"),
        (20, 10, None, None, None, "T20/S10"),
        (12, 12, None, None, None, "T12/S12"),
        (15, 12, None, None, None, "T15/S12"),
        (10, 25, None, None, None, "T10/S25 (old)"),
        (12, 25, None, None, None, "T12/S25 (original)"),
    ]

    for tgt, stp, te, ta, to_, label in strategies:
        outcomes = {}
        for date in valid_days:
            outcomes[date] = precompute_bounce_outcomes(
                entries_by_date[date], day_caches[date],
                float(tgt), float(stp), time_exit_secs=te,
                trail_activate_pts=ta, trail_offset_pts=to_,
            )
        trades = replay(
            valid_days, entries_by_date, enriched_by_date, outcomes,
            w, -1, 3, 150.0, 3,
        )
        print(summarize(trades, N, label))

    # B) Time-based exit: cut losers early.
    print(f"\n  --- B) Time exit sweep (T12/S8, score>=-1, max 3, $150/3) ---")
    for time_exit in [30, 60, 90, 120, 180, 300, None]:
        label = f"T12/S8 + exit@{time_exit}s" if time_exit else "T12/S8 (no time exit)"
        outcomes = {}
        for date in valid_days:
            outcomes[date] = precompute_bounce_outcomes(
                entries_by_date[date], day_caches[date],
                12.0, 8.0, time_exit_secs=time_exit,
            )
        trades = replay(
            valid_days, entries_by_date, enriched_by_date, outcomes,
            w, -1, 3, 150.0, 3,
        )
        print(summarize(trades, N, label))

    # C) Trailing stop: lock in profits.
    print(f"\n  --- C) Trailing stop sweep (T15/S8, score>=-1, max 3, $150/3) ---")
    for trail_act, trail_off in [(4, 3), (5, 3), (5, 4), (6, 4), (8, 5), (8, 6)]:
        label = f"T15/S8 trail@{trail_act}/{trail_off}"
        outcomes = {}
        for date in valid_days:
            outcomes[date] = precompute_bounce_outcomes(
                entries_by_date[date], day_caches[date],
                15.0, 8.0, trail_activate_pts=float(trail_act),
                trail_offset_pts=float(trail_off),
            )
        trades = replay(
            valid_days, entries_by_date, enriched_by_date, outcomes,
            w, -1, 3, 150.0, 3,
        )
        print(summarize(trades, N, label))

    # D) Combined: time exit + trailing stop on best configs.
    print(f"\n  --- D) Combined: time exit + trailing stop ---")
    combos = [
        (12, 8, 120, 5, 3, "T12/S8 exit@120s trail@5/3"),
        (12, 8, 120, 6, 4, "T12/S8 exit@120s trail@6/4"),
        (15, 8, 120, 5, 3, "T15/S8 exit@120s trail@5/3"),
        (15, 8, 120, 6, 4, "T15/S8 exit@120s trail@6/4"),
        (12, 8, 90, 5, 3, "T12/S8 exit@90s trail@5/3"),
        (15, 10, 120, 6, 4, "T15/S10 exit@120s trail@6/4"),
        (20, 10, 120, 8, 5, "T20/S10 exit@120s trail@8/5"),
    ]
    for tgt, stp, te, ta, to_, label in combos:
        outcomes = {}
        for date in valid_days:
            outcomes[date] = precompute_bounce_outcomes(
                entries_by_date[date], day_caches[date],
                float(tgt), float(stp), time_exit_secs=te,
                trail_activate_pts=float(ta), trail_offset_pts=float(to_),
            )
        trades = replay(
            valid_days, entries_by_date, enriched_by_date, outcomes,
            w, -1, 3, 150.0, 3,
        )
        print(summarize(trades, N, label))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Walk-forward on top configs
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  STAGE 3: Walk-forward validation (top configs)")
    print("=" * 90)

    # Pick the best configs from Stage 2 to validate OOS.
    wf_configs = [
        (10, 5, None, None, None, "T10/S5"),
        (12, 5, None, None, None, "T12/S5"),
        (10, 8, None, None, None, "T10/S8"),
        (12, 8, None, None, None, "T12/S8"),
        (15, 8, None, None, None, "T15/S8"),
        (12, 8, 120, 5, 3, "T12/S8 exit@120s trail@5/3"),
        (15, 8, 120, 6, 4, "T15/S8 exit@120s trail@6/4"),
        (15, 10, None, None, None, "T15/S10"),
        (20, 10, None, None, None, "T20/S10"),
        (10, 25, None, None, None, "T10/S25 (old)"),
    ]

    # Precompute outcomes for all WF configs.
    print("  Precomputing outcomes for WF configs...", flush=True)
    wf_outcomes = {}
    for tgt, stp, te, ta, to_, label in wf_configs:
        key = label
        wf_outcomes[key] = {}
        for date in valid_days:
            wf_outcomes[key][date] = precompute_bounce_outcomes(
                entries_by_date[date], day_caches[date],
                float(tgt), float(stp), time_exit_secs=te,
                trail_activate_pts=float(ta) if ta else None,
                trail_offset_pts=float(to_) if to_ else None,
            )
    print(f"  Done in {time.time()-t0:.0f}s")

    # Walk-forward.
    oos_by_config: dict[str, list[Trade]] = {}
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

        # Train weights on training data.
        train_eo = [
            (eb, ref_outcomes[d].outcome[eb.idx])
            for d in train_days
            for eb in enriched_by_date.get(d, [])
        ]
        wt = fit_bot_weights(train_eo)

        # Test each config OOS.
        for tgt, stp, te, ta, to_, label in wf_configs:
            for min_s in [-1, 0, 1]:
                cfg_key = f"{label} score>={min_s}"
                trades = replay(
                    test_days, entries_by_date, enriched_by_date,
                    wf_outcomes[label], wt, min_s, 3, 150.0, 3,
                )
                oos_by_config.setdefault(cfg_key, []).extend(trades)

        k += STEP_DAYS

    print(f"\n  {windows} windows, {oos_days} OOS days\n")

    # Sort by P&L/day.
    results = []
    for cfg, trades in oos_by_config.items():
        if not trades:
            continue
        pnl = sum(t.pnl_usd for t in trades)
        ppd = pnl / oos_days
        results.append((cfg, trades, ppd))
    results.sort(key=lambda x: x[2], reverse=True)

    for cfg, trades, ppd in results[:20]:
        print(summarize(trades, oos_days, cfg))

    # Recent 60 days for best config.
    print(f"\n  --- Recent 60 days ---")
    recent = valid_days[-60:]
    rn = len(recent)
    pre = [d for d in valid_days if d < recent[0]]
    pre_eo = [
        (eb, ref_outcomes[d].outcome[eb.idx])
        for d in pre
        for eb in enriched_by_date.get(d, [])
    ]
    w_recent = fit_bot_weights(pre_eo)

    for cfg, _, _ in results[:10]:
        # Parse config to find the right outcomes and score.
        parts = cfg.rsplit(" score>=", 1)
        label = parts[0]
        min_s = int(parts[1])
        if label in wf_outcomes:
            trades = replay(
                recent, entries_by_date, enriched_by_date,
                wf_outcomes[label], w_recent, min_s, 3, 150.0, 3,
            )
            print(summarize(trades, rn, f"RECENT: {cfg}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
