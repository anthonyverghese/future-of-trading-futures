"""
bot_fast_backtest.py — Fast-winner / quick-cut bot backtest.

Based on MAE analysis showing winners resolve in median 27s while losers
hold for 78s. Tests time-based exits to cut losers early and free the
position slot for more trades.

Walk-forward validated with 1-position-at-a-time constraint.
Optimizes for $/day.

Usage:
    python -u bot_fast_backtest.py
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
    precompute_pct_outcomes,
)

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# TRADE EVALUATION WITH TIME CUT
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_fast_trade(
    entry_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    target_pts: float,
    stop_pts: float,
    time_cut_secs: int | None,
    max_hold_secs: int = 900,
    eod_cutoff_ns: int | None = None,
) -> tuple[str, int, float]:
    """Evaluate trade with optional time-based cut for losers.

    Time cut: after time_cut_secs, if trade is losing (price on wrong
    side of line), exit at current price. Winners and trades that are
    still near breakeven are left to run.

    Returns: (outcome, exit_idx, pnl_pts)
    """
    entry_ns = ts_ns[entry_idx]
    max_ns = entry_ns + np.int64(max_hold_secs * 1_000_000_000)
    if eod_cutoff_ns is not None and eod_cutoff_ns < max_ns:
        max_ns = np.int64(eod_cutoff_ns)

    cut_ns = (
        entry_ns + np.int64(time_cut_secs * 1_000_000_000)
        if time_cut_secs is not None
        else None
    )

    if direction == "up":
        target_price = line_price + target_pts
        stop_price = line_price - stop_pts
    else:
        target_price = line_price - target_pts
        stop_price = line_price + stop_pts

    last_idx = entry_idx

    for j in range(entry_idx + 1, len(prices)):
        if ts_ns[j] > max_ns:
            break
        last_idx = j
        p = float(prices[j])

        # Target hit (limit order).
        if direction == "up" and p >= target_price:
            return "win", j, target_pts - FEE_PTS
        if direction == "down" and p <= target_price:
            return "win", j, target_pts - FEE_PTS

        # Stop hit.
        if direction == "up" and p <= stop_price:
            return "stop", j, -(stop_pts + FEE_PTS)
        if direction == "down" and p >= stop_price:
            return "stop", j, -(stop_pts + FEE_PTS)

        # Time cut: if past the cut time and trade is losing, exit.
        if cut_ns is not None and ts_ns[j] >= cut_ns:
            if direction == "up":
                fav = p - line_price
            else:
                fav = line_price - p
            if fav < 0:
                # Trade is losing — cut it.
                pnl = fav - FEE_PTS
                return "time_cut", j, pnl

    # Timeout.
    exit_price = float(prices[last_idx])
    if direction == "up":
        pnl = exit_price - line_price - FEE_PTS
    else:
        pnl = line_price - exit_price - FEE_PTS
    return "timeout", last_idx, pnl


# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FastOutcomes:
    date: datetime.date
    outcome: list[str]
    exit_ns: np.ndarray
    pnl_pts: np.ndarray
    pnl_usd: np.ndarray


def precompute_fast_outcomes(
    de: DayEntries,
    dc: DayCache,
    target_pts: float,
    stop_pts: float,
    time_cut_secs: int | None,
    max_hold_secs: int = 900,
) -> FastOutcomes:
    n = len(de.global_idx)
    outcomes: list[str] = []
    exit_ns = np.zeros(n, dtype=np.int64)
    pnl_pts = np.zeros(n, dtype=np.float64)
    eod_ns = _eod_cutoff_ns(dc.date)

    for i in range(n):
        out, eidx, pnl = evaluate_fast_trade(
            int(de.global_idx[i]),
            float(de.ref_price[i]),
            de.direction[i],
            dc.full_ts_ns,
            dc.full_prices,
            target_pts,
            stop_pts,
            time_cut_secs,
            max_hold_secs,
            eod_ns,
        )
        outcomes.append(out)
        exit_ns[i] = int(dc.full_ts_ns[eidx])
        pnl_pts[i] = pnl

    return FastOutcomes(
        date=de.date,
        outcome=outcomes,
        exit_ns=exit_ns,
        pnl_pts=pnl_pts,
        pnl_usd=pnl_pts * MULTIPLIER,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY WITH 1-POSITION CONSTRAINT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    direction: str
    outcome: str
    pnl_usd: float
    hold_secs: float


def replay_1pos(
    days,
    entries_by_date,
    enriched_by_date,
    outcomes_by_date,
    weights,
    min_score,
    max_entries_per_level,
    daily_loss_usd,
    max_consec_losses,
    include_vwap=False,
):
    """Replay with 1-position-at-a-time constraint."""
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
            # 1-position-at-a-time: skip if slot is occupied.
            if eb.entry_ns < pos_exit_ns:
                continue
            # Skip VWAP unless included.
            if not include_vwap and eb.level == "VWAP":
                continue
            # Per-level cap.
            lv_count = lc.get(eb.level, 0)
            if lv_count >= max_entries_per_level:
                continue
            # Volatility filter.
            range_30m_pct = (
                eb.range_30m / eb.entry_price * 100
                if eb.range_30m is not None and eb.entry_price > 0
                else None
            )
            if range_30m_pct is not None and range_30m_pct < 0.15:
                continue
            # Score filter.
            score = score_bot_entry(eb, weights, cw, cl)
            if score < min_score:
                continue

            i = eb.idx
            outcome = do.outcome[i]
            pnl = float(do.pnl_usd[i])
            entry_ns = int(eb.entry_ns)
            exit_ns = int(do.exit_ns[i])
            hold_secs = (exit_ns - entry_ns) / 1e9

            # Block slot until this trade exits.
            pos_exit_ns = exit_ns
            lc[eb.level] = lv_count + 1

            trades.append(
                Trade(date, eb.level, eb.direction, outcome, pnl, hold_secs)
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


def fmt(trades, n_days, label=""):
    """One-line summary."""
    if not trades:
        return f"  {label:>55s}  no trades"
    w = sum(1 for t in trades if t.outcome == "win")
    stops = sum(1 for t in trades if t.outcome == "stop")
    cuts = sum(1 for t in trades if t.outcome == "time_cut")
    tos = sum(1 for t in trades if t.outcome == "timeout")
    total_pnl = sum(t.pnl_usd for t in trades)
    ppd = total_pnl / n_days if n_days else 0
    decided = w + stops
    wr = w / decided * 100 if decided else 0
    avg_hold = sum(t.hold_secs for t in trades) / len(trades)

    eq = STARTING_BALANCE
    peak = eq
    dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)

    return (
        f"  {label:>55s}  "
        f"{len(trades):>4} ({len(trades)/n_days:.1f}/d) "
        f"{w}W/{stops}S/{cuts}C/{tos}T  {wr:>5.1f}%  "
        f"avg {avg_hold:>4.0f}s  "
        f"${total_pnl:>+8,.0f} (${ppd:>+5.1f}/d)  DD${dd:>5,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 100)
    print("  FAST-WINNER / QUICK-CUT BACKTEST")
    print("  Optimizing for $/day with 1-position-at-a-time constraint")
    print("=" * 100)

    # Load data.
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
    print(f"\n  {N} days loaded ({valid_days[0]} → {valid_days[-1]})")

    # Precompute entries + enrichment (includes VWAP for optional inclusion).
    entries_by_date = {}
    enriched_by_date = {}
    for date in valid_days:
        de = precompute_day_entries(day_caches[date])
        entries_by_date[date] = de
        enriched_by_date[date] = enrich_entries(de, day_caches[date])
    print(f"  Entries enriched in {time.time()-t0:.0f}s")

    # Train weights on full data for Stage 1 sweep.
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
    w_full = fit_bot_weights(all_eo)
    print(f"  Weights trained in {time.time()-t0:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: Full sweep (in-sample, to find promising configs)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 1: Config sweep (in-sample, 1-position constraint)")
    print("=" * 100)

    TARGET_GRID = [6, 8, 10, 12]
    STOP_GRID = [10, 15, 20, 25]
    TIME_CUT_GRID = [None, 45, 60, 90, 120]

    # Precompute all outcome combinations.
    print(f"\n  Precomputing outcomes ({len(TARGET_GRID)}×{len(STOP_GRID)}×{len(TIME_CUT_GRID)} = {len(TARGET_GRID)*len(STOP_GRID)*len(TIME_CUT_GRID)} configs)...", flush=True)
    t1 = time.time()
    all_outcomes = {}
    for tgt in TARGET_GRID:
        for stp in STOP_GRID:
            if stp < tgt:
                continue  # skip nonsensical combos where stop < target
            for tc in TIME_CUT_GRID:
                key = (tgt, stp, tc)
                per_day = {}
                for date in valid_days:
                    per_day[date] = precompute_fast_outcomes(
                        entries_by_date[date],
                        day_caches[date],
                        float(tgt),
                        float(stp),
                        tc,
                    )
                all_outcomes[key] = per_day
    print(f"  {len(all_outcomes)} configs computed in {time.time()-t1:.0f}s")

    # Sweep all combos.
    results = []
    for key, outcomes in all_outcomes.items():
        tgt, stp, tc = key
        for min_score in [-1, 0, 1]:
            for include_vwap in [False, True]:
                trades = replay_1pos(
                    valid_days,
                    entries_by_date,
                    enriched_by_date,
                    outcomes,
                    w_full,
                    min_score,
                    3,  # max entries per level
                    150.0,
                    3,  # max consec losses
                    include_vwap=include_vwap,
                )
                pnl = sum(t.pnl_usd for t in trades)
                ppd = pnl / N
                vwap_str = "+VWAP" if include_vwap else ""
                tc_str = str(tc) if tc is not None else "none"
                label = f"T{tgt}/S{stp} cut@{tc_str:>4s} score>={min_score} {vwap_str}"
                results.append((label, trades, ppd))

    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top 30 configs by $/day (in-sample):\n")
    for label, trades, ppd in results[:30]:
        print(fmt(trades, N, label))

    print(f"\n  Bottom 5 (worst):\n")
    for label, trades, ppd in results[-5:]:
        print(fmt(trades, N, label))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward on top configs
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 2: Walk-forward validation (top configs, OOS)")
    print("=" * 100)

    # Select top distinct (tgt, stp, tc) combos to validate OOS.
    seen_keys: set[tuple] = set()
    wf_keys: list[tuple[int, int, int | None]] = []
    for label, trades, ppd in results:
        # Find the outcome key that produced this label.
        for key in all_outcomes:
            tgt, stp, tc = key
            tc_str = str(tc) if tc is not None else "none"
            if f"T{tgt}/S{stp}" in label and f"cut@{tc_str}" in label:
                if key not in seen_keys:
                    seen_keys.add(key)
                    wf_keys.append(key)
                break
        if len(wf_keys) >= 15:
            break

    # Always include no-cut baselines.
    for tgt in [8, 10]:
        for stp in [20, 25]:
            key = (tgt, stp, None)
            if key not in seen_keys and key in all_outcomes:
                seen_keys.add(key)
                wf_keys.append(key)

    print(f"\n  Validating {len(wf_keys)} T/S/TC configs OOS...")

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

        # Train weights.
        train_eo = [
            (eb, ref_outcomes[d].outcome[eb.idx])
            for d in train_days
            for eb in enriched_by_date.get(d, [])
        ]
        wt = fit_bot_weights(train_eo)

        # Test each config.
        for tgt, stp, tc in wf_keys:
            key = (tgt, stp, tc)

            for min_score in [-1, 0, 1]:
                for include_vwap in [False, True]:
                    vwap_str = "+VWAP" if include_vwap else ""
                    tc_str = str(tc) if tc is not None else "none"
                    cfg_label = f"T{tgt}/S{stp} cut@{tc_str:>4s} score>={min_score} {vwap_str}"
                    trades = replay_1pos(
                        test_days,
                        entries_by_date,
                        enriched_by_date,
                        all_outcomes[key],
                        wt,
                        min_score,
                        3,
                        150.0,
                        3,
                        include_vwap=include_vwap,
                    )
                    oos_by_config.setdefault(cfg_label, []).extend(trades)

        k += STEP_DAYS

    print(f"  {windows} windows, {oos_days} OOS days\n")

    # Sort by $/day OOS.
    oos_results = []
    for cfg, trades in oos_by_config.items():
        pnl = sum(t.pnl_usd for t in trades)
        ppd = pnl / oos_days if oos_days else 0
        oos_results.append((cfg, trades, ppd))
    oos_results.sort(key=lambda x: x[2], reverse=True)

    print(f"  Top 20 configs by $/day (OOS walk-forward):\n")
    for cfg, trades, ppd in oos_results[:20]:
        print(fmt(trades, oos_days, cfg))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days for top 10
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 3: Recent 60 days (top configs)")
    print("=" * 100)

    recent = valid_days[-60:]
    rn = len(recent)
    pre = [d for d in valid_days if d < recent[0]]
    pre_eo = [
        (eb, ref_outcomes[d].outcome[eb.idx])
        for d in pre
        for eb in enriched_by_date.get(d, [])
    ]
    w_recent = fit_bot_weights(pre_eo)

    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")

    for cfg, _, _ in oos_results[:10]:
        # Parse config from label string.
        parts = cfg.strip().split()
        ts_parts = parts[0].replace("T", "").split("/S")
        tgt = int(ts_parts[0])
        stp = int(ts_parts[1])
        tc_raw = parts[1].split("@")[1].strip()
        tc = None if tc_raw == "none" else int(tc_raw)
        score_part = [p for p in parts if p.startswith("score>=")][0]
        min_score = int(score_part.split(">=")[1])
        include_vwap = "+VWAP" in cfg
        key = (tgt, stp, tc)

        if key not in all_outcomes:
            continue

        trades = replay_1pos(
            recent,
            entries_by_date,
            enriched_by_date,
            all_outcomes[key],
            w_recent,
            min_score,
            3,
            150.0,
            3,
            include_vwap=include_vwap,
        )
        print(fmt(trades, rn, f"RECENT: {cfg}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
