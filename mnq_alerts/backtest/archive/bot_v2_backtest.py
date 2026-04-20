"""
bot_v2_backtest.py — Fixed bot backtest with correct human-to-bot entry matching.

Previous backtests had a bug: parallel zone state machines caused human
scores to not match bot entries correctly. This version uses the simple
correct approach: run the human zone, find where price hits the line,
evaluate from there.

Usage:
    python -u bot_v2_backtest.py
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
    DayCache, load_cached_days, load_day, preprocess_day,
    simulate_day, _run_zone_numpy,
    ALERT_THRESHOLD, EXIT_THRESHOLD, HIT_THRESHOLD,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, FEE_PTS, evaluate_bot_trade
from score_optimizer import Weights, compute_tick_rate, score_alert, EnrichedAlert
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY: human alert → find line-touch tick → that's the bot entry
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BotEntry:
    level: str
    direction: str
    line_price: float
    hit_idx: int  # global index where price first touches the line
    hit_ns: int
    human_score: int
    entry_count: int


def get_bot_entries(
    dc: DayCache, human_w: Weights, cw: int, cl: int,
) -> tuple[list[BotEntry], int, int]:
    """For each decided human alert, find where price hits the line.

    Returns (entries, updated_cw, updated_cl).
    """
    alerts = simulate_day(dc)
    alerts.sort(key=lambda a: a.alert_time)
    first_price = float(dc.post_ib_prices[0])

    entries = []
    for a in alerts:
        if a.outcome not in ("correct", "incorrect"):
            continue

        # Score the human alert.
        if hasattr(a.alert_time, "astimezone") and a.alert_time.tzinfo:
            now_et = a.alert_time.astimezone(
                datetime.timezone(datetime.timedelta(hours=-4))
            ).time()
        else:
            now_et = None
        tr = compute_tick_rate(dc.full_df, pd.Timestamp(a.alert_time))
        sm = a.entry_price - first_price
        ea = EnrichedAlert(
            date=dc.date, level=a.level, direction=a.direction,
            entry_count=a.level_test_count, outcome=a.outcome,
            entry_price=a.entry_price, line_price=a.line_price,
            alert_time=a.alert_time, now_et=now_et, tick_rate=tr,
            session_move_pts=sm, consecutive_wins=cw, consecutive_losses=cl,
        )
        sc = score_alert(ea, human_w)
        if a.outcome == "correct":
            cw += 1; cl = 0
        else:
            cl += 1; cw = 0

        # Find the tick where price first touches the line (within 1 pt).
        # The human evaluation already confirmed this happens (outcome is decided).
        alert_ts = pd.Timestamp(a.alert_time)
        alert_ns = int(alert_ts.value)
        hit_idx = -1
        for j in range(len(dc.full_ts_ns)):
            if dc.full_ts_ns[j] < alert_ns:
                continue
            if abs(float(dc.full_prices[j]) - a.line_price) <= HIT_THRESHOLD:
                hit_idx = j
                break

        if hit_idx < 0:
            continue  # shouldn't happen for decided alerts

        entries.append(BotEntry(
            level=a.level, direction=a.direction,
            line_price=a.line_price, hit_idx=hit_idx,
            hit_ns=int(dc.full_ts_ns[hit_idx]),
            human_score=sc, entry_count=a.level_test_count,
        ))

    return entries, cw, cl


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(entry, dc, target_pts, stop_pts, timeout=900):
    eod_ns = _eod_cutoff_ns(dc.date)
    out, eidx, pnl = evaluate_bot_trade(
        entry.hit_idx, entry.line_price, entry.direction,
        dc.full_ts_ns, dc.full_prices,
        target_pts, stop_pts, timeout, eod_ns,
    )
    return out, int(dc.full_ts_ns[eidx]), pnl * MULTIPLIER


# ══════════════════════════════════════════════════════════════════════════════
# PER-LEVEL TARGET LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

# Will be populated by Stage 0.
BEST_TARGET = {}


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    outcome: str
    pnl_usd: float


def replay(days, data, min_score, target_fn, stop_pts, max_per_level=5, daily_loss=150.0, max_consec=3):
    """Replay with 1-position constraint.
    target_fn: either a fixed number or a callable(level) -> target.
    """
    trades = []
    for date in days:
        d = data.get(date)
        if not d: continue
        entries, outcomes = d
        eod = _eod_cutoff_ns(date)
        pos_exit = 0; dpnl = 0.0; dcons = 0; stopped = False; lc = {}
        for i, e in enumerate(entries):
            if stopped: break
            if e.hit_ns >= eod: break
            if e.hit_ns < pos_exit: continue
            if e.human_score < min_score: continue
            lv = lc.get(e.level, 0)
            if lv >= max_per_level: continue
            out, exit_ns, pnl = outcomes[i]
            pos_exit = exit_ns
            lc[e.level] = lv + 1
            trades.append(Trade(date, e.level, out, pnl))
            dpnl += pnl
            if pnl < 0: dcons += 1
            else: dcons = 0
            if daily_loss and dpnl <= -daily_loss: stopped = True
            if max_consec and dcons >= max_consec: stopped = True
    return trades


def fmt(trades, nd, label=""):
    if not trades: return f"  {label:>50s}  no trades"
    w = sum(1 for t in trades if t.outcome == "win")
    l = sum(1 for t in trades if t.outcome == "loss")
    o = len(trades) - w - l
    d = w + l; wr = w/d*100 if d else 0
    pnl = sum(t.pnl_usd for t in trades); ppd = pnl/nd
    eq = STARTING_BALANCE; peak = eq; dd = 0.0
    for t in trades:
        eq += t.pnl_usd; peak = max(peak, eq); dd = max(dd, peak - eq)
    aw = sum(t.pnl_usd for t in trades if t.outcome == "win")/w if w else 0
    al = sum(t.pnl_usd for t in trades if t.outcome == "loss")/l if l else 0
    return (
        f"  {label:>50s}  {len(trades):>4} ({len(trades)/nd:.1f}/d) "
        f"{w}W/{l}L/{o}O {wr:>5.1f}%  "
        f"W:{aw:>+6.1f} L:{al:>+6.1f}  "
        f"PnL {pnl:>+8,.0f} ({ppd:>+5.1f}/d)  DD {dd:>5,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 110)
    print("  BOT V2 — Correct entry matching, per-level targets, walk-forward")
    print("=" * 110)

    days = load_cached_days()
    day_caches = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None: day_caches[date] = dc
        except: pass
    valid_days = sorted(day_caches.keys())
    N = len(valid_days)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    # Step 1: Get bot entries (human alert → line touch).
    human_w = Weights()
    print(f"  Computing entries...", flush=True)
    t1 = time.time()
    entries_by_date = {}
    total = 0
    cw = cl = 0
    for i, date in enumerate(valid_days):
        entries, cw, cl = get_bot_entries(day_caches[date], human_w, cw, cl)
        entries_by_date[date] = entries
        total += len(entries)
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{N}... ({total} entries)", flush=True)
    print(f"  {total} entries ({total/N:.1f}/day) in {time.time()-t1:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 0: Per-level optimal target (S20 stop, sweep T5-T14)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 0: Optimal target per level (all entries, S20 stop)")
    print(f"{'='*110}\n")

    LEVELS = ["IBH", "IBL", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272", "VWAP"]
    TARGETS = [5, 6, 7, 8, 10, 12, 14]

    print(f"  {'Level':>20s}", end="")
    for t in TARGETS:
        print(f"  {'T'+str(t):>12s}", end="")
    print()

    for lv in LEVELS:
        print(f"  {lv:>20s}", end="")
        best_ppd = -999; best_t = 8
        for tgt in TARGETS:
            lw = ll = 0; lpnl = 0.0
            for date in valid_days:
                for e in entries_by_date[date]:
                    if e.level != lv: continue
                    out, _, pnl = evaluate(e, day_caches[date], float(tgt), 20.0)
                    lpnl += pnl
                    if out == "win": lw += 1
                    elif out == "loss": ll += 1
            d = lw + ll
            wr = lw/d*100 if d else 0
            ppd = lpnl/N
            print(f"  {wr:>5.1f}%${ppd:>+5.1f}", end="")
            if ppd > best_ppd: best_ppd = ppd; best_t = tgt
        BEST_TARGET[lv] = best_t
        print(f"  → T{best_t}")

    print(f"\n  Per-level targets: {BEST_TARGET}")

    # Step 2: Precompute outcomes for each exit config.
    CONFIGS = {
        "T8/S20 (human baseline)": lambda lv: (8, 20),
        "T8/S5": lambda lv: (8, 5),
        "T8/S7": lambda lv: (8, 7),
        "T10/S7": lambda lv: (10, 7),
        "T10/S20": lambda lv: (10, 20),
        "per-level/S5": lambda lv: (BEST_TARGET.get(lv, 8), 5),
        "per-level/S7": lambda lv: (BEST_TARGET.get(lv, 8), 7),
        "per-level/S10": lambda lv: (BEST_TARGET.get(lv, 8), 10),
        "per-level/S20": lambda lv: (BEST_TARGET.get(lv, 8), 20),
    }

    print(f"\n  Computing outcomes for {len(CONFIGS)} configs...", flush=True)
    data = {}  # date → { cfg_name → [(out, exit_ns, pnl), ...] }
    for date in valid_days:
        entries = entries_by_date[date]
        dc = day_caches[date]
        cfg_outcomes = {}
        for cfg_name, ts_fn in CONFIGS.items():
            outs = []
            for e in entries:
                tgt, stp = ts_fn(e.level)
                outs.append(evaluate(e, dc, float(tgt), float(stp)))
            cfg_outcomes[cfg_name] = outs
        data[date] = cfg_outcomes
    print(f"  Done in {time.time()-t1:.0f}s")

    # Helper to package data for replay.
    def make_replay_data(cfg_name):
        rd = {}
        for date in valid_days:
            rd[date] = (entries_by_date[date], data[date][cfg_name])
        return rd

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: In-sample (all N days)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 1: In-sample (all {N} days, $150/3 risk)")
    print(f"{'='*110}\n")

    is_results = []
    for cfg_name in CONFIGS:
        rd = make_replay_data(cfg_name)
        for min_score in [-2, 0, 2, 3, 4, 5]:
            trades = replay(valid_days, rd, min_score, None, None)
            pnl = sum(t.pnl_usd for t in trades)
            ppd = pnl / N
            label = f"{cfg_name} score>={min_score}"
            is_results.append((label, trades, ppd))

    is_results.sort(key=lambda x: x[2], reverse=True)
    print(f"  Top 25 in-sample:\n")
    for label, trades, ppd in is_results[:25]:
        print(fmt(trades, N, label))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward OOS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 2: Walk-forward OOS")
    print(f"{'='*110}")

    # Top 15 configs.
    seen = set(); wf_labels = []
    for label, _, _ in is_results:
        if label not in seen:
            seen.add(label)
            wf_labels.append(label)
        if len(wf_labels) >= 15: break
    bl = "T8/S20 (human baseline) score>=5"
    if bl not in seen: wf_labels.append(bl)

    oos = {}; oos_days = 0; k = INITIAL_TRAIN_DAYS; wins = 0
    while k < N:
        test = valid_days[k:k+STEP_DAYS]
        if not test: break
        wins += 1; oos_days += len(test)
        for label in wf_labels:
            parts = label.rsplit(" score>=", 1)
            cfg_name = parts[0]; min_score = int(parts[1])
            rd = {}
            for date in test:
                rd[date] = (entries_by_date[date], data[date][cfg_name])
            trades = replay(test, rd, min_score, None, None)
            oos.setdefault(label, []).extend(trades)
        k += STEP_DAYS

    print(f"\n  {wins} windows, {oos_days} OOS days\n")
    oos_r = [(l, t, sum(x.pnl_usd for x in t)/oos_days) for l, t in oos.items()]
    oos_r.sort(key=lambda x: x[2], reverse=True)
    for l, t, p in oos_r:
        print(fmt(t, oos_days, f"OOS: {l}"))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 3: Recent 60 days")
    print(f"{'='*110}")
    recent = valid_days[-60:]; rn = len(recent)
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")
    for l, _, _ in oos_r[:10]:
        name = l.replace("OOS: ", "")
        parts = name.rsplit(" score>=", 1)
        cfg_name = parts[0]; min_score = int(parts[1])
        rd = {}
        for date in recent:
            rd[date] = (entries_by_date[date], data[date][cfg_name])
        trades = replay(recent, rd, min_score, None, None)
        print(fmt(trades, rn, f"RECENT: {name}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
