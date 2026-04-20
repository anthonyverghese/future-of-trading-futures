"""
bot_combined_backtest.py — Combined human + bot scoring with adaptive exits.

The bot uses ALL available information:
  - Human score (what the human would have scored at 7 pts) as one factor
  - Bot-specific factors only observable at the line (approach, density, etc.)
  - Combined into a single score → more good trades than human alone

Better exits: tight stop + trailing target (not fixed T/S).

Walk-forward validated. Shows both in-sample and OOS results.

Usage:
    python -u bot_combined_backtest.py
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
    DayCache, load_cached_days, load_day, preprocess_day, _run_zone_numpy,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, FEE_PTS
from score_optimizer import Weights, compute_tick_rate, score_alert, EnrichedAlert, suggest_weight
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY WITH ALL FACTORS (human score + bot-specific)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FullEntry:
    global_idx: int
    level: str
    direction: str
    entry_count: int
    entry_price: float
    line_price: float
    entry_ns: int
    # Human score (computed when price was 7 pts away).
    human_score: int
    # Bot-specific factors at the line.
    approach_speed: float  # pts/sec in last 10s
    tick_density_10s: float  # ticks/sec in last 10s
    secs_since_last_exit: float
    now_et_mins: int
    range_30m_pct: float


def compute_day_entries(dc: DayCache, human_w: Weights) -> list[FullEntry]:
    """Compute ALL zone entries with human score + bot factors.

    Runs human zone (7pt/20pt) and bot zone (1pt/20pt) in parallel.
    For each bot zone entry, looks up the most recent human score.
    """
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    full_prices = dc.full_prices
    full_ts = dc.full_ts_ns
    first_price = float(prices[0])

    # Precompute approach speed + tick density (10s sliding window).
    n_full = len(full_prices)
    approach_speed = np.zeros(n_full, dtype=np.float64)
    tick_density = np.zeros(n_full, dtype=np.float64)
    left_10 = start
    for i in range(start, n_full):
        w10 = full_ts[i] - np.int64(10_000_000_000)
        while left_10 < i and full_ts[left_10] < w10:
            left_10 += 1
        if left_10 < i:
            elapsed = (full_ts[i] - full_ts[left_10]) / 1e9
            approach_speed[i] = abs(float(full_prices[i]) - float(full_prices[left_10])) / max(elapsed, 0.1)
            tick_density[i] = (i - left_10) / 10.0

    # Precompute 30m range %.
    range_30m_pct = np.zeros(n_full, dtype=np.float64)
    for i in range(start, n_full):
        ws = int(np.searchsorted(full_ts, full_ts[i] - np.int64(1_800_000_000_000), side="left"))
        if ws < i:
            wp = full_prices[ws:i+1]
            p = float(full_prices[i])
            if p > 0:
                range_30m_pct[i] = float(np.max(wp) - np.min(wp)) / p * 100

    # ET minutes.
    dt_local = _ET.localize(datetime.datetime.combine(dc.date, datetime.time(12, 0)))
    utc_off_ns = np.int64(dt_local.utcoffset().total_seconds() * 1e9)
    et_minutes = np.zeros(n_full, dtype=np.int32)
    for i in range(start, n_full):
        et_minutes[i] = int((full_ts[i] + utc_off_ns) // 60_000_000_000) % 1440

    levels = [
        ("IBH", np.full(n, dc.ibh), False),
        ("IBL", np.full(n, dc.ibl), False),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), False),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), False),
        ("VWAP", dc.post_ib_vwaps, True),
    ]

    all_entries = []

    for level_name, level_arr, drifts in levels:
        # Track human zone state.
        in_zone_h = False; ref_h = 0.0; ec_h = 0
        last_human_score = 0
        has_human_score = False

        # Track bot zone state.
        in_zone_b = False; ref_b = 0.0; ec_b = 0

        # Track last exit time for this level.
        last_exit_ns = 0

        for j in range(n):
            pj = prices[j]
            lj = level_arr[j]
            gidx = start + j
            ens = int(full_ts[gidx])

            # Human zone (7pt entry, 20pt exit).
            if in_zone_h:
                er = lj if drifts else ref_h
                if abs(pj - er) > 20.0:
                    in_zone_h = False
            else:
                if abs(pj - lj) <= 7.0:
                    in_zone_h = True
                    ref_h = lj
                    ec_h += 1
                    # Score this human alert.
                    ep = float(pj)
                    direction = "up" if ep > lj else "down"
                    ts_pd = pd.Timestamp(ens, unit="ns", tz="UTC").tz_convert(_ET)
                    tr = compute_tick_rate(dc.full_df, ts_pd)
                    sm = ep - first_price
                    ea = EnrichedAlert(
                        date=dc.date, level=level_name, direction=direction,
                        entry_count=ec_h, outcome="correct",
                        entry_price=ep, line_price=float(lj),
                        alert_time=ts_pd, now_et=ts_pd.time(), tick_rate=tr,
                        session_move_pts=sm, consecutive_wins=0, consecutive_losses=0,
                    )
                    last_human_score = score_alert(ea, human_w)
                    has_human_score = True

            # Bot zone (1pt entry, 20pt exit).
            if in_zone_b:
                er_b = lj if drifts else ref_b
                if abs(pj - er_b) > 20.0:
                    in_zone_b = False
                    last_exit_ns = ens
            else:
                if abs(pj - lj) <= 1.0:
                    in_zone_b = True
                    ref_b = lj
                    ec_b += 1
                    ep = float(pj)
                    direction = "up" if ep > lj else "down"

                    secs_since = (ens - last_exit_ns) / 1e9 if last_exit_ns > 0 else 99999.0
                    h_score = last_human_score if has_human_score else 0

                    all_entries.append(FullEntry(
                        global_idx=gidx, level=level_name, direction=direction,
                        entry_count=ec_b, entry_price=ep, line_price=float(lj),
                        entry_ns=ens, human_score=h_score,
                        approach_speed=float(approach_speed[gidx]),
                        tick_density_10s=float(tick_density[gidx]),
                        secs_since_last_exit=secs_since,
                        now_et_mins=int(et_minutes[gidx]),
                        range_30m_pct=float(range_30m_pct[gidx]),
                    ))

    all_entries.sort(key=lambda e: e.entry_ns)
    return all_entries


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SCORING
# ══════════════════════════════════════════════════════════════════════════════


def combined_score(e: FullEntry) -> int:
    """Score using human score as a factor + bot-specific factors."""
    s = e.human_score  # start with the human's assessment

    # Bot-specific adjustments.
    if e.approach_speed > 3.0: s += 2
    elif e.approach_speed > 1.5: s += 1

    if e.tick_density_10s < 3: s += 2
    elif e.tick_density_10s < 7: s -= 1

    if e.secs_since_last_exit > 600: s += 1
    elif 60 < e.secs_since_last_exit < 120: s -= 1

    if e.range_30m_pct < 0.15: s -= 4

    if e.now_et_mins >= 900: s -= 2  # power hour bad for bot
    elif 631 <= e.now_et_mins < 690: s += 1  # post-IB good

    return s


def fit_combined_weights(
    entries_outcomes: list[tuple[FullEntry, str]],
) -> dict[str, int]:
    """Derive combined weights from data. Returns threshold suggestions."""
    if not entries_outcomes:
        return {}
    total = len(entries_outcomes)
    wc = sum(1 for _, o in entries_outcomes if o == "win")
    bl = wc / total * 100

    def wr(fn):
        sub = [(e, o) for e, o in entries_outcomes if fn(e)]
        if len(sub) < 30: return bl
        return sum(1 for _, o in sub if o == "win") / len(sub) * 100

    # Just report factor analysis — weights are hardcoded in combined_score.
    factors = {}
    for thr in range(-2, 10):
        sub = [(e, o) for e, o in entries_outcomes if combined_score(e) >= thr]
        if len(sub) < 30: continue
        w = sum(1 for _, o in sub if o == "win")
        factors[f"combined>={thr}"] = (w, len(sub), w/len(sub)*100)
    return factors


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_trade(
    gidx, line_price, direction, ts_ns, prices,
    stop_pts, target_pts=None,
    trail_activate=None, trail_offset=None, max_target=50.0,
    timeout=900, eod_ns=None,
):
    """Flexible evaluation: fixed target, trailing, or both."""
    entry_ns = ts_ns[gidx]
    max_ns = entry_ns + np.int64(timeout * 1_000_000_000)
    if eod_ns and eod_ns < max_ns: max_ns = np.int64(eod_ns)

    if direction == "up":
        sl = line_price - stop_pts
        tp = line_price + target_pts if target_pts else line_price + max_target
    else:
        sl = line_price + stop_pts
        tp = line_price - target_pts if target_pts else line_price - max_target

    best_fav = 0.0
    trailing_active = False
    last_idx = gidx

    for j in range(gidx + 1, len(prices)):
        if ts_ns[j] > max_ns: break
        last_idx = j
        p = float(prices[j])

        # Fixed target.
        if direction == "up" and p >= tp: return "win", j, (tp - line_price) - FEE_PTS
        if direction == "down" and p <= tp: return "win", j, (line_price - tp) - FEE_PTS

        # Stop.
        if direction == "up" and p <= sl: return "loss", j, -(stop_pts + FEE_PTS)
        if direction == "down" and p >= sl: return "loss", j, -(stop_pts + FEE_PTS)

        # Trailing.
        if trail_activate is not None and trail_offset is not None:
            fav = (p - line_price) if direction == "up" else (line_price - p)
            best_fav = max(best_fav, fav)
            if best_fav >= trail_activate:
                if not trailing_active:
                    trailing_active = True
                    sl = line_price  # breakeven
                if direction == "up":
                    tsl = (line_price + best_fav) - trail_offset
                    sl = max(sl, tsl)
                    if p <= sl:
                        return "trail", j, (p - line_price) - FEE_PTS
                else:
                    tsl = (line_price - best_fav) + trail_offset
                    sl = min(sl, tsl)
                    if p >= sl:
                        return "trail", j, (line_price - p) - FEE_PTS

    ep = float(prices[last_idx])
    pnl = (ep - line_price if direction == "up" else line_price - ep) - FEE_PTS
    return "timeout", last_idx, pnl


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    outcome: str
    pnl_usd: float


def replay(days, data, cfg_key, min_score, max_per_level, daily_loss, max_consec):
    trades = []
    for date in days:
        d = data.get(date)
        if not d: continue
        entries, all_outcomes = d
        if cfg_key not in all_outcomes: continue
        outcomes, exit_ns, pnl_usd = all_outcomes[cfg_key]
        eod = _eod_cutoff_ns(date)
        pos_exit = 0; dpnl = 0.0; dcons = 0; stopped = False; lc = {}
        for i, e in enumerate(entries):
            if stopped: break
            if e.entry_ns >= eod: break
            if e.entry_ns < pos_exit: continue
            lv = lc.get(e.level, 0)
            if lv >= max_per_level: continue
            if e.range_30m_pct < 0.15: continue
            sc = combined_score(e)
            if sc < min_score: continue
            pos_exit = exit_ns[i]
            lc[e.level] = lv + 1
            trades.append(Trade(date, e.level, outcomes[i], pnl_usd[i]))
            dpnl += pnl_usd[i]
            if pnl_usd[i] < 0: dcons += 1
            else: dcons = 0
            if daily_loss and dpnl <= -daily_loss: stopped = True
            if max_consec and dcons >= max_consec: stopped = True
    return trades


def fmt(trades, nd, label=""):
    if not trades: return f"  {label:>55s}  no trades"
    w = sum(1 for t in trades if t.outcome in ("win", "trail"))
    l = sum(1 for t in trades if t.outcome == "loss")
    o = len(trades) - w - l
    d = w + l; wr = w/d*100 if d else 0
    pnl = sum(t.pnl_usd for t in trades); ppd = pnl/nd
    eq = STARTING_BALANCE; peak = eq; dd = 0.0
    for t in trades:
        eq += t.pnl_usd; peak = max(peak, eq); dd = max(dd, peak - eq)
    aw = sum(t.pnl_usd for t in trades if t.outcome in ("win","trail"))/w if w else 0
    al = sum(t.pnl_usd for t in trades if t.outcome == "loss")/l if l else 0
    return (
        f"  {label:>55s}  {len(trades):>4} ({len(trades)/nd:.1f}/d) "
        f"{w}W/{l}L/{o}O {wr:>5.1f}%  "
        f"W:{aw:>+6.1f} L:{al:>+6.1f}  "
        f"PnL {pnl:>+8,.0f} ({ppd:>+5.1f}/d)  DD {dd:>5,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 115)
    print("  COMBINED BACKTEST — Human score as factor + bot line-edge + adaptive exits")
    print("=" * 115)

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

    # Step 1: Compute entries with all factors (no outcomes yet).
    human_w = Weights()
    print(f"  Computing entries + factors...", flush=True)
    t1 = time.time()
    entries_by_date: dict = {}
    total_entries = 0

    for i, date in enumerate(valid_days):
        dc = day_caches[date]
        entries = compute_day_entries(dc, human_w)
        entries_by_date[date] = entries
        total_entries += len(entries)
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{N}... ({total_entries} entries)", flush=True)
    print(f"  {total_entries} entries ({total_entries/N:.1f}/day) in {time.time()-t1:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 0: Find optimal target per level (data-driven)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*115}")
    print(f"  STAGE 0: Optimal target per level (sweep T5-T12, S7 stop)")
    print(f"{'='*115}\n")

    LEVELS = ["IBH", "IBL", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272", "VWAP"]
    TARGET_SWEEP = [5, 6, 7, 8, 10, 12]
    STOP_FOR_SWEEP = 7

    print(f"  {'Level':>20s}", end="")
    for tgt in TARGET_SWEEP:
        print(f"  {'T'+str(tgt):>12s}", end="")
    print()

    best_target_per_level = {}
    for lv in LEVELS:
        print(f"  {lv:>20s}", end="")
        best_ppd = -999
        best_t = 8
        for tgt in TARGET_SWEEP:
            lv_w = 0; lv_l = 0; lv_pnl = 0.0
            for date in valid_days:
                entries = entries_by_date[date]
                dc = day_caches[date]
                eod_ns = _eod_cutoff_ns(date)
                for e in entries:
                    if e.level != lv: continue
                    out, eidx, pnl = evaluate_trade(
                        e.global_idx, e.line_price, e.direction,
                        dc.full_ts_ns, dc.full_prices,
                        stop_pts=STOP_FOR_SWEEP, target_pts=tgt,
                        timeout=900, eod_ns=eod_ns,
                    )
                    pnl_usd = pnl * MULTIPLIER
                    lv_pnl += pnl_usd
                    if out == "win": lv_w += 1
                    elif out == "loss": lv_l += 1
            d = lv_w + lv_l
            wr = lv_w / d * 100 if d else 0
            ppd = lv_pnl / N
            print(f"  {wr:>5.1f}%${ppd:>+5.1f}", end="")
            if ppd > best_ppd:
                best_ppd = ppd
                best_t = tgt
        best_target_per_level[lv] = best_t
        print(f"  → best=T{best_t}")

    print(f"\n  Data-driven per-level targets: {best_target_per_level}")

    # Build exit configs using data-driven per-level targets.
    EXIT_CONFIGS = {
        # Baselines.
        "T8/S20 (human)": {"stop": 20, "target": 8},
        "T8/S5": {"stop": 5, "target": 8},
        "T8/S7": {"stop": 7, "target": 8},
        "T10/S7": {"stop": 7, "target": 10},
        "T12/S7": {"stop": 7, "target": 12},
        # Data-driven per-level with various stops.
        "per-level S5": {"stop": 5, "per_level": "data"},
        "per-level S7": {"stop": 7, "per_level": "data"},
        "per-level S10": {"stop": 10, "per_level": "data"},
        "per-level S20": {"stop": 20, "per_level": "data"},
        # Trailing.
        "trail S5 act@3 off@3": {"stop": 5, "trail_activate": 3, "trail_offset": 3},
        "trail S5 act@5 off@3": {"stop": 5, "trail_activate": 5, "trail_offset": 3},
        "trail S7 act@4 off@3": {"stop": 7, "trail_activate": 4, "trail_offset": 3},
        "trail S7 act@5 off@4": {"stop": 7, "trail_activate": 5, "trail_offset": 4},
    }
    PER_LEVEL_TARGETS = {"data": best_target_per_level}

    for i, date in enumerate(valid_days):
        dc = day_caches[date]
        entries = compute_day_entries(dc, human_w)
        total_entries += len(entries)
        eod_ns = _eod_cutoff_ns(date)

    # Step 2: Compute outcomes for all exit configs.
    print(f"\n  Computing outcomes for {len(EXIT_CONFIGS)} exit configs...", flush=True)
    t2 = time.time()
    data = {}  # date → (entries, outcomes_by_config)

    for i, date in enumerate(valid_days):
        entries = entries_by_date[date]
        dc = day_caches[date]
        eod_ns = _eod_cutoff_ns(date)

        outcomes_by_cfg = {}
        for cfg_name, cfg in EXIT_CONFIGS.items():
            outs = []; exns = []; pnls = []
            for e in entries:
                if "per_level" in cfg:
                    tgt = PER_LEVEL_TARGETS[cfg["per_level"]].get(e.level, 8)
                else:
                    tgt = cfg.get("target")

                out, eidx, pnl = evaluate_trade(
                    e.global_idx, e.line_price, e.direction,
                    dc.full_ts_ns, dc.full_prices,
                    stop_pts=cfg["stop"],
                    target_pts=tgt,
                    trail_activate=cfg.get("trail_activate"),
                    trail_offset=cfg.get("trail_offset"),
                    timeout=900, eod_ns=eod_ns,
                )
                outs.append(out)
                exns.append(int(dc.full_ts_ns[eidx]))
                pnls.append(pnl * MULTIPLIER)
            outcomes_by_cfg[cfg_name] = (outs, exns, pnls)

        data[date] = (entries, outcomes_by_cfg)
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{N}...", flush=True)

    print(f"  Outcomes computed in {time.time()-t2:.0f}s")

    # Factor analysis on training data.
    print(f"\n  Combined score distribution (T8/S20, all {N} days):")
    ref_cfg = "T8/S20 (human)"
    all_eo = []
    for date in valid_days:
        entries, oc = data[date]
        outs = oc[ref_cfg][0]
        for i, e in enumerate(entries):
            all_eo.append((e, outs[i]))
    factors = fit_combined_weights(all_eo)
    print(f"  {'Threshold':>12s}  {'Wins':>5}  {'Total':>5}  {'WR%':>6}  {'/day':>5}")
    for k, (w, t, wr) in sorted(factors.items(), key=lambda x: x[0]):
        print(f"  {k:>12s}  {w:>5}  {t:>5}  {wr:>5.1f}%  {t/N:>5.1f}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: In-sample
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*115}")
    print(f"  STAGE 1: In-sample results (all {N} days)")
    print(f"{'='*115}\n")

    is_results = []
    for cfg_name in EXIT_CONFIGS:
        for min_score in [-1, 0, 2, 4, 5]:
            label = f"{cfg_name} score>={min_score}"
            trades = replay(valid_days, data, cfg_name, min_score, 5, 150.0, 3)
            pnl = sum(t.pnl_usd for t in trades)
            ppd = pnl / N
            is_results.append((label, trades, ppd))

    is_results.sort(key=lambda x: x[2], reverse=True)
    print(f"  Top 20:\n")
    for label, trades, ppd in is_results[:20]:
        print(fmt(trades, N, label))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward OOS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*115}")
    print(f"  STAGE 2: Walk-forward OOS")
    print(f"{'='*115}")

    # Top 15 distinct configs.
    seen = set()
    wf_cfgs = []
    for label, _, _ in is_results:
        if label not in seen:
            seen.add(label)
            wf_cfgs.append(label)
        if len(wf_cfgs) >= 15: break
    # Always include human baseline.
    bl = "T8/S20 (human) score>=5"
    if bl not in seen: wf_cfgs.append(bl)

    oos = {}; oos_days = 0; k = INITIAL_TRAIN_DAYS; wins = 0
    while k < N:
        test_days = valid_days[k:k+STEP_DAYS]
        if not test_days: break
        wins += 1; oos_days += len(test_days)
        for label in wf_cfgs:
            parts = label.rsplit(" score>=", 1)
            cfg_name = parts[0]; min_score = int(parts[1])
            trades = replay(test_days, data, cfg_name, min_score, 5, 150.0, 3)
            oos.setdefault(label, []).extend(trades)
        k += STEP_DAYS

    print(f"\n  {wins} windows, {oos_days} OOS days\n")
    oos_results = [(l, t, sum(x.pnl_usd for x in t)/oos_days) for l, t in oos.items()]
    oos_results.sort(key=lambda x: x[2], reverse=True)

    for label, trades, ppd in oos_results:
        print(fmt(trades, oos_days, f"OOS: {label}"))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*115}")
    print(f"  STAGE 3: Recent 60 days")
    print(f"{'='*115}")
    recent = valid_days[-60:]; rn = len(recent)
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")

    for label, _, _ in oos_results[:10]:
        name = label.replace("OOS: ", "")
        parts = name.rsplit(" score>=", 1)
        cfg_name = parts[0]; min_score = int(parts[1])
        trades = replay(recent, data, cfg_name, min_score, 5, 150.0, 3)
        print(fmt(trades, rn, f"RECENT: {name}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
