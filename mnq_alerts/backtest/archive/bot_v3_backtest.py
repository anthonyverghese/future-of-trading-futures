"""
bot_v3_backtest.py — Bot scoring trained on bot entries, per-level targets.

1. Bot zone (1-pt entry, 20-pt exit) to get entries
2. Compute ALL scoring factors at the 1-pt entry point
3. Train weights on bot entry outcomes via walk-forward
4. Include approach-dynamic factors (bot's unique data)
5. Sweep score thresholds + per-level targets
6. Show in-sample AND OOS results

Usage:
    python -u bot_v3_backtest.py
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

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache, load_cached_days, load_day, preprocess_day, _run_zone_numpy,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, FEE_PTS, evaluate_bot_trade
from score_optimizer import suggest_weight
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY WITH ALL FACTORS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Entry:
    global_idx: int
    level: str
    direction: str
    entry_count: int
    line_price: float
    entry_ns: int
    # Scoring factors (computed at 1-pt entry point)
    now_et_mins: int
    tick_rate: float
    session_move_pct: float
    range_30m_pct: float
    approach_speed: float
    tick_density_10s: float
    secs_since_last_exit: float


def compute_day(dc: DayCache):
    """Compute bot entries (1pt/20pt) with all scoring factors.

    Returns (entries, precomputed arrays for fast factor lookup).
    """
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    fp = dc.full_prices
    ft = dc.full_ts_ns
    nf = len(fp)
    first_price = float(prices[0])

    # Precompute factors as arrays (one pass each).
    # Tick rate (3-min window).
    tick_rates = np.zeros(nf, dtype=np.float64)
    left = start
    for i in range(start, nf):
        w = ft[i] - np.int64(180_000_000_000)
        while left < i and ft[left] < w: left += 1
        tick_rates[i] = (i - left) / 3.0

    # 30-min range %.
    range_30m = np.zeros(nf, dtype=np.float64)
    for i in range(start, nf):
        ws = int(np.searchsorted(ft, ft[i] - np.int64(1_800_000_000_000), side="left"))
        if ws < i:
            wp = fp[ws:i+1]
            p = float(fp[i])
            if p > 0: range_30m[i] = float(np.max(wp) - np.min(wp)) / p * 100

    # Approach speed + tick density (10s window).
    approach_speed = np.zeros(nf, dtype=np.float64)
    tick_density = np.zeros(nf, dtype=np.float64)
    left10 = start
    for i in range(start, nf):
        w10 = ft[i] - np.int64(10_000_000_000)
        while left10 < i and ft[left10] < w10: left10 += 1
        if left10 < i:
            elapsed = (ft[i] - ft[left10]) / 1e9
            approach_speed[i] = abs(float(fp[i]) - float(fp[left10])) / max(elapsed, 0.1)
            tick_density[i] = (i - left10) / 10.0

    # ET minutes.
    dt_local = _ET.localize(datetime.datetime.combine(dc.date, datetime.time(12, 0)))
    utc_off = np.int64(dt_local.utcoffset().total_seconds() * 1e9)
    et_mins = ((ft + utc_off) // 60_000_000_000 % 1440).astype(np.int32)

    # Zone entries per level.
    levels = [
        ("IBH", np.full(n, dc.ibh), False),
        ("IBL", np.full(n, dc.ibl), False),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), False),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), False),
        ("VWAP", dc.post_ib_vwaps, True),
    ]

    all_entries = []
    for lname, larr, drifts in levels:
        ents = _run_zone_numpy(prices, larr, 1.0, 20.0, use_current_exit=drifts)
        # Track last exit time for this level.
        last_exit = 0
        in_zone = False
        ref = 0.0
        ec = 0
        for j in range(n):
            pj = prices[j]
            lj = larr[j]
            gidx = start + j
            if in_zone:
                er = lj if drifts else ref
                if abs(pj - er) > 20.0:
                    in_zone = False
                    last_exit = int(ft[gidx])
            else:
                if abs(pj - lj) <= 1.0:
                    in_zone = True
                    ref = lj
                    ec += 1
                    ens = int(ft[gidx])
                    ep = float(fp[gidx])
                    direction = "up" if ep > lj else "down"
                    secs_since = (ens - last_exit) / 1e9 if last_exit > 0 else 99999.0
                    sm_pct = (ep - first_price) / first_price * 100 if first_price > 0 else 0

                    all_entries.append(Entry(
                        global_idx=gidx, level=lname, direction=direction,
                        entry_count=ec, line_price=float(lj), entry_ns=ens,
                        now_et_mins=int(et_mins[gidx]),
                        tick_rate=float(tick_rates[gidx]),
                        session_move_pct=sm_pct,
                        range_30m_pct=float(range_30m[gidx]),
                        approach_speed=float(approach_speed[gidx]),
                        tick_density_10s=float(tick_density[gidx]),
                        secs_since_last_exit=secs_since,
                    ))

    all_entries.sort(key=lambda e: e.entry_ns)
    return all_entries


# ══════════════════════════════════════════════════════════════════════════════
# SCORING — trained from bot entry data
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BotWeights:
    level: dict = field(default_factory=dict)       # level_name → weight
    combo: dict = field(default_factory=dict)        # (level, dir) → weight
    time_post_ib: int = 0
    time_power_hour: int = 0
    test_2: int = 0
    test_3: int = 0
    tick_low: int = 0       # <500
    tick_mid_low: int = 0   # 500-1000
    tick_high: int = 0      # 2500+
    move_mild_red: int = 0
    move_near_zero: int = 0
    move_strong_green: int = 0
    vol_dead: int = 0       # <0.15%
    vol_high: int = 0       # >0.50%
    approach_fast: int = 0  # >1.5 pts/s
    approach_vfast: int = 0 # >3.0 pts/s
    density_low: int = 0    # <5
    fresh_test: int = 0     # >600s
    stale_retest: int = 0   # 60-120s


def score(e: Entry, w: BotWeights) -> int:
    s = 0
    s += w.level.get(e.level, 0)
    s += w.combo.get((e.level, e.direction), 0)
    if 631 <= e.now_et_mins < 690: s += w.time_post_ib
    elif e.now_et_mins >= 900: s += w.time_power_hour
    if e.entry_count == 2: s += w.test_2
    elif e.entry_count == 3: s += w.test_3
    if e.tick_rate < 500: s += w.tick_low
    elif e.tick_rate < 1000: s += w.tick_mid_low
    elif e.tick_rate >= 2500: s += w.tick_high
    if -0.09 < e.session_move_pct <= -0.04: s += w.move_mild_red
    elif 0 < e.session_move_pct <= 0.04: s += w.move_near_zero
    elif e.session_move_pct > 0.20: s += w.move_strong_green
    if e.range_30m_pct < 0.15: s += w.vol_dead
    elif e.range_30m_pct > 0.50: s += w.vol_high
    if e.approach_speed > 3.0: s += w.approach_vfast
    elif e.approach_speed > 1.5: s += w.approach_fast
    if e.tick_density_10s < 5: s += w.density_low
    if e.secs_since_last_exit > 600: s += w.fresh_test
    elif 60 < e.secs_since_last_exit < 120: s += w.stale_retest
    return s


def train_weights(entries_outcomes: list[tuple[Entry, str]]) -> BotWeights:
    if not entries_outcomes: return BotWeights()
    total = len(entries_outcomes)
    wc = sum(1 for _, o in entries_outcomes if o == "win")
    bl = wc / total * 100
    sw = suggest_weight

    def wr(fn):
        sub = [(e, o) for e, o in entries_outcomes if fn(e)]
        if len(sub) < 30: return bl
        return sum(1 for _, o in sub if o == "win") / len(sub) * 100

    w = BotWeights()
    for lv in ["IBH","IBL","VWAP","FIB_EXT_HI_1.272","FIB_EXT_LO_1.272"]:
        w.level[lv] = sw(wr(lambda e, l=lv: e.level == l), bl)
    for lv in ["IBH","IBL","VWAP","FIB_EXT_HI_1.272","FIB_EXT_LO_1.272"]:
        for d in ["up","down"]:
            w.combo[(lv,d)] = sw(wr(lambda e, l=lv, dr=d: e.level==l and e.direction==dr), bl)
    w.time_post_ib = sw(wr(lambda e: 631 <= e.now_et_mins < 690), bl)
    w.time_power_hour = sw(wr(lambda e: e.now_et_mins >= 900), bl)
    w.test_2 = sw(wr(lambda e: e.entry_count == 2), bl)
    w.test_3 = sw(wr(lambda e: e.entry_count == 3), bl)
    w.tick_low = sw(wr(lambda e: e.tick_rate < 500), bl)
    w.tick_mid_low = sw(wr(lambda e: 500 <= e.tick_rate < 1000), bl)
    w.tick_high = sw(wr(lambda e: e.tick_rate >= 2500), bl)
    w.move_mild_red = sw(wr(lambda e: -0.09 < e.session_move_pct <= -0.04), bl)
    w.move_near_zero = sw(wr(lambda e: 0 < e.session_move_pct <= 0.04), bl)
    w.move_strong_green = sw(wr(lambda e: e.session_move_pct > 0.20), bl)
    w.vol_dead = sw(wr(lambda e: e.range_30m_pct < 0.15), bl)
    w.vol_high = sw(wr(lambda e: e.range_30m_pct > 0.50), bl)
    w.approach_fast = sw(wr(lambda e: 1.5 < e.approach_speed <= 3.0), bl)
    w.approach_vfast = sw(wr(lambda e: e.approach_speed > 3.0), bl)
    w.density_low = sw(wr(lambda e: e.tick_density_10s < 5), bl)
    w.fresh_test = sw(wr(lambda e: e.secs_since_last_exit > 600), bl)
    w.stale_retest = sw(wr(lambda e: 60 < e.secs_since_last_exit < 120), bl)
    return w


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    level: str
    outcome: str
    pnl_usd: float


def replay(days, entries_by_date, outcomes_by_date, weights, min_score,
           max_per_level=5, daily_loss=150.0, max_consec=3):
    trades = []
    for date in days:
        entries = entries_by_date.get(date, [])
        outs = outcomes_by_date.get(date, [])
        if not entries: continue
        eod = _eod_cutoff_ns(date)
        pos_exit = 0; dpnl = 0.0; dcons = 0; stopped = False; lc = {}
        for i, e in enumerate(entries):
            if stopped: break
            if e.entry_ns >= eod: break
            if e.entry_ns < pos_exit: continue
            lv = lc.get(e.level, 0)
            if lv >= max_per_level: continue
            sc = score(e, weights)
            if sc < min_score: continue
            out, exit_ns, pnl = outs[i]
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
    aw = sum(t.pnl_usd for t in trades if t.outcome=="win")/w if w else 0
    al = sum(t.pnl_usd for t in trades if t.outcome=="loss")/l if l else 0
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
    print("  BOT V3 — Bot-trained scoring + per-level targets + walk-forward")
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

    # Compute entries.
    print(f"  Computing entries + factors...", flush=True)
    t1 = time.time()
    entries_by_date = {}
    total = 0
    for i, date in enumerate(valid_days):
        entries_by_date[date] = compute_day(day_caches[date])
        total += len(entries_by_date[date])
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{N}... ({total} entries)", flush=True)
    print(f"  {total} entries ({total/N:.1f}/day) in {time.time()-t1:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 0: Per-level optimal target (sweep T5-T14, S20)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 0: Per-level optimal target (all entries, S20)")
    print(f"{'='*110}\n")

    LEVELS = ["IBH","IBL","FIB_EXT_HI_1.272","FIB_EXT_LO_1.272","VWAP"]
    TGTS = [5,6,7,8,10,12,14]
    best_tgt = {}

    print(f"  {'Level':>20s}", end="")
    for t in TGTS: print(f"  {'T'+str(t):>12s}", end="")
    print()

    for lv in LEVELS:
        print(f"  {lv:>20s}", end="")
        best_ppd = -999; bt = 8
        for tgt in TGTS:
            lw=ll=0; lpnl=0.0
            for date in valid_days:
                dc = day_caches[date]
                eod = _eod_cutoff_ns(date)
                for e in entries_by_date[date]:
                    if e.level != lv: continue
                    out,eidx,pnl = evaluate_bot_trade(
                        e.global_idx, e.line_price, e.direction,
                        dc.full_ts_ns, dc.full_prices, float(tgt), 20.0, 900, eod)
                    lpnl += pnl*MULTIPLIER
                    if out=="win": lw+=1
                    elif out=="loss": ll+=1
            d=lw+ll; wr=lw/d*100 if d else 0; ppd=lpnl/N
            print(f"  {wr:>5.1f}%${ppd:>+5.1f}", end="")
            if ppd > best_ppd: best_ppd=ppd; bt=tgt
        best_tgt[lv] = bt
        print(f"  → T{bt}")
    print(f"\n  Per-level targets: {best_tgt}")

    # Precompute outcomes for each config.
    EXIT_CONFIGS = {
        "T8/S20": (lambda lv: 8, 20),
        "per-level/S20": (lambda lv: best_tgt.get(lv, 8), 20),
        "T8/S7": (lambda lv: 8, 7),
        "per-level/S7": (lambda lv: best_tgt.get(lv, 8), 7),
        "T8/S10": (lambda lv: 8, 10),
        "per-level/S10": (lambda lv: best_tgt.get(lv, 8), 10),
    }

    print(f"\n  Computing outcomes for {len(EXIT_CONFIGS)} configs...", flush=True)
    outcomes_by_cfg = {}  # cfg → date → [(out, exit_ns, pnl)]
    for cfg_name, (tgt_fn, stp) in EXIT_CONFIGS.items():
        obd = {}
        for date in valid_days:
            dc = day_caches[date]
            eod = _eod_cutoff_ns(date)
            outs = []
            for e in entries_by_date[date]:
                tgt = tgt_fn(e.level)
                out, eidx, pnl = evaluate_bot_trade(
                    e.global_idx, e.line_price, e.direction,
                    dc.full_ts_ns, dc.full_prices, float(tgt), float(stp), 900, eod)
                outs.append((out, int(dc.full_ts_ns[eidx]), pnl*MULTIPLIER))
            obd[date] = outs
        outcomes_by_cfg[cfg_name] = obd
    print(f"  Done.")

    # Train weights on full data for factor analysis.
    ref_outs = outcomes_by_cfg["T8/S20"]
    all_eo = [(e, ref_outs[date][i][0])
              for date in valid_days for i, e in enumerate(entries_by_date[date])]
    w_full = train_weights(all_eo)

    # Factor analysis.
    total_e = len(all_eo)
    wc = sum(1 for _, o in all_eo if o == "win")
    bl = wc/total_e*100
    print(f"\n  Factor analysis (T8/S20, {total_e} entries, baseline {bl:.1f}% WR):")

    def fa(label, fn):
        sub = [(e,o) for e,o in all_eo if fn(e)]
        if len(sub) < 30: return
        w = sum(1 for _,o in sub if o=="win")
        wr = w/len(sub)*100; delta = wr-bl
        print(f"    {label:<45s} {w:>4}W/{len(sub):>5} = {wr:>5.1f}% ({delta:>+5.1f}pp) wt={suggest_weight(wr,bl):>+d}")

    for lv in LEVELS: fa(lv, lambda e,l=lv: e.level==l)
    for lv in LEVELS:
        for d in ["up","down"]:
            fa(f"{lv} × {d}", lambda e,l=lv,dr=d: e.level==l and e.direction==dr)
    fa("Post-IB", lambda e: 631<=e.now_et_mins<690)
    fa("Power hour", lambda e: e.now_et_mins>=900)
    fa("Test #2", lambda e: e.entry_count==2)
    fa("Test #3", lambda e: e.entry_count==3)
    fa("Tick <500", lambda e: e.tick_rate<500)
    fa("Tick 500-1000", lambda e: 500<=e.tick_rate<1000)
    fa("Tick 2500+", lambda e: e.tick_rate>=2500)
    fa("Move mild red", lambda e: -0.09<e.session_move_pct<=-0.04)
    fa("Move near zero green", lambda e: 0<e.session_move_pct<=0.04)
    fa("Move strong green", lambda e: e.session_move_pct>0.20)
    fa("Vol dead (<0.15%)", lambda e: e.range_30m_pct<0.15)
    fa("Vol high (>0.50%)", lambda e: e.range_30m_pct>0.50)
    fa("Approach fast (1.5-3)", lambda e: 1.5<e.approach_speed<=3.0)
    fa("Approach very fast (>3)", lambda e: e.approach_speed>3.0)
    fa("Tick density <5", lambda e: e.tick_density_10s<5)
    fa("Fresh test (>10min)", lambda e: e.secs_since_last_exit>600)
    fa("Stale retest (1-2min)", lambda e: 60<e.secs_since_last_exit<120)

    # Score distribution.
    print(f"\n  Score distribution (T8/S20, full-data weights):")
    print(f"  {'Thr':>5} {'Entries':>7} {'WR%':>6} {'/day':>5} {'$/day':>7}")
    for thr in range(-4, 10):
        sub = [(e,o) for e,o in all_eo if score(e, w_full) >= thr]
        if not sub: continue
        w = sum(1 for _,o in sub if o=="win")
        l = sum(1 for _,o in sub if o=="loss")
        d = w+l
        if d == 0: continue
        wr = w/d*100
        pnl = w*(8-FEE_PTS)*MULTIPLIER + l*(-(20+FEE_PTS))*MULTIPLIER
        print(f"  {'>=' + str(thr):>5} {len(sub):>7} {wr:>5.1f}% {len(sub)/N:>5.1f} {pnl/N:>+6.1f}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: In-sample
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 1: In-sample (all {N} days)")
    print(f"{'='*110}\n")

    is_results = []
    for cfg_name in EXIT_CONFIGS:
        obd = outcomes_by_cfg[cfg_name]
        for min_s in [-2, -1, 0, 1, 2, 3, 4]:
            trades = replay(valid_days, entries_by_date, obd, w_full, min_s)
            pnl = sum(t.pnl_usd for t in trades)
            label = f"{cfg_name} score>={min_s}"
            is_results.append((label, trades, pnl/N))

    is_results.sort(key=lambda x: x[2], reverse=True)
    print(f"  Top 20:\n")
    for l, t, p in is_results[:20]: print(fmt(t, N, l))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward OOS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 2: Walk-forward OOS")
    print(f"{'='*110}")

    seen = set(); wf = []
    for l, _, _ in is_results:
        if l not in seen: seen.add(l); wf.append(l)
        if len(wf) >= 20: break

    oos = {}; oos_days = 0; k = INITIAL_TRAIN_DAYS; wins = 0
    while k < N:
        train = valid_days[:k]
        test = valid_days[k:k+STEP_DAYS]
        if not test: break
        wins += 1; oos_days += len(test)

        # Train weights on training data.
        train_eo = [(e, outcomes_by_cfg["T8/S20"][d][i][0])
                     for d in train for i, e in enumerate(entries_by_date.get(d, []))]
        wt = train_weights(train_eo)

        for label in wf:
            parts = label.rsplit(" score>=", 1)
            cfg = parts[0]; min_s = int(parts[1])
            obd = outcomes_by_cfg[cfg]
            trades = replay(test, entries_by_date, obd, wt, min_s)
            oos.setdefault(label, []).extend(trades)
        k += STEP_DAYS

    print(f"\n  {wins} windows, {oos_days} OOS days\n")
    oos_r = [(l, t, sum(x.pnl_usd for x in t)/oos_days) for l, t in oos.items()]
    oos_r.sort(key=lambda x: x[2], reverse=True)
    for l, t, p in oos_r: print(fmt(t, oos_days, f"OOS: {l}"))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 3: Recent 60 days")
    print(f"{'='*110}")
    recent = valid_days[-60:]; rn = len(recent)
    pre = [d for d in valid_days if d < recent[0]]
    pre_eo = [(e, outcomes_by_cfg["T8/S20"][d][i][0])
              for d in pre for i, e in enumerate(entries_by_date.get(d, []))]
    wr = train_weights(pre_eo)
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")
    for l, _, _ in oos_r[:10]:
        name = l.replace("OOS: ", "")
        parts = name.rsplit(" score>=", 1)
        cfg = parts[0]; min_s = int(parts[1])
        obd = outcomes_by_cfg[cfg]
        trades = replay(recent, entries_by_date, obd, wr, min_s)
        print(fmt(trades, rn, f"RECENT: {name}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
