"""
bot_v4_backtest.py — Bot scoring with ALL human factors + bot extras.

Previous versions were missing key factors (streak, tick sweet spot,
session move buckets, entry count #1/#5). This version includes every
factor the human scoring uses, with the same bucket boundaries, PLUS
approach-dynamic factors unique to the bot. Streak tracked during replay.

Usage:
    python -u bot_v4_backtest.py
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
    DayCache, load_cached_days, load_day, preprocess_day, _run_zone_numpy,
)
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, FEE_PTS, evaluate_bot_trade
from score_optimizer import suggest_weight
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS

_ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY — all factors computed at 1-pt entry point
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Entry:
    global_idx: int
    level: str
    direction: str
    entry_count: int
    line_price: float
    entry_ns: int
    # ALL human factors (same as score_optimizer.py)
    now_et_mins: int
    tick_rate: float
    session_move_pts: float  # points, not % — matches human buckets
    range_30m_pts: float     # points — for vol penalty (human uses >75 pts)
    # Bot-only factors
    approach_speed: float
    tick_density_10s: float
    secs_since_last_exit: float


def compute_day(dc: DayCache) -> list[Entry]:
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    fp = dc.full_prices
    ft = dc.full_ts_ns
    nf = len(fp)
    first_price = float(prices[0])

    # Tick rate (3-min window).
    tick_rates = np.zeros(nf, dtype=np.float64)
    left = start
    for i in range(start, nf):
        w = ft[i] - np.int64(180_000_000_000)
        while left < i and ft[left] < w: left += 1
        tick_rates[i] = (i - left) / 3.0

    # 30-min range in POINTS (human uses >75 pts threshold).
    range_30m = np.zeros(nf, dtype=np.float64)
    for i in range(start, nf):
        ws = int(np.searchsorted(ft, ft[i] - np.int64(1_800_000_000_000), side="left"))
        if ws < i:
            wp = fp[ws:i+1]
            range_30m[i] = float(np.max(wp) - np.min(wp))

    # Approach speed + tick density (10s).
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

    levels = [
        ("IBH", np.full(n, dc.ibh), False),
        ("IBL", np.full(n, dc.ibl), False),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi), False),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo), False),
        ("VWAP", dc.post_ib_vwaps, True),
    ]

    all_entries = []
    for lname, larr, drifts in levels:
        in_zone = False; ref = 0.0; ec = 0; last_exit = 0
        for j in range(n):
            pj = prices[j]; lj = larr[j]; gidx = start + j
            if in_zone:
                er = lj if drifts else ref
                if abs(pj - er) > 20.0:
                    in_zone = False
                    last_exit = int(ft[gidx])
            else:
                if abs(pj - lj) <= 1.0:
                    in_zone = True; ref = lj; ec += 1
                    ens = int(ft[gidx])
                    ep = float(fp[gidx])
                    direction = "up" if ep > lj else "down"
                    secs_since = (ens - last_exit) / 1e9 if last_exit > 0 else 99999.0
                    all_entries.append(Entry(
                        global_idx=gidx, level=lname, direction=direction,
                        entry_count=ec, line_price=float(lj), entry_ns=ens,
                        now_et_mins=int(et_mins[gidx]),
                        tick_rate=float(tick_rates[gidx]),
                        session_move_pts=ep - first_price,
                        range_30m_pts=float(range_30m[gidx]),
                        approach_speed=float(approach_speed[gidx]),
                        tick_density_10s=float(tick_density[gidx]),
                        secs_since_last_exit=secs_since,
                    ))

    all_entries.sort(key=lambda e: e.entry_ns)
    return all_entries


# ══════════════════════════════════════════════════════════════════════════════
# SCORING — ALL human factors (same buckets) + bot extras
# ══════════════════════════════════════════════════════════════════════════════

# 28 weight fields: 5 level + 9 combo + 1 time + 2 tick + 4 test +
#                   5 session_move + 2 streak + 1 vol + 3 approach + 1 density

WEIGHT_NAMES = [
    # Level (5)
    "lv_ibh", "lv_ibl", "lv_fib_hi", "lv_fib_lo", "lv_vwap",
    # Combos (9)
    "co_fib_hi_up", "co_fib_lo_down", "co_ibl_down", "co_vwap_up",
    "co_ibh_up", "co_ibl_up", "co_fib_lo_up", "co_fib_hi_down", "co_vwap_down",
    # Time (1)
    "time_power",
    # Tick rate (2) — human sweet spot 1750-2000, plus bot low-tick
    "tick_sweet", "tick_low",
    # Test count (4) — same as human: #1, #2, #3, #5
    "test_1", "test_2", "test_3", "test_5",
    # Session move (5) — same buckets as human (points-based)
    "move_sweet_green", "move_sweet_red", "move_strong_red",
    "move_near_zero_green", "move_strong_green",
    # Streak (2) — same as human
    "streak_win", "streak_loss",
    # Volatility (1) — human uses >75 pts
    "vol_high",
    # Bot-only (3)
    "approach_fast", "approach_vfast", "fresh_test",
]


def score_entry(e: Entry, weights: dict, cw: int, cl: int) -> int:
    s = 0
    # Level
    lv_map = {"IBH": "lv_ibh", "IBL": "lv_ibl", "FIB_EXT_HI_1.272": "lv_fib_hi",
              "FIB_EXT_LO_1.272": "lv_fib_lo", "VWAP": "lv_vwap"}
    s += weights.get(lv_map.get(e.level, ""), 0)

    # Direction combo
    combo_map = {
        ("FIB_EXT_HI_1.272","up"): "co_fib_hi_up",
        ("FIB_EXT_LO_1.272","down"): "co_fib_lo_down",
        ("IBL","down"): "co_ibl_down",
        ("VWAP","up"): "co_vwap_up",
        ("IBH","up"): "co_ibh_up",
        ("IBL","up"): "co_ibl_up",
        ("FIB_EXT_LO_1.272","up"): "co_fib_lo_up",
        ("FIB_EXT_HI_1.272","down"): "co_fib_hi_down",
        ("VWAP","down"): "co_vwap_down",
    }
    s += weights.get(combo_map.get((e.level, e.direction), ""), 0)

    # Time — power hour (same as human: >= 15:00 ET)
    if e.now_et_mins >= 900:
        s += weights.get("time_power", 0)

    # Tick rate — human sweet spot 1750-2000
    if 1750 <= e.tick_rate < 2000:
        s += weights.get("tick_sweet", 0)
    elif e.tick_rate < 500:
        s += weights.get("tick_low", 0)

    # Test count — same as human: #1, #2, #3, #5
    if e.entry_count == 1: s += weights.get("test_1", 0)
    elif e.entry_count == 2: s += weights.get("test_2", 0)
    elif e.entry_count == 3: s += weights.get("test_3", 0)
    elif e.entry_count == 5: s += weights.get("test_5", 0)

    # Session move — same POINT-BASED buckets as human
    m = e.session_move_pts
    if 10 < m <= 20: s += weights.get("move_sweet_green", 0)
    elif -20 < m <= -10: s += weights.get("move_sweet_red", 0)
    elif m <= -50: s += weights.get("move_strong_red", 0)
    elif 0 < m <= 10: s += weights.get("move_near_zero_green", 0)
    elif m > 50: s += weights.get("move_strong_green", 0)

    # Streak — same as human: 2+ wins or 2+ losses
    if cw >= 2: s += weights.get("streak_win", 0)
    elif cl >= 2: s += weights.get("streak_loss", 0)

    # Volatility — human uses 30m range > 75 pts
    if e.range_30m_pts > 75.0:
        s += weights.get("vol_high", 0)

    # Bot-only: approach speed
    if e.approach_speed > 3.0: s += weights.get("approach_vfast", 0)
    elif e.approach_speed > 1.5: s += weights.get("approach_fast", 0)

    # Bot-only: fresh test
    if e.secs_since_last_exit > 600:
        s += weights.get("fresh_test", 0)

    return s


def train_weights(entries_outcomes: list[tuple[Entry, str, int, int]]) -> dict:
    """Train weights from (entry, outcome, consecutive_wins, consecutive_losses).

    Returns dict of weight_name → int.
    """
    if not entries_outcomes: return {}
    total = len(entries_outcomes)
    wc = sum(1 for _, o, _, _ in entries_outcomes if o == "win")
    bl = wc / total * 100
    sw = suggest_weight

    def wr(fn):
        sub = [(e,o,cw,cl) for e,o,cw,cl in entries_outcomes if fn(e,cw,cl)]
        if len(sub) < 30: return bl
        return sum(1 for _,o,_,_ in sub if o == "win") / len(sub) * 100

    w = {}
    # Level
    w["lv_ibh"] = sw(wr(lambda e,cw,cl: e.level=="IBH"), bl)
    w["lv_ibl"] = sw(wr(lambda e,cw,cl: e.level=="IBL"), bl)
    w["lv_fib_hi"] = sw(wr(lambda e,cw,cl: e.level=="FIB_EXT_HI_1.272"), bl)
    w["lv_fib_lo"] = sw(wr(lambda e,cw,cl: e.level=="FIB_EXT_LO_1.272"), bl)
    w["lv_vwap"] = sw(wr(lambda e,cw,cl: e.level=="VWAP"), bl)
    # Combos
    w["co_fib_hi_up"] = sw(wr(lambda e,cw,cl: e.level=="FIB_EXT_HI_1.272" and e.direction=="up"), bl)
    w["co_fib_lo_down"] = sw(wr(lambda e,cw,cl: e.level=="FIB_EXT_LO_1.272" and e.direction=="down"), bl)
    w["co_ibl_down"] = sw(wr(lambda e,cw,cl: e.level=="IBL" and e.direction=="down"), bl)
    w["co_vwap_up"] = sw(wr(lambda e,cw,cl: e.level=="VWAP" and e.direction=="up"), bl)
    w["co_ibh_up"] = sw(wr(lambda e,cw,cl: e.level=="IBH" and e.direction=="up"), bl)
    w["co_ibl_up"] = sw(wr(lambda e,cw,cl: e.level=="IBL" and e.direction=="up"), bl)
    w["co_fib_lo_up"] = sw(wr(lambda e,cw,cl: e.level=="FIB_EXT_LO_1.272" and e.direction=="up"), bl)
    w["co_fib_hi_down"] = sw(wr(lambda e,cw,cl: e.level=="FIB_EXT_HI_1.272" and e.direction=="down"), bl)
    w["co_vwap_down"] = sw(wr(lambda e,cw,cl: e.level=="VWAP" and e.direction=="down"), bl)
    # Time
    w["time_power"] = sw(wr(lambda e,cw,cl: e.now_et_mins >= 900), bl)
    # Tick rate
    w["tick_sweet"] = sw(wr(lambda e,cw,cl: 1750 <= e.tick_rate < 2000), bl)
    w["tick_low"] = sw(wr(lambda e,cw,cl: e.tick_rate < 500), bl)
    # Test count
    w["test_1"] = sw(wr(lambda e,cw,cl: e.entry_count == 1), bl)
    w["test_2"] = sw(wr(lambda e,cw,cl: e.entry_count == 2), bl)
    w["test_3"] = sw(wr(lambda e,cw,cl: e.entry_count == 3), bl)
    w["test_5"] = sw(wr(lambda e,cw,cl: e.entry_count == 5), bl)
    # Session move (point-based, same as human)
    w["move_sweet_green"] = sw(wr(lambda e,cw,cl: 10 < e.session_move_pts <= 20), bl)
    w["move_sweet_red"] = sw(wr(lambda e,cw,cl: -20 < e.session_move_pts <= -10), bl)
    w["move_strong_red"] = sw(wr(lambda e,cw,cl: e.session_move_pts <= -50), bl)
    w["move_near_zero_green"] = sw(wr(lambda e,cw,cl: 0 < e.session_move_pts <= 10), bl)
    w["move_strong_green"] = sw(wr(lambda e,cw,cl: e.session_move_pts > 50), bl)
    # Streak
    w["streak_win"] = sw(wr(lambda e,cw,cl: cw >= 2), bl)
    w["streak_loss"] = sw(wr(lambda e,cw,cl: cl >= 2), bl)
    # Volatility
    w["vol_high"] = sw(wr(lambda e,cw,cl: e.range_30m_pts > 75.0), bl)
    # Bot-only
    w["approach_fast"] = sw(wr(lambda e,cw,cl: 1.5 < e.approach_speed <= 3.0), bl)
    w["approach_vfast"] = sw(wr(lambda e,cw,cl: e.approach_speed > 3.0), bl)
    w["fresh_test"] = sw(wr(lambda e,cw,cl: e.secs_since_last_exit > 600), bl)

    return w


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY with streak tracking
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
    cw = cl = 0  # streak tracked across days
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
            sc = score_entry(e, weights, cw, cl)
            if sc < min_score: continue
            out, exit_ns, pnl = outs[i]
            pos_exit = exit_ns
            lc[e.level] = lv + 1
            trades.append(Trade(date, e.level, out, pnl))
            dpnl += pnl
            # Update streak.
            if pnl >= 0: cw += 1; cl = 0
            else: cw = 0; cl += 1
            # Risk limits.
            if pnl < 0: dcons += 1
            else: dcons = 0
            if daily_loss and dpnl <= -daily_loss: stopped = True
            if max_consec and dcons >= max_consec: stopped = True
    return trades


def collect_training_data(days, entries_by_date, outcomes_by_date):
    """Collect (entry, outcome, cw, cl) for weight training, tracking streak."""
    data = []
    cw = cl = 0
    for date in days:
        entries = entries_by_date.get(date, [])
        outs = outcomes_by_date.get(date, [])
        for i, e in enumerate(entries):
            out, _, pnl = outs[i]
            data.append((e, out, cw, cl))
            if pnl >= 0: cw += 1; cl = 0
            else: cw = 0; cl += 1
    return data


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
    print("  BOT V4 — All human factors + streak + bot extras, walk-forward")
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
    print(f"  Computing entries...", flush=True)
    t1 = time.time()
    entries_by_date = {}
    total = 0
    for i, date in enumerate(valid_days):
        entries_by_date[date] = compute_day(day_caches[date])
        total += len(entries_by_date[date])
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{N}... ({total} entries)", flush=True)
    print(f"  {total} entries ({total/N:.1f}/day) in {time.time()-t1:.0f}s")

    # Per-level optimal target (Stage 0).
    print(f"\n  Per-level target sweep (S20)...")
    LEVELS = ["IBH","IBL","FIB_EXT_HI_1.272","FIB_EXT_LO_1.272","VWAP"]
    TGTS = [5,6,7,8,10,12,14]
    best_tgt = {}
    for lv in LEVELS:
        best_ppd = -999; bt = 8
        for tgt in TGTS:
            lpnl = 0.0
            for date in valid_days:
                dc = day_caches[date]; eod = _eod_cutoff_ns(date)
                for e in entries_by_date[date]:
                    if e.level != lv: continue
                    _,_,pnl = evaluate_bot_trade(e.global_idx, e.line_price, e.direction,
                                                  dc.full_ts_ns, dc.full_prices, float(tgt), 20.0, 900, eod)
                    lpnl += pnl * MULTIPLIER
            ppd = lpnl / N
            if ppd > best_ppd: best_ppd = ppd; bt = tgt
        best_tgt[lv] = bt
    print(f"  Per-level targets: {best_tgt}")

    # Precompute outcomes.
    CFGS = {
        "T8/S20": lambda lv: (8, 20),
        "per-level/S20": lambda lv: (best_tgt.get(lv, 8), 20),
        "per-level/S7": lambda lv: (best_tgt.get(lv, 8), 7),
    }
    outcomes = {}  # cfg → date → [(out, exit_ns, pnl)]
    for cfg, fn in CFGS.items():
        obd = {}
        for date in valid_days:
            dc = day_caches[date]; eod = _eod_cutoff_ns(date)
            outs = []
            for e in entries_by_date[date]:
                tgt, stp = fn(e.level)
                out, eidx, pnl = evaluate_bot_trade(
                    e.global_idx, e.line_price, e.direction,
                    dc.full_ts_ns, dc.full_prices, float(tgt), float(stp), 900, eod)
                outs.append((out, int(dc.full_ts_ns[eidx]), pnl*MULTIPLIER))
            obd[date] = outs
        outcomes[cfg] = obd

    # Train full-data weights + factor analysis.
    train_data = collect_training_data(valid_days, entries_by_date, outcomes["T8/S20"])
    w_full = train_weights(train_data)

    print(f"\n  Factor analysis (T8/S20, baseline {sum(1 for _,o,_,_ in train_data if o=='win')/len(train_data)*100:.1f}% WR):")
    print(f"  Trained weights:")
    for k in WEIGHT_NAMES:
        v = w_full.get(k, 0)
        if v != 0: print(f"    {k}: {v:+d}")

    # Score distribution.
    print(f"\n  Score distribution (T8/S20, full weights + streak):")
    print(f"  {'Thr':>5} {'N':>6} {'WR%':>6} {'/day':>5} {'$/day':>7}")
    cw = cl = 0
    scored = []
    for e, out, _, _ in train_data:
        sc = score_entry(e, w_full, cw, cl)
        scored.append((sc, out))
        if out == "win": cw += 1; cl = 0
        else: cw = 0; cl += 1
    for thr in range(-4, 12):
        sub = [(s, o) for s, o in scored if s >= thr]
        if len(sub) < 30: continue
        w = sum(1 for _, o in sub if o == "win")
        l = sum(1 for _, o in sub if o == "loss")
        d = w + l
        if d == 0: continue
        wr = w/d*100
        pnl = w*(8-FEE_PTS)*MULTIPLIER + l*(-(20+FEE_PTS))*MULTIPLIER
        print(f"  {'>=' + str(thr):>5} {len(sub):>6} {wr:>5.1f}% {len(sub)/N:>5.1f} {pnl/N:>+6.1f}")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: In-sample
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 1: In-sample (all {N} days)")
    print(f"{'='*110}\n")

    is_results = []
    for cfg in CFGS:
        for ms in [-2, -1, 0, 1, 2, 3, 4, 5]:
            trades = replay(valid_days, entries_by_date, outcomes[cfg], w_full, ms)
            pnl = sum(t.pnl_usd for t in trades)
            is_results.append((f"{cfg} score>={ms}", trades, pnl/N))
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

    oos = {}; oos_days = 0; k = INITIAL_TRAIN_DAYS
    while k < N:
        train = valid_days[:k]; test = valid_days[k:k+STEP_DAYS]
        if not test: break
        oos_days += len(test)
        td = collect_training_data(train, entries_by_date, outcomes["T8/S20"])
        wt = train_weights(td)
        for label in wf:
            parts = label.rsplit(" score>=", 1)
            cfg = parts[0]; ms = int(parts[1])
            trades = replay(test, entries_by_date, outcomes[cfg], wt, ms)
            oos.setdefault(label, []).extend(trades)
        k += STEP_DAYS

    print(f"\n  {oos_days} OOS days\n")
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
    td = collect_training_data(pre, entries_by_date, outcomes["T8/S20"])
    wr = train_weights(td)
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")
    for l, _, _ in oos_r[:10]:
        name = l.replace("OOS: ", "")
        parts = name.rsplit(" score>=", 1)
        cfg = parts[0]; ms = int(parts[1])
        trades = replay(recent, entries_by_date, outcomes[cfg], wr, ms)
        print(fmt(trades, rn, f"RECENT: {name}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
