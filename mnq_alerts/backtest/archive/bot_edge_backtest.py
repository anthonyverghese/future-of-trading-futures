"""
bot_edge_backtest.py — Line-specific factor backtest.

The bot enters at 1 pt from the line. It sees what happened as price
approached — approach speed, momentum, tick density, time since last
test. These factors are NOT available to the human app (which decides
at 7 pts away). This backtest tests whether they can push WR above
the human's 81% and reach $40/day.

Uses T8/S20 (matching human T/S), walk-forward validated.

Usage:
    python -u bot_edge_backtest.py
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
    entry_price: float
    line_price: float
    entry_ns: int
    # Existing factors
    now_et_mins: int  # minutes since midnight ET
    tick_rate: float  # trades/min in 3-min window
    session_move_pct: float  # (price - open) / open * 100
    range_30m_pct: float  # 30-min high-low range / price * 100
    # NEW line-specific factors
    approach_speed: float  # abs(price change) in last 10 seconds (pts/sec)
    approach_direction: float  # signed price change in last 10 seconds
    tick_density_10s: float  # trades per second in last 10 seconds
    secs_since_last_exit: float  # seconds since this level's zone last exited


@dataclass
class DayFactors:
    """Precomputed per-tick scoring factors for one day. Shared across exit thresholds."""
    tick_rates: np.ndarray
    range_30m_pct: np.ndarray
    approach_speed: np.ndarray
    approach_dir: np.ndarray
    tick_density_10s: np.ndarray
    et_minutes: np.ndarray  # ET minutes since midnight per tick
    session_move_pct: np.ndarray
    first_price: float


def precompute_factors(dc: DayCache) -> DayFactors:
    """Precompute all per-tick scoring factors once per day. O(n) per factor."""
    full_prices = dc.full_prices
    full_ts = dc.full_ts_ns
    n = len(full_prices)
    start = dc.post_ib_start_idx
    first_price = float(dc.post_ib_prices[0])

    # Tick rate (3-min sliding window, O(n)).
    tick_rates = np.zeros(n, dtype=np.float64)
    left = start
    for right in range(start, n):
        w = full_ts[right] - np.int64(180_000_000_000)
        while left < right and full_ts[left] < w:
            left += 1
        tick_rates[right] = (right - left) / 3.0

    # 30-min range: use searchsorted for window start, track running max/min.
    range_30m_pct = np.zeros(n, dtype=np.float64)
    for i in range(start, n):
        ws = int(np.searchsorted(full_ts, full_ts[i] - np.int64(1_800_000_000_000), side="left"))
        if ws < i:
            wp = full_prices[ws : i + 1]
            r = float(np.max(wp) - np.min(wp))
            p = float(full_prices[i])
            range_30m_pct[i] = r / p * 100 if p > 0 else 0

    # Approach speed + direction + tick density (10-second window).
    approach_speed = np.zeros(n, dtype=np.float64)
    approach_dir = np.zeros(n, dtype=np.float64)
    tick_density_10s = np.zeros(n, dtype=np.float64)
    left_10 = start
    for i in range(start, n):
        w10 = full_ts[i] - np.int64(10_000_000_000)
        while left_10 < i and full_ts[left_10] < w10:
            left_10 += 1
        if left_10 < i:
            p_ago = float(full_prices[left_10])
            p_now = float(full_prices[i])
            elapsed = (full_ts[i] - full_ts[left_10]) / 1e9
            approach_speed[i] = abs(p_now - p_ago) / max(elapsed, 0.1)
            approach_dir[i] = p_now - p_ago
            tick_density_10s[i] = (i - left_10) / 10.0

    # ET minutes: convert nanosecond timestamps to ET minutes since midnight.
    # ET offset: -4h (EDT) or -5h (EST). Use the date to determine.
    import pytz as _pz
    dt_local = _pz.timezone("America/New_York").localize(
        datetime.datetime.combine(dc.date, datetime.time(12, 0))
    )
    utc_offset_ns = np.int64(dt_local.utcoffset().total_seconds() * 1e9)
    et_minutes = np.zeros(n, dtype=np.int32)
    for i in range(start, n):
        et_ns = full_ts[i] + utc_offset_ns
        total_mins = int(et_ns // 60_000_000_000) % 1440
        et_minutes[i] = total_mins

    # Session move %.
    session_move_pct = np.zeros(n, dtype=np.float64)
    if first_price > 0:
        for i in range(start, n):
            session_move_pct[i] = (float(full_prices[i]) - first_price) / first_price * 100

    return DayFactors(
        tick_rates=tick_rates, range_30m_pct=range_30m_pct,
        approach_speed=approach_speed, approach_dir=approach_dir,
        tick_density_10s=tick_density_10s, et_minutes=et_minutes,
        session_move_pct=session_move_pct, first_price=first_price,
    )


def precompute_day(
    dc: DayCache,
    factors: DayFactors,
    entry_threshold: float,
    exit_threshold: float,
    include_vwap: bool = True,
) -> tuple[list[Entry], list[str], list[int], list[float]]:
    """Compute entries + outcomes using precomputed factors. Fast."""
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    full_ts = dc.full_ts_ns
    full_prices = dc.full_prices
    eod_ns = _eod_cutoff_ns(dc.date)

    levels_config = [
        ("IBH", np.full(n, dc.ibh)),
        ("IBL", np.full(n, dc.ibl)),
        ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo)),
        ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi)),
    ]
    if include_vwap:
        levels_config.append(("VWAP", dc.post_ib_vwaps))

    last_exit_ns: dict[str, int] = {}
    raw: list[tuple[int, str, int, float, int]] = []

    for level_name, level_arr in levels_config:
        use_current = level_name == "VWAP"
        in_zone = False
        ref_price = 0.0
        entry_count = 0
        for j in range(n):
            pj = prices[j]
            lj = level_arr[j]
            if in_zone:
                er = lj if use_current else ref_price
                if abs(pj - er) > exit_threshold:
                    in_zone = False
                    last_exit_ns[level_name] = int(full_ts[start + j])
            else:
                if abs(pj - lj) <= entry_threshold:
                    in_zone = True
                    ref_price = lj
                    entry_count += 1
                    raw.append((start + j, level_name, entry_count, ref_price,
                                last_exit_ns.get(level_name, 0)))

    raw.sort(key=lambda x: x[0])

    entries = []
    outcomes = []
    exit_ns_list = []
    pnl_list = []

    for gidx, level_name, ec, ref_price, last_exit in raw:
        ep = float(full_prices[gidx])
        ens = int(full_ts[gidx])
        direction = "up" if ep > ref_price else "down"

        secs_since = (ens - last_exit) / 1e9 if last_exit > 0 and ens > last_exit else 99999.0

        entries.append(Entry(
            global_idx=gidx, level=level_name, direction=direction,
            entry_count=ec, entry_price=ep, line_price=ref_price,
            entry_ns=ens,
            now_et_mins=int(factors.et_minutes[gidx]),
            tick_rate=float(factors.tick_rates[gidx]),
            session_move_pct=float(factors.session_move_pct[gidx]),
            range_30m_pct=float(factors.range_30m_pct[gidx]),
            approach_speed=float(factors.approach_speed[gidx]),
            approach_direction=float(factors.approach_dir[gidx]),
            tick_density_10s=float(factors.tick_density_10s[gidx]),
            secs_since_last_exit=secs_since,
        ))

        out, eidx, pnl = evaluate_bot_trade(
            gidx, ref_price, direction,
            full_ts, full_prices, 8.0, 20.0, 900, eod_ns,
        )
        outcomes.append(out)
        exit_ns_list.append(int(full_ts[eidx]))
        pnl_list.append(pnl * MULTIPLIER)

    return entries, outcomes, exit_ns_list, pnl_list


# ══════════════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BotWeights:
    """All scoring weights including line-specific factors."""
    # Level quality
    level_ibh: int = 0
    level_ibl: int = 0
    level_fib_hi: int = 0
    level_fib_lo: int = 0
    level_vwap: int = 0
    # Direction combos
    combo_ibl_down: int = 0
    combo_fib_lo_down: int = 0
    combo_fib_hi_up: int = 0
    combo_ibh_up: int = 0
    combo_fib_hi_down: int = 0
    combo_vwap_down: int = 0
    # Time of day
    time_post_ib: int = 0  # 10:31-11:30
    time_power_hour: int = 0  # 15:00+
    # Entry count
    test_2: int = 0
    test_3: int = 0
    # Tick rate
    tick_low: int = 0  # <500
    tick_very_low: int = 0  # <1000 (renamed from tick_500_1000)
    tick_high: int = 0  # 2500+
    # Session move
    move_mild_red: int = 0
    move_strong_green: int = 0
    # Volatility
    vol_dead: int = 0  # 30m range < 0.15%
    vol_high: int = 0  # > 0.50%
    # NEW: approach speed
    approach_slow: int = 0  # < 0.3 pts/sec
    approach_fast: int = 0  # > 1.5 pts/sec
    approach_very_fast: int = 0  # > 3.0 pts/sec
    # NEW: tick density
    density_low: int = 0  # < 5 ticks/sec
    density_high: int = 0  # > 20 ticks/sec
    # NEW: time since last test
    fresh_test: int = 0  # > 600s (10 min)
    rapid_retest: int = 0  # < 120s (2 min)


def score_entry(e: Entry, w: BotWeights) -> int:
    s = 0
    # Level
    if e.level == "IBH": s += w.level_ibh
    elif e.level == "IBL": s += w.level_ibl
    elif e.level == "FIB_EXT_HI_1.272": s += w.level_fib_hi
    elif e.level == "FIB_EXT_LO_1.272": s += w.level_fib_lo
    elif e.level == "VWAP": s += w.level_vwap
    # Combos
    c = (e.level, e.direction)
    if c == ("IBL", "down"): s += w.combo_ibl_down
    elif c == ("FIB_EXT_LO_1.272", "down"): s += w.combo_fib_lo_down
    elif c == ("FIB_EXT_HI_1.272", "up"): s += w.combo_fib_hi_up
    elif c == ("IBH", "up"): s += w.combo_ibh_up
    elif c == ("FIB_EXT_HI_1.272", "down"): s += w.combo_fib_hi_down
    elif c == ("VWAP", "down"): s += w.combo_vwap_down
    # Time
    if 631 <= e.now_et_mins < 690: s += w.time_post_ib
    elif e.now_et_mins >= 900: s += w.time_power_hour
    # Entry count
    if e.entry_count == 2: s += w.test_2
    elif e.entry_count == 3: s += w.test_3
    # Tick rate
    if e.tick_rate < 500: s += w.tick_low
    elif e.tick_rate < 1000: s += w.tick_very_low
    elif e.tick_rate >= 2500: s += w.tick_high
    # Session move
    if -0.09 < e.session_move_pct <= -0.04: s += w.move_mild_red
    elif e.session_move_pct > 0.20: s += w.move_strong_green
    # Volatility
    if e.range_30m_pct < 0.15: s += w.vol_dead
    elif e.range_30m_pct > 0.50: s += w.vol_high
    # Approach speed
    if e.approach_speed < 0.3: s += w.approach_slow
    elif e.approach_speed > 3.0: s += w.approach_very_fast
    elif e.approach_speed > 1.5: s += w.approach_fast
    # Tick density
    if e.tick_density_10s < 5: s += w.density_low
    elif e.tick_density_10s > 20: s += w.density_high
    # Time since last test
    if e.secs_since_last_exit > 600: s += w.fresh_test
    elif e.secs_since_last_exit < 120: s += w.rapid_retest
    return s


def fit_weights(entries_outcomes: list[tuple[Entry, str]]) -> BotWeights:
    """Derive all weights from data."""
    if not entries_outcomes:
        return BotWeights()
    total = len(entries_outcomes)
    wc = sum(1 for _, o in entries_outcomes if o == "win")
    bl = wc / total * 100  # baseline WR

    def wr(fn):
        sub = [(e, o) for e, o in entries_outcomes if fn(e)]
        if len(sub) < 30: return bl
        return sum(1 for _, o in sub if o == "win") / len(sub) * 100

    sw = suggest_weight
    w = BotWeights()
    w.level_ibh = sw(wr(lambda e: e.level == "IBH"), bl)
    w.level_ibl = sw(wr(lambda e: e.level == "IBL"), bl)
    w.level_fib_hi = sw(wr(lambda e: e.level == "FIB_EXT_HI_1.272"), bl)
    w.level_fib_lo = sw(wr(lambda e: e.level == "FIB_EXT_LO_1.272"), bl)
    w.level_vwap = sw(wr(lambda e: e.level == "VWAP"), bl)
    w.combo_ibl_down = sw(wr(lambda e: e.level == "IBL" and e.direction == "down"), bl)
    w.combo_fib_lo_down = sw(wr(lambda e: e.level == "FIB_EXT_LO_1.272" and e.direction == "down"), bl)
    w.combo_fib_hi_up = sw(wr(lambda e: e.level == "FIB_EXT_HI_1.272" and e.direction == "up"), bl)
    w.combo_ibh_up = sw(wr(lambda e: e.level == "IBH" and e.direction == "up"), bl)
    w.combo_fib_hi_down = sw(wr(lambda e: e.level == "FIB_EXT_HI_1.272" and e.direction == "down"), bl)
    w.combo_vwap_down = sw(wr(lambda e: e.level == "VWAP" and e.direction == "down"), bl)
    w.time_post_ib = sw(wr(lambda e: 631 <= e.now_et_mins < 690), bl)
    w.time_power_hour = sw(wr(lambda e: e.now_et_mins >= 900), bl)
    w.test_2 = sw(wr(lambda e: e.entry_count == 2), bl)
    w.test_3 = sw(wr(lambda e: e.entry_count == 3), bl)
    w.tick_low = sw(wr(lambda e: e.tick_rate < 500), bl)
    w.tick_very_low = sw(wr(lambda e: 500 <= e.tick_rate < 1000), bl)
    w.tick_high = sw(wr(lambda e: e.tick_rate >= 2500), bl)
    w.move_mild_red = sw(wr(lambda e: -0.09 < e.session_move_pct <= -0.04), bl)
    w.move_strong_green = sw(wr(lambda e: e.session_move_pct > 0.20), bl)
    w.vol_dead = sw(wr(lambda e: e.range_30m_pct < 0.15), bl)
    w.vol_high = sw(wr(lambda e: e.range_30m_pct > 0.50), bl)
    # NEW factors
    w.approach_slow = sw(wr(lambda e: e.approach_speed < 0.3), bl)
    w.approach_fast = sw(wr(lambda e: 1.5 < e.approach_speed <= 3.0), bl)
    w.approach_very_fast = sw(wr(lambda e: e.approach_speed > 3.0), bl)
    w.density_low = sw(wr(lambda e: e.tick_density_10s < 5), bl)
    w.density_high = sw(wr(lambda e: e.tick_density_10s > 20), bl)
    w.fresh_test = sw(wr(lambda e: e.secs_since_last_exit > 600), bl)
    w.rapid_retest = sw(wr(lambda e: e.secs_since_last_exit < 120), bl)
    return w


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    date: datetime.date
    pnl_usd: float
    outcome: str


def replay_1pos(
    days, data_by_date, weights, min_score,
    max_per_level, daily_loss, max_consec,
):
    trades = []
    for date in days:
        dat = data_by_date.get(date)
        if not dat:
            continue
        entries, outcomes, exit_ns, pnl_usd = dat
        eod = _eod_cutoff_ns(date)
        pos_exit = 0
        dpnl = 0.0
        dcons = 0
        stopped = False
        lc = {}
        for i, e in enumerate(entries):
            if stopped: break
            if e.entry_ns >= eod: break
            if e.entry_ns < pos_exit: continue
            lv = lc.get(e.level, 0)
            if lv >= max_per_level: continue
            if e.range_30m_pct < 0.15: continue  # vol filter
            sc = score_entry(e, weights)
            if sc < min_score: continue
            pos_exit = exit_ns[i]
            lc[e.level] = lv + 1
            trades.append(Trade(date, pnl_usd[i], outcomes[i]))
            dpnl += pnl_usd[i]
            if pnl_usd[i] < 0:
                dcons += 1
            else:
                dcons = 0
            if daily_loss and dpnl <= -daily_loss: stopped = True
            if max_consec and dcons >= max_consec: stopped = True
    return trades


def fmt(trades, nd, label=""):
    if not trades:
        return f"  {label:>55s}  no trades"
    w = sum(1 for t in trades if t.outcome == "win")
    l = sum(1 for t in trades if t.outcome == "loss")
    o = len(trades) - w - l
    d = w + l
    wr = w / d * 100 if d else 0
    pnl = sum(t.pnl_usd for t in trades)
    ppd = pnl / nd
    eq = STARTING_BALANCE; peak = eq; dd = 0.0
    for t in trades:
        eq += t.pnl_usd; peak = max(peak, eq); dd = max(dd, peak - eq)
    return (
        f"  {label:>55s}  {len(trades):>4} ({len(trades)/nd:.1f}/d) "
        f"{w}W/{l}L/{o}O {wr:>5.1f}%  "
        f"PnL {pnl:>+8,.0f} ({ppd:>+5.1f}/d)  DD {dd:>5,.0f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    print("=" * 100)
    print("  LINE-EDGE BACKTEST — T8/S20, line-specific factors, walk-forward")
    print("=" * 100)

    days = load_cached_days()
    day_caches = {}
    for date in days:
        try:
            df = load_day(date)
            dc = preprocess_day(df, date)
            if dc is not None:
                day_caches[date] = dc
        except: pass
    valid_days = sorted(day_caches.keys())
    N = len(valid_days)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    # Precompute scoring factors ONCE per day (shared across exit thresholds).
    print(f"\n  Precomputing per-day scoring factors...", flush=True)
    t1 = time.time()
    factors_by_date: dict[datetime.date, DayFactors] = {}
    for i, date in enumerate(valid_days):
        factors_by_date[date] = precompute_factors(day_caches[date])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{N}...", flush=True)
    print(f"  Factors done in {time.time()-t1:.0f}s")

    # Precompute entries + outcomes for each exit threshold.
    EXIT_GRID = [10, 12, 15]
    all_data: dict[int, dict] = {}

    for ex in EXIT_GRID:
        print(f"\n  Computing entries for exit={ex}...", flush=True)
        t2 = time.time()
        data = {}
        for date in valid_days:
            data[date] = precompute_day(
                day_caches[date], factors_by_date[date], 1.0, float(ex)
            )
        all_data[ex] = data
        n_entries = sum(len(d[0]) for d in data.values())
        print(f"  exit={ex}: {n_entries} entries in {time.time()-t2:.0f}s")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: Factor analysis (exit=12, full dataset)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 1: Factor analysis — line-specific factors (exit=12, T8/S20)")
    print("=" * 100)

    ref_data = all_data[12]
    all_eo = []
    for date in valid_days:
        entries, outcomes, _, _ = ref_data[date]
        for i, e in enumerate(entries):
            all_eo.append((e, outcomes[i]))

    total = len(all_eo)
    wc = sum(1 for _, o in all_eo if o == "win")
    bl = wc / total * 100
    print(f"\n  Baseline: {wc}W / {total} = {bl:.1f}% WR\n")

    def wr_line(label, fn):
        sub = [(e, o) for e, o in all_eo if fn(e)]
        if len(sub) < 30: return
        w = sum(1 for _, o in sub if o == "win")
        wr = w / len(sub) * 100
        delta = wr - bl
        wt = suggest_weight(wr, bl)
        print(f"  {label:<50s} {w:>5}W/{len(sub):>5} = {wr:>5.1f}% ({delta:>+5.1f}pp) wt={wt:>+d}")

    print("  --- NEW: Approach speed (pts/sec in last 10s) ---")
    for lo, hi, label in [
        (0, 0.3, "Very slow (<0.3)"),
        (0.3, 0.7, "Slow (0.3-0.7)"),
        (0.7, 1.5, "Medium (0.7-1.5)"),
        (1.5, 3.0, "Fast (1.5-3.0)"),
        (3.0, 999, "Very fast (>3.0)"),
    ]:
        wr_line(f"Approach speed {label}", lambda e, l=lo, h=hi: l <= e.approach_speed < h)

    print("\n  --- NEW: Tick density (ticks/sec in last 10s) ---")
    for lo, hi, label in [
        (0, 3, "<3"), (3, 7, "3-7"), (7, 15, "7-15"),
        (15, 25, "15-25"), (25, 999, ">25"),
    ]:
        wr_line(f"Tick density {label}", lambda e, l=lo, h=hi: l <= e.tick_density_10s < h)

    print("\n  --- NEW: Time since last test (seconds) ---")
    for lo, hi, label in [
        (0, 60, "<1 min"), (60, 120, "1-2 min"), (120, 300, "2-5 min"),
        (300, 600, "5-10 min"), (600, 1800, "10-30 min"),
        (1800, 99999, ">30 min / first test"),
    ]:
        wr_line(f"Since last test {label}", lambda e, l=lo, h=hi: l <= e.secs_since_last_exit < h)

    print("\n  --- Existing factors (for comparison) ---")
    for lv in ["IBH", "IBL", "VWAP", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]:
        wr_line(lv, lambda e, l=lv: e.level == l)
    wr_line("Post-IB (10:31-11:30)", lambda e: 631 <= e.now_et_mins < 690)
    wr_line("Power hour (15:00+)", lambda e: e.now_et_mins >= 900)
    wr_line("Tick rate <500", lambda e: e.tick_rate < 500)
    wr_line("30m range <0.15%", lambda e: e.range_30m_pct < 0.15)

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 2: Walk-forward validation")
    print("=" * 100)

    SCORE_GRID = [-2, -1, 0, 1, 2]
    MAX_E_GRID = [3, 5, 8]

    oos_by_config: dict[str, list[Trade]] = {}
    oos_days = 0
    k = INITIAL_TRAIN_DAYS
    windows = 0

    while k < N:
        train_days = valid_days[:k]
        test_days = valid_days[k : k + STEP_DAYS]
        if not test_days: break
        windows += 1
        oos_days += len(test_days)

        for ex in EXIT_GRID:
            data = all_data[ex]
            # Train weights.
            train_eo = []
            for d in train_days:
                entries, outcomes, _, _ = data[d]
                for i, e in enumerate(entries):
                    train_eo.append((e, outcomes[i]))
            wt = fit_weights(train_eo)

            for min_s in SCORE_GRID:
                for max_e in MAX_E_GRID:
                    label = f"exit={ex} score>={min_s} max={max_e}"
                    trades = replay_1pos(
                        test_days, data, wt, min_s, max_e, 150.0, 3,
                    )
                    oos_by_config.setdefault(label, []).extend(trades)

        k += STEP_DAYS

    print(f"\n  {windows} windows, {oos_days} OOS days\n")

    oos_results = []
    for cfg, trades in oos_by_config.items():
        pnl = sum(t.pnl_usd for t in trades)
        ppd = pnl / oos_days if oos_days else 0
        oos_results.append((cfg, trades, ppd))
    oos_results.sort(key=lambda x: x[2], reverse=True)

    print(f"  Top 20 by $/day (OOS):\n")
    for cfg, trades, ppd in oos_results[:20]:
        print(fmt(trades, oos_days, cfg))

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  STAGE 3: Recent 60 days")
    print("=" * 100)

    recent = valid_days[-60:]
    rn = len(recent)
    pre = [d for d in valid_days if d < recent[0]]
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")

    for cfg, _, _ in oos_results[:10]:
        parts = cfg.split()
        ex = int(parts[0].split("=")[1])
        min_s = int(parts[1].split(">=")[1])
        max_e = int(parts[2].split("=")[1])

        data = all_data[ex]
        pre_eo = []
        for d in pre:
            entries, outcomes, _, _ = data[d]
            for i, e in enumerate(entries):
                pre_eo.append((e, outcomes[i]))
        wr = fit_weights(pre_eo)

        trades = replay_1pos(recent, data, wr, min_s, max_e, 150.0, 3)
        print(fmt(trades, rn, f"RECENT: {cfg}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
