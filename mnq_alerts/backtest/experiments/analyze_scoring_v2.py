"""Deep scoring analysis v2: new factors, wider entry, and factor combinations.

Previous scoring attempts (all failed):
- Bot weights, human weights, walk-forward weights: WR flat at 1pt entry
- Session move, tick rate, entry count, time of day: no signal
- Streak filter: removes profitable trades (84.5% WR after 2+L)

NEW approaches in this analysis:
1. Price vs VWAP at entry (distance and direction) — bot excludes VWAP
   as a level but could use it as a signal
2. IB range utilization — how far into/beyond IB range has price moved
3. Short-term momentum — price change in last 1/5/10 min before entry
4. Cumulative P&L at time of entry — does bot perform differently when
   up vs down on the day
5. Factor COMBINATIONS — pairs of factors that predict together
6. Level-specific factor analysis — scoring might work for some levels
7. WIDER ENTRY THRESHOLD (2pt, 3pt, 4pt) — user observed price bounces
   2-4pts from level without triggering bot. Target/stop anchored to
   ENTRY PRICE (not line price) for realistic P&L.

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/analyze_scoring_v2.py
"""
import os, sys, time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.scoring import EntryFactors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bot_risk_backtest import evaluate_bot_trade, MULTIPLIER, FEE_PTS
from walk_forward import _eod_cutoff_ns

# Global references for multiprocessing (COW via fork)
_DATES = None
_CACHES = None
_ARRAYS = None

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}
BASE_EXCLUDE = {"FIB_0.5", "IBL"}
IB_SET = 630

ALL_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
    "IBH": (6, 20), "IBL": (6, 20), "VWAP": (8, 25), "FIB_0.5": (10, 25),
}
ALL_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
    "IBH": 7, "IBL": 7, "VWAP": 12, "FIB_0.5": 5,
}


def _run_wide_job(args):
    """Worker for threshold or near-miss analysis (runs in forked process)."""
    job_type, param = args
    t0 = time.time()

    if job_type == "threshold":
        return _run_threshold(param, t0)
    else:
        return _run_nearmiss(param, t0)


def _run_threshold(threshold, t0):
    """Simulate wider entry threshold across all days."""
    total_trades = 0
    total_wins = 0
    total_pnl_usd = 0.0
    level_stats = defaultdict(lambda: [0, 0, 0.0])
    level_dir_stats = defaultdict(lambda: [0, 0, 0.0])

    for date in _DATES:
        dc = _CACHES[date]
        arr = _ARRAYS[date]
        fp = dc.full_prices
        ft = dc.full_ts_ns
        n = len(dc.post_ib_prices)
        start = dc.post_ib_start_idx
        eod = _eod_cutoff_ns(dc.date)

        ib_range = dc.ibh - dc.ibl
        levels = {
            "IBH": dc.ibh, "IBL": dc.ibl,
            "FIB_EXT_HI_1.272": dc.fib_hi, "FIB_EXT_LO_1.272": dc.fib_lo,
            "FIB_0.236": dc.ibl + 0.236 * ib_range,
            "FIB_0.5": dc.ibl + 0.5 * ib_range,
            "FIB_0.618": dc.ibl + 0.618 * ib_range,
            "FIB_0.764": dc.ibl + 0.764 * ib_range,
        }
        has_vwap = hasattr(dc, 'post_ib_vwaps') and dc.post_ib_vwaps is not None
        if has_vwap:
            levels["VWAP"] = float(dc.post_ib_vwaps[0])

        caps = dict(ALL_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        in_zone = {name: False for name in levels}
        entry_counts = {name: 0 for name in levels}
        pos_exit_idx = -1
        day_pnl = 0.0

        for j in range(n):
            gi = start + j
            if int(ft[gi]) >= eod:
                break
            if gi <= pos_exit_idx:
                continue
            if day_pnl <= -100:
                break

            pj = float(dc.post_ib_prices[j])
            et_mins = int(arr.et_mins[gi])

            if has_vwap and j < len(dc.post_ib_vwaps):
                levels["VWAP"] = float(dc.post_ib_vwaps[j])

            if 810 <= et_mins < 840:
                continue

            for name, line_price in levels.items():
                dist = abs(pj - line_price)

                if in_zone[name]:
                    if dist > threshold:
                        in_zone[name] = False
                    continue

                if dist <= threshold:
                    in_zone[name] = True
                    d = "up" if pj > line_price else "down"

                    if entry_counts[name] >= caps.get(name, 12):
                        continue

                    entry_counts[name] += 1
                    target_pts, stop_pts = ALL_TS.get(name, (8, 25))
                    entry_price = pj

                    if d == "up":
                        target_price = entry_price + target_pts
                        stop_price = entry_price - stop_pts
                    else:
                        target_price = entry_price - target_pts
                        stop_price = entry_price + stop_pts

                    entry_ns = int(ft[gi])
                    window_ns = 900 * 1_000_000_000
                    eval_end_ns = entry_ns + window_ns
                    if eod < eval_end_ns:
                        eval_end_ns = eod

                    target_idx = -1
                    stop_idx = -1
                    last_idx = gi

                    for i in range(gi + 1, len(fp)):
                        if int(ft[i]) > eval_end_ns:
                            break
                        last_idx = i
                        pi = float(fp[i])
                        if d == "up":
                            if target_idx < 0 and pi >= target_price:
                                target_idx = i
                            if stop_idx < 0 and pi <= stop_price:
                                stop_idx = i
                        else:
                            if target_idx < 0 and pi <= target_price:
                                target_idx = i
                            if stop_idx < 0 and pi >= stop_price:
                                stop_idx = i
                        if target_idx >= 0 and stop_idx >= 0:
                            break

                    if target_idx >= 0 and stop_idx >= 0:
                        if target_idx <= stop_idx:
                            pnl_pts = target_pts - FEE_PTS
                            exit_idx = target_idx
                        else:
                            pnl_pts = -(stop_pts + FEE_PTS)
                            exit_idx = stop_idx
                    elif target_idx >= 0:
                        pnl_pts = target_pts - FEE_PTS
                        exit_idx = target_idx
                    elif stop_idx >= 0:
                        pnl_pts = -(stop_pts + FEE_PTS)
                        exit_idx = stop_idx
                    else:
                        timeout_price = float(fp[last_idx])
                        if d == "up":
                            pnl_pts = timeout_price - entry_price - FEE_PTS
                        else:
                            pnl_pts = entry_price - timeout_price - FEE_PTS
                        exit_idx = last_idx

                    pnl_usd = pnl_pts * MULTIPLIER
                    pos_exit_idx = exit_idx
                    day_pnl += pnl_usd

                    total_trades += 1
                    if pnl_usd >= 0:
                        total_wins += 1
                    total_pnl_usd += pnl_usd
                    level_stats[name][0] += (1 if pnl_usd >= 0 else 0)
                    level_stats[name][1] += (1 if pnl_usd < 0 else 0)
                    level_stats[name][2] += pnl_usd
                    ld_key = f"{name} {d.upper()}"
                    level_dir_stats[ld_key][0] += (1 if pnl_usd >= 0 else 0)
                    level_dir_stats[ld_key][1] += (1 if pnl_usd < 0 else 0)
                    level_dir_stats[ld_key][2] += pnl_usd
                    break  # 1 position at a time

    return {
        "type": "threshold", "threshold": threshold,
        "trades": total_trades, "wins": total_wins, "pnl": total_pnl_usd,
        "level_stats": dict(level_stats), "level_dir_stats": dict(level_dir_stats),
        "elapsed": time.time() - t0,
    }


def _run_nearmiss(range_tuple, t0):
    """Analyze near-misses at a given distance range."""
    miss_range_lo, miss_range_hi = range_tuple
    near_miss_count = 0
    would_have_won = 0
    would_have_lost = 0
    would_have_pnl = 0.0
    nm_level_stats = defaultdict(lambda: [0, 0, 0.0])

    for date in _DATES:
        dc = _CACHES[date]
        fp = dc.full_prices
        ft = dc.full_ts_ns
        n = len(dc.post_ib_prices)
        start = dc.post_ib_start_idx
        eod = _eod_cutoff_ns(dc.date)

        ib_range = dc.ibh - dc.ibl
        levels = {
            "IBH": dc.ibh, "IBL": dc.ibl,
            "FIB_EXT_HI_1.272": dc.fib_hi, "FIB_EXT_LO_1.272": dc.fib_lo,
            "FIB_0.236": dc.ibl + 0.236 * ib_range,
            "FIB_0.5": dc.ibl + 0.5 * ib_range,
            "FIB_0.618": dc.ibl + 0.618 * ib_range,
            "FIB_0.764": dc.ibl + 0.764 * ib_range,
        }

        in_near = {name: False for name in levels}
        touched_1pt = {name: False for name in levels}

        for j in range(n):
            gi = start + j
            if int(ft[gi]) >= eod:
                break
            pj = float(dc.post_ib_prices[j])

            for name, line_price in levels.items():
                dist = abs(pj - line_price)

                if dist <= 1.0:
                    touched_1pt[name] = True
                    in_near[name] = False
                    continue

                if miss_range_lo < dist <= miss_range_hi:
                    if not in_near[name] and not touched_1pt[name]:
                        in_near[name] = True
                        d = "up" if pj > line_price else "down"

                        target_pts, stop_pts = ALL_TS.get(name, (8, 25))
                        entry_price = pj
                        if d == "up":
                            target_price = entry_price + target_pts
                            stop_price = entry_price - stop_pts
                        else:
                            target_price = entry_price - target_pts
                            stop_price = entry_price + stop_pts

                        entry_ns = int(ft[gi])
                        window_ns = 900 * 1_000_000_000
                        eval_end_ns = min(entry_ns + window_ns, eod)

                        t_idx = -1
                        s_idx = -1
                        for i in range(gi + 1, len(fp)):
                            if int(ft[i]) > eval_end_ns:
                                break
                            pi = float(fp[i])
                            if d == "up":
                                if t_idx < 0 and pi >= target_price:
                                    t_idx = i
                                if s_idx < 0 and pi <= stop_price:
                                    s_idx = i
                            else:
                                if t_idx < 0 and pi <= target_price:
                                    t_idx = i
                                if s_idx < 0 and pi >= stop_price:
                                    s_idx = i
                            if t_idx >= 0 and s_idx >= 0:
                                break

                        near_miss_count += 1
                        if t_idx >= 0 and (s_idx < 0 or t_idx <= s_idx):
                            would_have_won += 1
                            nm_pnl = (target_pts - FEE_PTS) * MULTIPLIER
                            would_have_pnl += nm_pnl
                            nm_level_stats[name][0] += 1
                            nm_level_stats[name][2] += nm_pnl
                        elif s_idx >= 0:
                            would_have_lost += 1
                            nm_pnl = -(stop_pts + FEE_PTS) * MULTIPLIER
                            would_have_pnl += nm_pnl
                            nm_level_stats[name][1] += 1
                            nm_level_stats[name][2] += nm_pnl

                elif dist > miss_range_hi + 5:
                    in_near[name] = False
                    touched_1pt[name] = False

    return {
        "type": "nearmiss", "range": range_tuple,
        "count": near_miss_count, "wins": would_have_won,
        "losses": would_have_lost, "pnl": would_have_pnl,
        "level_stats": dict(nm_level_stats),
        "elapsed": time.time() - t0,
    }


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    dates, caches = load_all_days()
    print(f"Loaded {len(dates)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    arrays = {d: precompute_arrays(caches[d]) for d in dates}

    # =====================================================================
    # Simulate baseline to get all trades with full context
    # =====================================================================
    print("Simulating all days...", flush=True)
    day_results = []
    streak = (0, 0)
    for date in dates:
        dc = caches[date]
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        trades, streak = simulate_day(
            dc, arrays[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=30,
            direction_filter={"IBH": "down"},
        )
        day_pnl = sum(t.pnl_usd for t in trades)
        day_results.append((date, trades, day_pnl))

    all_trades = [t for _, dt, _ in day_results for t in dt]
    bad_days = [(d, t, p) for d, t, p in day_results if p <= -100]
    good_days = [(d, t, p) for d, t, p in day_results if p >= 50]

    n_days = len(dates)
    total_pnl = sum(p for _, _, p in day_results)
    print(f"\nBaseline: {n_days} days, {len(all_trades)} trades, "
          f"${total_pnl/n_days:+.2f}/day")
    print(f"Bad days: {len(bad_days)}, Good days: {len(good_days)}")

    # =====================================================================
    # 1. PRICE vs VWAP AT ENTRY
    # =====================================================================
    print("\n" + "=" * 80)
    print("1. PRICE vs VWAP AT ENTRY")
    print("    Bot excludes VWAP as a level, but VWAP position could be a signal")
    print("=" * 80)

    vwap_buckets = defaultdict(lambda: [0, 0, 0.0])
    for date, dt, _ in day_results:
        dc = caches[date]
        if not hasattr(dc, 'post_ib_vwaps') or dc.post_ib_vwaps is None:
            continue
        start = dc.post_ib_start_idx
        for t in dt:
            j = t.entry_idx - start
            if j < 0 or j >= len(dc.post_ib_vwaps):
                continue
            vwap = float(dc.post_ib_vwaps[j])
            entry_price = float(dc.full_prices[t.entry_idx])
            dist = entry_price - vwap
            # Bucket by distance from VWAP
            if dist > 100:
                bucket = ">100 above"
            elif dist > 50:
                bucket = "50-100 above"
            elif dist > 20:
                bucket = "20-50 above"
            elif dist > 0:
                bucket = "0-20 above"
            elif dist > -20:
                bucket = "0-20 below"
            elif dist > -50:
                bucket = "20-50 below"
            elif dist > -100:
                bucket = "50-100 below"
            else:
                bucket = ">100 below"
            w = 1 if t.pnl_usd >= 0 else 0
            vwap_buckets[bucket][0] += w
            vwap_buckets[bucket][1] += (1 - w)
            vwap_buckets[bucket][2] += t.pnl_usd

    print(f"\n  a) WR by distance from VWAP:")
    print(f"    {'Bucket':<16} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8}")
    for b in [">100 above", "50-100 above", "20-50 above", "0-20 above",
              "0-20 below", "20-50 below", "50-100 below", ">100 below"]:
        if b in vwap_buckets:
            w, l, p = vwap_buckets[b]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            avg = p / total if total > 0 else 0
            print(f"    {b:<16} {total:>7} {wr:>5.1f}% {p:>+9.0f} {avg:>+8.2f}")

    # VWAP direction match: does trading WITH VWAP direction help?
    print(f"\n  b) Direction vs VWAP position:")
    vwap_dir = defaultdict(lambda: [0, 0, 0.0])
    for date, dt, _ in day_results:
        dc = caches[date]
        if not hasattr(dc, 'post_ib_vwaps') or dc.post_ib_vwaps is None:
            continue
        start = dc.post_ib_start_idx
        for t in dt:
            j = t.entry_idx - start
            if j < 0 or j >= len(dc.post_ib_vwaps):
                continue
            vwap = float(dc.post_ib_vwaps[j])
            entry_price = float(dc.full_prices[t.entry_idx])
            above_vwap = entry_price > vwap
            # "With VWAP" = BUY when above VWAP, SELL when below
            if (t.direction == "up" and above_vwap) or (t.direction == "down" and not above_vwap):
                label = "With VWAP"
            else:
                label = "Against VWAP"
            w = 1 if t.pnl_usd >= 0 else 0
            vwap_dir[label][0] += w
            vwap_dir[label][1] += (1 - w)
            vwap_dir[label][2] += t.pnl_usd
    for label in ["With VWAP", "Against VWAP"]:
        if label in vwap_dir:
            w, l, p = vwap_dir[label]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            print(f"    {label:<16}: {total} trades, {wr:.1f}% WR, ${p:+.0f}")

    # =====================================================================
    # 2. IB RANGE UTILIZATION
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. IB RANGE UTILIZATION AT ENTRY")
    print("    How far into/beyond the IB range has price moved?")
    print("=" * 80)

    ib_util_buckets = defaultdict(lambda: [0, 0, 0.0])
    for date, dt, _ in day_results:
        dc = caches[date]
        ib_range = dc.ibh - dc.ibl
        if ib_range <= 0:
            continue
        ib_mid = (dc.ibh + dc.ibl) / 2
        for t in dt:
            entry_price = float(dc.full_prices[t.entry_idx])
            # Position within IB range: 0 = at IBL, 1 = at IBH
            ib_pct = (entry_price - dc.ibl) / ib_range
            if ib_pct > 1.5:
                bucket = ">150% (far above IBH)"
            elif ib_pct > 1.0:
                bucket = "100-150% (above IBH)"
            elif ib_pct > 0.75:
                bucket = "75-100% (upper IB)"
            elif ib_pct > 0.5:
                bucket = "50-75% (mid-upper IB)"
            elif ib_pct > 0.25:
                bucket = "25-50% (mid-lower IB)"
            elif ib_pct > 0:
                bucket = "0-25% (lower IB)"
            elif ib_pct > -0.5:
                bucket = "-50-0% (below IBL)"
            else:
                bucket = "<-50% (far below IBL)"
            w = 1 if t.pnl_usd >= 0 else 0
            ib_util_buckets[bucket][0] += w
            ib_util_buckets[bucket][1] += (1 - w)
            ib_util_buckets[bucket][2] += t.pnl_usd

    print(f"    {'IB position':<28} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8}")
    for b in [">150% (far above IBH)", "100-150% (above IBH)", "75-100% (upper IB)",
              "50-75% (mid-upper IB)", "25-50% (mid-lower IB)", "0-25% (lower IB)",
              "-50-0% (below IBL)", "<-50% (far below IBL)"]:
        if b in ib_util_buckets:
            w, l, p = ib_util_buckets[b]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            avg = p / total if total > 0 else 0
            print(f"    {b:<28} {total:>7} {wr:>5.1f}% {p:>+9.0f} {avg:>+8.2f}")

    # =====================================================================
    # 3. SHORT-TERM MOMENTUM BEFORE ENTRY
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. SHORT-TERM MOMENTUM (price change before entry)")
    print("    Does the speed/direction of approach predict outcome?")
    print("=" * 80)

    for lookback_label, lookback_ticks in [("1 min (~200 ticks)", 200),
                                            ("5 min (~1000 ticks)", 1000),
                                            ("10 min (~2000 ticks)", 2000)]:
        mom_buckets = defaultdict(lambda: [0, 0, 0.0])
        for date, dt, _ in day_results:
            dc = caches[date]
            fp = dc.full_prices
            for t in dt:
                idx = t.entry_idx
                start_idx = max(0, idx - lookback_ticks)
                if start_idx >= len(fp) or idx >= len(fp):
                    continue
                momentum = float(fp[idx]) - float(fp[start_idx])
                # Relative to direction: positive = price moving WITH trade direction
                if t.direction == "down":
                    momentum = -momentum
                if momentum > 20:
                    bucket = ">20 with"
                elif momentum > 10:
                    bucket = "10-20 with"
                elif momentum > 5:
                    bucket = "5-10 with"
                elif momentum > 0:
                    bucket = "0-5 with"
                elif momentum > -5:
                    bucket = "0-5 against"
                elif momentum > -10:
                    bucket = "5-10 against"
                elif momentum > -20:
                    bucket = "10-20 against"
                else:
                    bucket = ">20 against"
                w = 1 if t.pnl_usd >= 0 else 0
                mom_buckets[bucket][0] += w
                mom_buckets[bucket][1] += (1 - w)
                mom_buckets[bucket][2] += t.pnl_usd

        print(f"\n  {lookback_label}:")
        print(f"    {'Momentum':<16} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8}")
        for b in [">20 with", "10-20 with", "5-10 with", "0-5 with",
                  "0-5 against", "5-10 against", "10-20 against", ">20 against"]:
            if b in mom_buckets:
                w, l, p = mom_buckets[b]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                avg = p / total if total > 0 else 0
                print(f"    {b:<16} {total:>7} {wr:>5.1f}% {p:>+9.0f} {avg:>+8.2f}")

    # =====================================================================
    # 4. CUMULATIVE P&L AT TIME OF ENTRY
    # =====================================================================
    print("\n" + "=" * 80)
    print("4. CUMULATIVE P&L AT TIME OF ENTRY")
    print("    Does the bot trade differently when up vs down on the day?")
    print("=" * 80)

    pnl_buckets = defaultdict(lambda: [0, 0, 0.0])
    for date, dt, _ in day_results:
        cum_pnl = 0.0
        for t in dt:
            if cum_pnl <= -75:
                bucket = "<-$75"
            elif cum_pnl <= -50:
                bucket = "-$75 to -$50"
            elif cum_pnl <= -25:
                bucket = "-$50 to -$25"
            elif cum_pnl <= 0:
                bucket = "-$25 to $0"
            elif cum_pnl <= 25:
                bucket = "$0 to +$25"
            elif cum_pnl <= 50:
                bucket = "+$25 to +$50"
            elif cum_pnl <= 75:
                bucket = "+$50 to +$75"
            else:
                bucket = ">+$75"
            w = 1 if t.pnl_usd >= 0 else 0
            pnl_buckets[bucket][0] += w
            pnl_buckets[bucket][1] += (1 - w)
            pnl_buckets[bucket][2] += t.pnl_usd
            cum_pnl += t.pnl_usd

    print(f"    {'Day P&L at entry':<18} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/trade':>8}")
    for b in ["<-$75", "-$75 to -$50", "-$50 to -$25", "-$25 to $0",
              "$0 to +$25", "+$25 to +$50", "+$50 to +$75", ">+$75"]:
        if b in pnl_buckets:
            w, l, p = pnl_buckets[b]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            avg = p / total if total > 0 else 0
            print(f"    {b:<18} {total:>7} {wr:>5.1f}% {p:>+9.0f} {avg:>+8.2f}")

    # =====================================================================
    # 5. LEVEL-SPECIFIC FACTOR ANALYSIS
    # =====================================================================
    print("\n" + "=" * 80)
    print("5. LEVEL-SPECIFIC FACTOR ANALYSIS")
    print("    Do factors work differently for different levels?")
    print("=" * 80)

    # For each level, check momentum and VWAP position
    for level in ["FIB_0.236", "FIB_0.618", "FIB_0.764", "IBH",
                   "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]:
        level_trades = [t for t in all_trades if t.level == level]
        if len(level_trades) < 50:
            continue
        print(f"\n  {level} ({len(level_trades)} trades):")

        # Momentum (5 min) for this level
        mom_lv = defaultdict(lambda: [0, 0])
        for date, dt, _ in day_results:
            dc = caches[date]
            fp = dc.full_prices
            for t in dt:
                if t.level != level:
                    continue
                idx = t.entry_idx
                start_idx = max(0, idx - 1000)
                if start_idx >= len(fp) or idx >= len(fp):
                    continue
                momentum = float(fp[idx]) - float(fp[start_idx])
                if t.direction == "down":
                    momentum = -momentum
                bucket = "with" if momentum > 0 else "against"
                w = 1 if t.pnl_usd >= 0 else 0
                mom_lv[bucket][0] += w
                mom_lv[bucket][1] += (1 - w)
        for b in ["with", "against"]:
            if b in mom_lv:
                w, l = mom_lv[b]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"    5min momentum {b:<8}: {total} trades, {wr:.1f}% WR")

        # VWAP position for this level
        vwap_lv = defaultdict(lambda: [0, 0])
        for date, dt, _ in day_results:
            dc = caches[date]
            if not hasattr(dc, 'post_ib_vwaps') or dc.post_ib_vwaps is None:
                continue
            start = dc.post_ib_start_idx
            for t in dt:
                if t.level != level:
                    continue
                j = t.entry_idx - start
                if j < 0 or j >= len(dc.post_ib_vwaps):
                    continue
                vwap = float(dc.post_ib_vwaps[j])
                entry_price = float(dc.full_prices[t.entry_idx])
                bucket = "above VWAP" if entry_price > vwap else "below VWAP"
                w = 1 if t.pnl_usd >= 0 else 0
                vwap_lv[bucket][0] += w
                vwap_lv[bucket][1] += (1 - w)
        for b in ["above VWAP", "below VWAP"]:
            if b in vwap_lv:
                w, l = vwap_lv[b]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"    {b:<14}: {total} trades, {wr:.1f}% WR")

    # =====================================================================
    # 6. FACTOR COMBINATIONS
    # =====================================================================
    print("\n" + "=" * 80)
    print("6. FACTOR COMBINATIONS")
    print("    Do pairs of factors predict better than individual ones?")
    print("=" * 80)

    # Momentum (with/against) x VWAP position (above/below) x direction
    combo_stats = defaultdict(lambda: [0, 0, 0.0])
    for date, dt, _ in day_results:
        dc = caches[date]
        if not hasattr(dc, 'post_ib_vwaps') or dc.post_ib_vwaps is None:
            continue
        fp = dc.full_prices
        start = dc.post_ib_start_idx
        for t in dt:
            idx = t.entry_idx
            j = idx - start
            if j < 0 or j >= len(dc.post_ib_vwaps):
                continue
            # Momentum (5 min)
            start_idx = max(0, idx - 1000)
            momentum = float(fp[idx]) - float(fp[start_idx])
            if t.direction == "down":
                momentum = -momentum
            mom = "mom_with" if momentum > 0 else "mom_against"
            # VWAP
            vwap = float(dc.post_ib_vwaps[j])
            entry_price = float(fp[idx])
            vwap_pos = "above_vwap" if entry_price > vwap else "below_vwap"
            # Time
            et = t.factors.et_mins if t.factors else 0
            time_bucket = "first_hr" if et < IB_SET + 60 else "after_first_hr"

            combo = f"{mom} + {vwap_pos} + {time_bucket}"
            w = 1 if t.pnl_usd >= 0 else 0
            combo_stats[combo][0] += w
            combo_stats[combo][1] += (1 - w)
            combo_stats[combo][2] += t.pnl_usd

    print(f"    {'Combination':<50} {'Trades':>7} {'WR%':>6} {'P&L':>9}")
    for combo in sorted(combo_stats.keys()):
        w, l, p = combo_stats[combo]
        total = w + l
        if total < 30:
            continue
        wr = w / total * 100 if total > 0 else 0
        print(f"    {combo:<50} {total:>7} {wr:>5.1f}% {p:>+9.0f}")

    # =====================================================================
    # 7 & 8: WIDER ENTRY + NEAR-MISS (parallelized across 3 workers)
    # =====================================================================
    print("\n" + "=" * 80)
    print("7. WIDER ENTRY THRESHOLD (1pt, 2pt, 3pt, 4pt)")
    print("    Target/stop anchored to ENTRY PRICE, not line price.")
    print("    Clean slate: ALL levels, NO direction filters, NO exclusions.")
    print("8. NEAR-MISS ANALYSIS (1-2pt, 2-3pt, 3-4pt)")
    print("    Running 7 jobs across 3 workers...")
    print("=" * 80, flush=True)

    global _DATES, _CACHES, _ARRAYS
    _DATES = dates
    _CACHES = caches
    _ARRAYS = arrays

    jobs = [
        ("threshold", 1), ("threshold", 2), ("threshold", 3), ("threshold", 4),
        ("nearmiss", (1, 2)), ("nearmiss", (2, 3)), ("nearmiss", (3, 4)),
    ]
    with Pool(3) as pool:
        results = pool.map(_run_wide_job, jobs)

    # Print threshold results
    for r in results:
        if r["type"] == "threshold":
            threshold = r["threshold"]
            wr = r["wins"] / r["trades"] * 100 if r["trades"] > 0 else 0
            ppd = r["pnl"] / n_days
            print(f"\n  Threshold = {threshold}pt ({r['elapsed']:.0f}s):")
            print(f"    Total: {r['trades']} trades, {wr:.1f}% WR, "
                  f"${ppd:+.2f}/day, ${r['pnl']:+.0f} total")
            print(f"    {'Level':<20} {'Trades':>7} {'WR%':>6} {'P&L':>9} {'$/day':>7}")
            for lv in sorted(r["level_stats"].keys()):
                w, l, p = r["level_stats"][lv]
                total = w + l
                lvwr = w / total * 100 if total > 0 else 0
                print(f"    {lv:<20} {total:>7} {lvwr:>5.1f}% {p:>+9.0f} {p/n_days:>+7.2f}")
            print(f"\n    {'Level + Direction':<24} {'Trades':>7} {'WR%':>6} {'P&L':>9}")
            for ld in sorted(r["level_dir_stats"].keys()):
                w, l, p = r["level_dir_stats"][ld]
                total = w + l
                if total < 20:
                    continue
                ldwr = w / total * 100 if total > 0 else 0
                print(f"    {ld:<24} {total:>7} {ldwr:>5.1f}% {p:>+9.0f}")

    # Print near-miss results
    for r in results:
        if r["type"] == "nearmiss":
            lo, hi = r["range"]
            nm_total = r["wins"] + r["losses"]
            nm_wr = r["wins"] / nm_total * 100 if nm_total > 0 else 0
            print(f"\n  Near misses at {lo}-{hi}pt (never touched 1pt, {r['elapsed']:.0f}s):")
            print(f"    Count: {r['count']}, Resolved: {nm_total}")
            print(f"    Would have won: {r['wins']}, lost: {r['losses']}")
            print(f"    WR: {nm_wr:.1f}%, P&L: ${r['pnl']:+.0f}")
            if r["level_stats"]:
                print(f"    {'Level':<20} {'Wins':>6} {'Losses':>7} {'WR%':>6} {'P&L':>9}")
                for lv in sorted(r["level_stats"].keys()):
                    w, l, p = r["level_stats"][lv]
                    total = w + l
                    lvwr = w / total * 100 if total > 0 else 0
                    print(f"    {lv:<20} {w:>6} {l:>7} {lvwr:>5.1f}% {p:>+9.0f}")

    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("\nNext: design variants based on findings.")


if __name__ == "__main__":
    main()
