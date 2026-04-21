"""Run human app backtest — compare fixed exit vs trade-reset exit.

Tests the human alert system with:
1. Current logic: 7-pt entry, 20-pt fixed exit
2. New logic: 7-pt entry, zone resets when outcome is decided

Both with score threshold sweeps.

Usage:
    python -u run_human_backtest.py
"""

import sys, os, time, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from backtest.data import load_all_days, precompute_arrays, DayArrays
from backtest.zones import HumanZone, HumanZoneTradeReset
from backtest.scoring import score_entry, train_weights, EntryFactors, HUMAN_WEIGHTS
from backtest.report import fmt
from backtest.results import (
    BacktestParams, BacktestResult, compute_stats, save_result, display_results,
)

from targeted_backtest import DayCache, HIT_THRESHOLD
from bot_risk_backtest import evaluate_bot_trade, MULTIPLIER, FEE_PTS
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS
from score_optimizer import Weights, score_alert, EnrichedAlert, compute_tick_rate

import pandas as pd
import pytz

_ET = pytz.timezone("America/New_York")

TARGET_PTS = 8.0
STOP_PTS = 20.0


# ═══════════════════════════════════════════════════════════════
# HUMAN SIMULATION (two-stage: alert → line hit → T/S evaluation)
# ═══════════════════════════════════════════════════════════════

class HumanTradeRecord:
    def __init__(self, date, level, direction, entry_count, outcome, pnl_usd, score):
        self.date = date
        self.level = level
        self.direction = direction
        self.entry_count = entry_count
        self.outcome = outcome  # "correct", "incorrect", "inconclusive"
        self.pnl_usd = pnl_usd
        self.score = score
        self.exit_ns = 0


def simulate_human_day(
    dc: DayCache,
    arrays: DayArrays,
    zone_factory,  # callable(name, price, drifts) → zone
    human_w: Weights,
    cw: int, cl: int,
) -> tuple[list[HumanTradeRecord], int, int]:
    """Simulate one day of human alerts with pluggable zone logic.

    Two-stage evaluation per alert:
    1. Zone enters at 7 pts → compute score
    2. Check if price hits line (within 1 pt) within 15 min
    3. If hit: evaluate T8/S20 from line → correct/incorrect
    4. If not: inconclusive
    5. Zone resets when outcome is decided
    """
    prices = dc.post_ib_prices
    n = len(prices)
    start = dc.post_ib_start_idx
    fp = dc.full_prices
    ft = dc.full_ts_ns
    first_price = float(prices[0])
    eod = _eod_cutoff_ns(dc.date)

    levels = [
        ("IBH", dc.ibh, False),
        ("IBL", dc.ibl, False),
        ("FIB_EXT_HI_1.272", dc.fib_hi, False),
        ("FIB_EXT_LO_1.272", dc.fib_lo, False),
        ("VWAP", dc.post_ib_vwaps, True),
    ]

    zones = {}
    for name, price_or_arr, drifts in levels:
        price = float(price_or_arr) if not isinstance(price_or_arr, np.ndarray) else float(price_or_arr[0])
        zones[name] = zone_factory(name, price, drifts)

    trades = []

    # Pre-filter: ticks within 7 pts of any level.
    near_mask = np.zeros(n, dtype=bool)
    for name, price_or_arr, drifts in levels:
        if isinstance(price_or_arr, np.ndarray):
            near_mask |= np.abs(prices - price_or_arr) <= 7.0
        else:
            near_mask |= np.abs(prices - price_or_arr) <= 7.0
    candidates = np.nonzero(near_mask)[0]

    for j in candidates:
        gi = start + j
        ens = int(ft[gi])
        if ens >= eod:
            break

        pj = float(prices[j])

        # Update VWAP zone price.
        if isinstance(levels[4][1], np.ndarray):
            zones["VWAP"].price = float(dc.post_ib_vwaps[j])

        for name, zone in zones.items():
            if zone.in_zone:
                continue
            if not zone.update(pj):
                continue

            # Zone entry fired. Score using human scoring.
            direction = "up" if pj > zone.price else "down"
            # Use precomputed arrays instead of expensive pandas calls.
            et_min = int(arrays.et_mins[gi])
            now_et = datetime.time(et_min // 60, et_min % 60)
            tr = float(arrays.tick_rates[gi])
            sm = float(arrays.session_move[gi])

            ea = EnrichedAlert(
                date=dc.date, level=name, direction=direction,
                entry_count=zone.entry_count, outcome="correct",
                entry_price=pj, line_price=zone.price,
                alert_time=datetime.datetime.now(), now_et=now_et, tick_rate=tr,
                session_move_pts=sm, consecutive_wins=cw,
                consecutive_losses=cl,
            )
            sc = score_alert(ea, human_w)

            # Two-stage evaluation: find line hit, then evaluate T/S.
            window_ns = np.int64(15 * 60 * 1_000_000_000)
            hit_end = ens + window_ns
            hit_idx = -1
            for k in range(gi + 1, len(fp)):
                if ft[k] > hit_end:
                    break
                if abs(float(fp[k]) - zone.price) <= HIT_THRESHOLD:
                    hit_idx = k
                    break

            if hit_idx < 0:
                # Inconclusive — price never hit line.
                outcome = "inconclusive"
                pnl_usd = 0.0
                exit_ns = int(ft[min(gi + 1, len(ft) - 1)])
            else:
                # Evaluate T/S from line.
                out, eidx, pnl_pts = evaluate_bot_trade(
                    hit_idx, zone.price, direction,
                    ft, fp, TARGET_PTS, STOP_PTS, 900, eod,
                )
                exit_ns = int(ft[eidx])
                if out == "win":
                    outcome = "correct"
                    pnl_usd = (TARGET_PTS - FEE_PTS) * MULTIPLIER
                else:
                    outcome = "incorrect"
                    pnl_usd = -(STOP_PTS + FEE_PTS) * MULTIPLIER

            rec = HumanTradeRecord(dc.date, name, direction, zone.entry_count,
                                    outcome, pnl_usd, sc)
            rec.exit_ns = exit_ns
            trades.append(rec)

            # Update streak (on decided outcomes only).
            if outcome == "correct":
                cw += 1; cl = 0
            elif outcome == "incorrect":
                cw = 0; cl += 1

            # Zone resets when outcome is decided.
            zone.reset()
            break  # process one entry per tick

    return trades, cw, cl


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 110)
    print("  HUMAN APP BACKTEST — Fixed exit vs trade-reset exit")
    print("=" * 110)

    valid_days, day_caches = load_all_days()
    N = len(valid_days)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    print(f"  Precomputing factor arrays...", flush=True)
    t1 = time.time()
    arrays = {}
    for date in valid_days:
        arrays[date] = precompute_arrays(day_caches[date])
    print(f"  Done in {time.time()-t1:.0f}s")

    human_w = Weights()

    ZONE_TYPES = {
        "fixed-20pt-exit": lambda name, price, drifts: HumanZone(price, drifts),
        "trade-reset-exit": lambda name, price, drifts: HumanZoneTradeReset(price, drifts),
    }
    SCORE_THRESHOLDS = [0, 1, 2, 3, 4, 5]

    all_results = []

    for zone_name, zone_factory in ZONE_TYPES.items():
        print(f"\n{'='*110}")
        print(f"  Zone: {zone_name}")
        print(f"{'='*110}")

        # Simulate all days.
        print(f"  Simulating...", flush=True)
        t1 = time.time()
        all_trades = []
        trades_by_date = {}
        cw = cl = 0
        for date in valid_days:
            trades, cw, cl = simulate_human_day(
                day_caches[date], arrays[date], zone_factory, human_w, cw, cl,
            )
            all_trades.extend(trades)
            trades_by_date[date] = trades
        decided = [t for t in all_trades if t.outcome != "inconclusive"]
        correct = sum(1 for t in decided if t.outcome == "correct")
        print(f"  {len(all_trades)} alerts ({len(all_trades)/N:.1f}/d), "
              f"{len(decided)} decided ({correct}/{len(decided)} = "
              f"{correct/len(decided)*100:.1f}% WR) in {time.time()-t1:.0f}s")

        # Score threshold sweep.
        for min_score in SCORE_THRESHOLDS:
            # Filter by score, apply 1-position constraint per level.
            filtered = []
            for date in valid_days:
                pos_exit = 0
                for t in trades_by_date.get(date, []):
                    if t.outcome == "inconclusive":
                        continue
                    if t.score < min_score:
                        continue
                    filtered.append(t)

            if not filtered:
                continue

            w = sum(1 for t in filtered if t.outcome == "correct")
            l = sum(1 for t in filtered if t.outcome == "incorrect")
            d = w + l
            wr = w / d * 100 if d else 0
            pnl = sum(t.pnl_usd for t in filtered)
            ppd = pnl / N

            label = f"{zone_name} score>={min_score}"
            print(f"  {label:>40s}  {d:>4} ({d/N:.1f}/d)  "
                  f"{w}W/{l}L  {wr:.1f}% WR  PnL {pnl:+,.0f} ({ppd:+.1f}/d)")
            all_results.append((label, d, d/N, wr, ppd))

    # Summary.
    print(f"\n{'='*110}")
    print(f"  COMPARISON — ranked by P&L/day")
    print(f"{'='*110}\n")
    all_results.sort(key=lambda x: x[4], reverse=True)
    print(f"  {'Config':>45s} {'Alerts/d':>8} {'WR%':>6} {'$/day':>7}")
    print(f"  {'-'*45} {'-'*8} {'-'*6} {'-'*7}")
    for label, total, per_day, wr, ppd in all_results:
        print(f"  {label:>45s} {per_day:>7.1f} {wr:>5.1f}% {ppd:>+6.1f}")

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
