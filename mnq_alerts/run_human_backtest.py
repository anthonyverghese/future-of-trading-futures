"""Human app backtest — compare fixed 20-pt exit vs trade-reset exit.

Usage:
    python -u run_human_backtest.py
"""

import sys, os, time, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    DayCache, load_cached_days, load_day, preprocess_day,
    simulate_day as simulate_human_fixed, _run_zone_numpy,
    ALERT_THRESHOLD, EXIT_THRESHOLD, HIT_THRESHOLD, TARGET_POINTS, STOP_POINTS,
)
from bot_risk_backtest import evaluate_bot_trade, MULTIPLIER, FEE_PTS
from score_optimizer import Weights, score_alert, EnrichedAlert, compute_tick_rate
from walk_forward import _eod_cutoff_ns
from backtest.data import load_all_days, precompute_arrays

import pandas as pd
import pytz

_ET = pytz.timezone("America/New_York")

WIN_PNL = (TARGET_POINTS - FEE_PTS) * MULTIPLIER   # +$14.76
LOSS_PNL = -(STOP_POINTS + FEE_PTS) * MULTIPLIER   # -$41.24


def main():
    t0 = time.time()
    print("=" * 110)
    print("  HUMAN APP BACKTEST — Fixed 20pt exit vs trade-reset exit")
    print("=" * 110)

    valid_days, day_caches = load_all_days()
    N = len(valid_days)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    human_w = Weights()
    SCORE_THRESHOLDS = [0, 1, 2, 3, 4, 5, 6]

    # ═══════════════════════════════════════════════════════════
    # ZONE TYPE 1: Fixed 20-pt exit (current human app)
    # Uses existing simulate_day which is already fast.
    # Scoring uses searchsorted for tick rate (not pandas).
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  Zone: fixed-20pt-exit (current human app)")
    print(f"{'='*110}")
    print(f"  Simulating...", flush=True)
    t1 = time.time()

    cw = cl = 0
    all_fixed = []
    for di, date in enumerate(valid_days):
        if (di + 1) % 50 == 0:
            print(f"    {di+1}/{N}...", flush=True)
        dc = day_caches[date]
        first_price = float(dc.post_ib_prices[0])
        ft = dc.full_ts_ns
        utc_off = np.int64(_ET.localize(
            datetime.datetime.combine(date, datetime.time(12, 0))
        ).utcoffset().total_seconds() * 1e9)

        alerts = simulate_human_fixed(dc)
        alerts.sort(key=lambda a: a.alert_time)
        for a in alerts:
            if a.outcome not in ("correct", "incorrect"):
                continue
            # Fast scoring: compute tick rate via searchsorted, not pandas.
            alert_ns = int(pd.Timestamp(a.alert_time).value)
            gi = int(np.searchsorted(ft, alert_ns, side="left"))
            et_min = int(((alert_ns + utc_off) // 60_000_000_000) % 1440)
            now_et = datetime.time(et_min // 60, et_min % 60)
            w3 = alert_ns - 180_000_000_000
            tr_start = int(np.searchsorted(ft, w3, side="left"))
            tr = (gi - tr_start) / 3.0
            sm = a.entry_price - first_price

            ea = EnrichedAlert(
                date=date, level=a.level, direction=a.direction,
                entry_count=a.level_test_count, outcome=a.outcome,
                entry_price=a.entry_price, line_price=a.line_price,
                alert_time=a.alert_time, now_et=now_et, tick_rate=tr,
                session_move_pts=sm, consecutive_wins=cw, consecutive_losses=cl,
            )
            sc = score_alert(ea, human_w)
            if a.outcome == "correct": cw += 1; cl = 0
            else: cl += 1; cw = 0
            all_fixed.append((date, a.level, a.direction, a.level_test_count, a.outcome, sc))

    decided_fixed = len(all_fixed)
    correct_fixed = sum(1 for x in all_fixed if x[4] == "correct")
    print(f"  {decided_fixed} decided alerts ({decided_fixed/N:.1f}/d), "
          f"{correct_fixed/decided_fixed*100:.1f}% WR in {time.time()-t1:.0f}s")

    for ms in SCORE_THRESHOLDS:
        filtered = [x for x in all_fixed if x[5] >= ms]
        if not filtered: continue
        c = sum(1 for x in filtered if x[4] == "correct")
        ic = len(filtered) - c
        d = c + ic
        wr = c / d * 100 if d else 0
        pnl = c * WIN_PNL + ic * LOSS_PNL
        print(f"    score>={ms:>2}: {d:>5} ({d/N:.1f}/d)  {c}W/{ic}L  {wr:.1f}% WR  "
              f"PnL {pnl:+,.0f} ({pnl/N:+.1f}/d)")

    # ═══════════════════════════════════════════════════════════
    # ZONE TYPE 2: Trade-reset exit
    # Zone resets when outcome is decided (correct/incorrect).
    # Uses _run_zone_numpy per level with a per-entry approach.
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  Zone: trade-reset-exit (zone resets when outcome decided)")
    print(f"{'='*110}")
    print(f"  Simulating...", flush=True)
    t2 = time.time()

    cw = cl = 0
    all_reset = []
    window_ns = np.int64(15 * 60 * 1_000_000_000)

    for di, date in enumerate(valid_days):
        dc = day_caches[date]
        prices = dc.post_ib_prices
        n = len(prices)
        start = dc.post_ib_start_idx
        fp = dc.full_prices
        ft = dc.full_ts_ns
        first_price = float(prices[0])
        eod = _eod_cutoff_ns(date)
        utc_off_ns = np.int64(_ET.localize(
            datetime.datetime.combine(date, datetime.time(12, 0))
        ).utcoffset().total_seconds() * 1e9)
        at = ALERT_THRESHOLD

        # Clip to EOD once.
        eod_local = int(np.searchsorted(ft[start:start+n], eod, side="left"))

        # Process each level independently using numpy.
        level_configs = [
            ("IBH", np.full(n, dc.ibh)),
            ("IBL", np.full(n, dc.ibl)),
            ("FIB_EXT_HI_1.272", np.full(n, dc.fib_hi)),
            ("FIB_EXT_LO_1.272", np.full(n, dc.fib_lo)),
        ]
        if dc.post_ib_vwaps is not None:
            level_configs.append(("VWAP", dc.post_ib_vwaps))

        day_entries = []

        for lv_name, lv_arr in level_configs:
            # Vectorized distance computation.
            dist_arr = np.abs(prices[:eod_local] - lv_arr[:eod_local])
            in_zone_arr = dist_arr <= at  # boolean: within 7-pt zone

            # Walk transitions, not every tick. Find candidate entries:
            # indices where in_zone transitions from False to True.
            # After each entry, skip forward to where price leaves zone (pending_clear).
            ec_lv = 0
            j = 0
            m = len(dist_arr)
            while j < m:
                # Skip ticks outside zone.
                if not in_zone_arr[j]:
                    j += 1
                    continue

                # Zone entry at j.
                ec_lv += 1
                gi = start + j
                pj = float(prices[j])
                lv_price = float(lv_arr[j])
                direction = "up" if pj > lv_price else "down"

                # Score (fast: searchsorted for tick rate).
                ft_gi = ft[gi]
                et_min = int(((ft_gi + utc_off_ns) // 60_000_000_000) % 1440)
                now_et = datetime.time(et_min // 60, et_min % 60)
                w3 = ft_gi - np.int64(180_000_000_000)
                tr_start = int(np.searchsorted(ft, w3, side="left"))
                tr = (gi - tr_start) / 3.0
                sm = pj - first_price

                ea = EnrichedAlert(
                    date=date, level=lv_name, direction=direction,
                    entry_count=ec_lv, outcome="correct",
                    entry_price=pj, line_price=lv_price,
                    alert_time=datetime.datetime.now(), now_et=now_et,
                    tick_rate=tr, session_move_pts=sm,
                    consecutive_wins=cw, consecutive_losses=cl,
                )
                sc = score_alert(ea, human_w)

                # Find line hit (vectorized).
                ens = int(ft_gi)
                end_idx = min(int(np.searchsorted(ft, ens + window_ns, side="right")), len(fp))
                hit_idx = -1
                if gi + 1 < end_idx:
                    wp = fp[gi + 1 : end_idx]
                    hits = np.nonzero(np.abs(wp - lv_price) <= HIT_THRESHOLD)[0]
                    if len(hits) > 0:
                        hit_idx = gi + 1 + hits[0]

                if hit_idx < 0:
                    outcome = "inconclusive"
                else:
                    out, _, _ = evaluate_bot_trade(
                        hit_idx, lv_price, direction,
                        ft, fp, TARGET_POINTS, STOP_POINTS, 900, eod,
                    )
                    outcome = "correct" if out == "win" else "incorrect"

                if outcome != "inconclusive":
                    day_entries.append((gi, lv_name, direction, ec_lv, outcome, sc))

                # Zone resets. Skip forward until price leaves entry zone (pending_clear).
                j += 1
                while j < m and in_zone_arr[j]:
                    j += 1
                # j now points to the first tick outside the zone (cleared), advance past it.

        # Sort entries by time, update streak.
        day_entries.sort(key=lambda x: x[0])
        for gi, lv_name, direction, ec_lv, outcome, sc in day_entries:
            all_reset.append((date, lv_name, direction, ec_lv, outcome, sc))
            if outcome == "correct": cw += 1; cl = 0
            else: cl += 1; cw = 0

        if (di + 1) % 50 == 0:
            print(f"    {di+1}/{N}...", flush=True)

    decided_reset = len(all_reset)
    correct_reset = sum(1 for x in all_reset if x[4] == "correct")
    print(f"  {decided_reset} decided alerts ({decided_reset/N:.1f}/d), "
          f"{correct_reset/decided_reset*100:.1f}% WR in {time.time()-t2:.0f}s")

    for ms in SCORE_THRESHOLDS:
        filtered = [x for x in all_reset if x[5] >= ms]
        if not filtered: continue
        c = sum(1 for x in filtered if x[4] == "correct")
        ic = len(filtered) - c
        d = c + ic
        wr = c / d * 100 if d else 0
        pnl = c * WIN_PNL + ic * LOSS_PNL
        print(f"    score>={ms:>2}: {d:>5} ({d/N:.1f}/d)  {c}W/{ic}L  {wr:.1f}% WR  "
              f"PnL {pnl:+,.0f} ({pnl/N:+.1f}/d)")

    # ═══════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  COMPARISON — ranked by P&L/day")
    print(f"{'='*110}\n")
    all_results = []
    for ms in SCORE_THRESHOLDS:
        for label, data in [("fixed-20pt", all_fixed), ("trade-reset", all_reset)]:
            filtered = [x for x in data if x[5] >= ms]
            if not filtered: continue
            c = sum(1 for x in filtered if x[4] == "correct")
            ic = len(filtered) - c
            d = c + ic
            wr = c / d * 100 if d else 0
            pnl = c * WIN_PNL + ic * LOSS_PNL
            ppd = pnl / N
            all_results.append((f"{label} score>={ms}", d, d/N, wr, ppd))

    all_results.sort(key=lambda x: x[4], reverse=True)
    print(f"  {'Config':>35s} {'Alerts/d':>8} {'WR%':>6} {'$/day':>7}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*7}")
    for label, total, per_day, wr, ppd in all_results:
        print(f"  {label:>35s} {per_day:>7.1f} {wr:>5.1f}% {ppd:>+6.1f}")

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
