"""momentum_continuation_sensitivity_2026_05_06.py — sensitivity sweep on
continuation trades (best retest definition from prior run).

Locks retest definition: tol=1pt, max_secs=60min (best variant: +$4.81/day,
4/4 walk-forward in momentum_continuation_explore_2026_05_06).

Sweeps:
  Target multiplier: 1.0x / 1.25x / 1.5x / 2.0x (continuation might justify wider T)
  Stop multiplier:   1.0x / 0.75x / 0.5x       (tighter S to limit loss size)
  IBH:               include / exclude         (IBH continuation was $0/day)

Total: 4 × 3 × 2 = 24 variants. All post-process, fast.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import simulate_fill

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
PER_LEVEL_BUFFER = {"IBH": 0.75}
MNQ_PT = 2.0

RETEST_TOL_PTS = 1.0
MAX_RETEST_SECS = 3600

T_MULT_SWEEP = [1.0, 1.25, 1.5, 2.0]
S_MULT_SWEEP = [1.0, 0.75, 0.5]
IBH_MODES = [("include_IBH", True), ("exclude_IBH", False)]


def find_retest_idx(prices, ts_ns, start_idx, line_price, retest_tol_pts,
                    max_secs, eod_cutoff_ns):
    start_ns = int(ts_ns[start_idx])
    deadline_ns = start_ns + int(max_secs * 1e9)
    n = len(prices)
    for i in range(start_idx + 1, n):
        ts = int(ts_ns[i])
        if ts >= eod_cutoff_ns or ts > deadline_ns:
            return None
        if abs(float(prices[i]) - line_price) <= retest_tol_pts:
            return i
    return None


def simulate_path_to_outcome(prices, ts_ns, fill_idx, fill_price, direction,
                             target_pts, stop_pts, eod_cutoff_ns):
    sign = 1 if direction == "up" else -1
    target_price = fill_price + sign * target_pts
    stop_price = fill_price - sign * stop_pts
    n = len(prices)
    for i in range(fill_idx, n):
        if int(ts_ns[i]) >= eod_cutoff_ns:
            break
        price = float(prices[i])
        if (sign == 1 and price >= target_price) or (sign == -1 and price <= target_price):
            return float(target_pts)
        if (sign == 1 and price <= stop_price) or (sign == -1 and price >= stop_price):
            return -float(stop_pts)
    return 0.0


def _eod_cutoff_ns(date) -> int:
    import datetime
    import pytz
    et = pytz.timezone("America/New_York")
    eod_local = et.localize(
        datetime.datetime(date.year, date.month, date.day, 16, 0, 0)
    )
    return int(eod_local.timestamp() * 1e9)


def main() -> None:
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    # Identify losing trades (with retest data already cached for speed)
    print("\nIdentifying losses + caching retest data...", flush=True)
    losing_trades = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        if r["date"] not in caches:
            continue
        dc = caches[r["date"]]
        buf = PER_LEVEL_BUFFER.get(r["level"], 1.0)
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=buf, latency_ms=100.0,
        )
        if fill_idx is None:
            continue
        eod = _eod_cutoff_ns(r["date"])
        sign = 1 if r["direction"] == "up" else -1
        target_price = fill_price + sign * r["target_pts"]
        stop_price = fill_price - sign * r["stop_pts"]
        target_idx = stop_idx = None
        for i in range(fill_idx, len(dc.full_prices)):
            if int(dc.full_ts_ns[i]) >= eod:
                break
            p = float(dc.full_prices[i])
            if target_idx is None and ((sign == 1 and p >= target_price) or (sign == -1 and p <= target_price)):
                target_idx = i
            if stop_idx is None and ((sign == 1 and p <= stop_price) or (sign == -1 and p >= stop_price)):
                stop_idx = i
            if target_idx is not None or stop_idx is not None:
                break
        if stop_idx is not None and (target_idx is None or stop_idx < target_idx):
            losing_trades.append({**r, "stop_idx": stop_idx})

    days = len({r["date"] for r in losing_trades})
    print(f"  {len(losing_trades)} losses over {days} days", flush=True)

    # Pre-compute retest results (one find_retest call per loss, reused across
    # T/S variants) — this saves 24x repeated walks.
    print("\nPre-computing retests (tol=1pt, max=60min)...", flush=True)
    retest_cache = []  # list of (trade, retest_idx) for retests that succeeded
    for trade in losing_trades:
        dc = caches[trade["date"]]
        eod = _eod_cutoff_ns(trade["date"])
        retest_idx = find_retest_idx(
            dc.full_prices, dc.full_ts_ns,
            start_idx=trade["stop_idx"], line_price=trade["line_price"],
            retest_tol_pts=RETEST_TOL_PTS, max_secs=MAX_RETEST_SECS,
            eod_cutoff_ns=eod,
        )
        if retest_idx is not None:
            retest_cache.append((trade, retest_idx))
    print(f"  {len(retest_cache)} retests found", flush=True)

    # Now sweep T/S × IBH inclusion
    print("\nSweeping T/S multipliers × IBH inclusion...", flush=True)
    print(f"  {'Variant':<32} {'n':>6} {'WR%':>5} {'$/tr':>7} {'$/day':>7} "
          f"{'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Pos/4':>6}")

    sorted_dates = sorted({r["date"] for r in losing_trades})
    q_size = len(sorted_dates) // 4
    quarters = [
        ("Q1", set(sorted_dates[:q_size])),
        ("Q2", set(sorted_dates[q_size:2 * q_size])),
        ("Q3", set(sorted_dates[2 * q_size:3 * q_size])),
        ("Q4", set(sorted_dates[3 * q_size:])),
    ]

    sweep_results = {}
    for ibh_label, include_ibh in IBH_MODES:
        for t_mult in T_MULT_SWEEP:
            for s_mult in S_MULT_SWEEP:
                label = f"{ibh_label}_T{t_mult}xS{s_mult}x"
                day_pnl: dict = defaultdict(float)
                wins = losses = timeouts = 0
                pnls = []
                for trade, retest_idx in retest_cache:
                    if not include_ibh and trade["level"] == "IBH":
                        continue
                    target_pts = trade["target_pts"] * t_mult
                    stop_pts = trade["stop_pts"] * s_mult
                    break_dir = "up" if trade["direction"] == "down" else "down"
                    buf = PER_LEVEL_BUFFER.get(trade["level"], 1.0)
                    dc = caches[trade["date"]]
                    eod = _eod_cutoff_ns(trade["date"])
                    retest_entry_ns = int(dc.full_ts_ns[retest_idx])
                    fill_idx, fill_price = simulate_fill(
                        dc.full_prices, dc.full_ts_ns,
                        entry_ns=retest_entry_ns, direction=break_dir,
                        line=trade["line_price"], buffer=buf, latency_ms=100.0,
                    )
                    if fill_idx is None:
                        continue
                    pnl_pts = simulate_path_to_outcome(
                        dc.full_prices, dc.full_ts_ns, fill_idx, fill_price,
                        break_dir, target_pts, stop_pts, eod,
                    )
                    pnl_dollars = pnl_pts * MNQ_PT
                    pnls.append(pnl_dollars)
                    day_pnl[trade["date"]] += pnl_dollars
                    if pnl_pts > 0:
                        wins += 1
                    elif pnl_pts < 0:
                        losses += 1
                    else:
                        timeouts += 1

                n = len(pnls)
                total = sum(pnls)
                wr = wins / max(n, 1) * 100
                ptr = total / max(n, 1)
                pday = total / days

                # MaxDD over continuation-only trajectory
                cum = 0.0; peak = 0.0; max_dd = 0.0
                for d in sorted_dates:
                    cum += day_pnl.get(d, 0.0)
                    if cum > peak:
                        peak = cum
                    if peak - cum > max_dd:
                        max_dd = peak - cum

                # Per-quarter
                per_q = []
                per_q_dd = []
                pos_q = 0
                for q_label, qd in quarters:
                    q_dates_sorted = [d for d in sorted_dates if d in qd]
                    q_total = sum(p for d, p in day_pnl.items() if d in qd)
                    q_days = len(q_dates_sorted)
                    qday = q_total / q_days if q_days else 0
                    per_q.append(qday)
                    if qday > 0.5:
                        pos_q += 1
                    qcum = 0.0; qpeak = 0.0; qdd = 0.0
                    for d in q_dates_sorted:
                        qcum += day_pnl.get(d, 0.0)
                        if qcum > qpeak:
                            qpeak = qcum
                        if qpeak - qcum > qdd:
                            qdd = qpeak - qcum
                    per_q_dd.append(qdd)

                # Worst single day
                worst_day = min(day_pnl.values()) if day_pnl else 0.0

                sweep_results[label] = {
                    "ibh_included": include_ibh,
                    "t_mult": t_mult, "s_mult": s_mult,
                    "n_filled": n, "wins": wins, "losses": losses,
                    "timeouts": timeouts, "wr_pct": wr,
                    "pnl_per_tr": ptr, "pnl_per_day": pday,
                    "max_dd": max_dd,
                    "worst_day": worst_day,
                    "walk_forward": {
                        "Q1": per_q[0], "Q2": per_q[1],
                        "Q3": per_q[2], "Q4": per_q[3],
                        "positive_quarters": pos_q,
                    },
                    "walk_forward_dd": {
                        "Q1": per_q_dd[0], "Q2": per_q_dd[1],
                        "Q3": per_q_dd[2], "Q4": per_q_dd[3],
                    },
                }
                print(f"  {label:<32} {n:>5} {wr:>4.1f}% ${ptr:>+5.2f} "
                      f"${pday:>+5.2f} DD${max_dd:>4.0f} ${per_q[0]:>+5.2f} "
                      f"${per_q[1]:>+5.2f} ${per_q[2]:>+5.2f} ${per_q[3]:>+5.2f} {pos_q}/4")

    # Find best variant (highest $/day with 4/4 walk-forward)
    print(f"\n{'='*94}\nBEST VARIANTS (4/4 walk-forward, ranked by $/day)\n{'='*94}")
    qualified = [
        (k, v) for k, v in sweep_results.items()
        if v["walk_forward"]["positive_quarters"] == 4
    ]
    qualified.sort(key=lambda x: -x[1]["pnl_per_day"])
    if qualified:
        for k, v in qualified[:5]:
            print(f"  {k}: ${v['pnl_per_day']:+.2f}/day, "
                  f"{v['n_filled']} trades, "
                  f"WR {v['wr_pct']:.1f}%, "
                  f"per-Q: Q1=${v['walk_forward']['Q1']:+.2f} "
                  f"Q2=${v['walk_forward']['Q2']:+.2f} "
                  f"Q3=${v['walk_forward']['Q3']:+.2f} "
                  f"Q4=${v['walk_forward']['Q4']:+.2f}")
    else:
        print("  No 4/4 variants found.")

    print(f"\n3/4 walk-forward variants ranked by $/day:")
    near_qual = [
        (k, v) for k, v in sweep_results.items()
        if v["walk_forward"]["positive_quarters"] == 3
        and v["pnl_per_day"] > 0
    ]
    near_qual.sort(key=lambda x: -x[1]["pnl_per_day"])
    for k, v in near_qual[:5]:
        print(f"  {k}: ${v['pnl_per_day']:+.2f}/day, "
              f"per-Q: Q1=${v['walk_forward']['Q1']:+.2f} "
              f"Q2=${v['walk_forward']['Q2']:+.2f} "
              f"Q3=${v['walk_forward']['Q3']:+.2f} "
              f"Q4=${v['walk_forward']['Q4']:+.2f}")

    # Save summary
    summary = {
        "generated_at": _dt.datetime.now().isoformat(),
        "n_losing_trades": len(losing_trades),
        "n_retests_found": len(retest_cache),
        "n_days": days,
        "retest_definition": {
            "tol_pts": RETEST_TOL_PTS,
            "max_secs": MAX_RETEST_SECS,
        },
        "sweep_results": sweep_results,
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results",
        "momentum_continuation_sensitivity_2026_05_06.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
