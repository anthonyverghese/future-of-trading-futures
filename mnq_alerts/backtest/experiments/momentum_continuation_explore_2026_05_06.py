"""momentum_continuation_explore_2026_05_06.py — post-process upper bound on
break-and-retest entries.

The current bot is mean-reversion: SELL at IBH approach, BUY at FIB approach.
When a level BREAKS (our SELL stops out), the level becomes a *continuation*
signal — price went past it because of order flow. Hypothesis: trading the
break direction on retest captures that continuation move.

Test scope (this script):
  For every LOSING bot trade in the pickle (which is by definition a "level
  broke" event):
    - Walk tick path forward from stop hit
    - Look for "retest" within MAX_RETEST_SECS: price returns within
      RETEST_TOL_PTS of line_price
    - If retest found, simulate a NEW fill in the BREAK direction (opposite
      of original bot direction) using same buffer + slippage model
    - Walk path to T or S (using same T/S as the level's bot config)
    - Record outcome and P&L

If aggregate P&L per day is positive AND walk-forward 3-4/4 quarters,
this is a real signal worth implementing as a new entry mechanic.

Caveats:
  - Upper bound only — no cap-budget, position-1, daily-loss, timeout
  - Same T/S as bot — could be sub-optimal for continuation trades
  - Single retest definition — may not be the right one
  - 60min window for retest — could be too long/short
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

# Sweep multiple definitions to see sensitivity
RETEST_TOL_PTS_SWEEP = [1.0, 2.0, 3.0]
MAX_RETEST_SECS_SWEEP = [600, 1800, 3600]  # 10min / 30min / 60min


def find_break_idx(prices, fill_idx, fill_price, direction, stop_pts):
    """Find idx where stop was hit. Returns idx or None."""
    sign = 1 if direction == "up" else -1
    stop_price = fill_price - sign * stop_pts
    for i in range(fill_idx, len(prices)):
        price = prices[i]
        if (sign == 1 and price <= stop_price) or (sign == -1 and price >= stop_price):
            return i
    return None


def find_retest_idx(prices, ts_ns, start_idx, line_price, retest_tol_pts,
                    max_secs, eod_cutoff_ns):
    """Find first time price comes within retest_tol_pts of line_price after start_idx."""
    start_ns = int(ts_ns[start_idx])
    deadline_ns = start_ns + int(max_secs * 1e9)
    n = len(prices)
    for i in range(start_idx + 1, n):
        ts = int(ts_ns[i])
        if ts >= eod_cutoff_ns:
            return None
        if ts > deadline_ns:
            return None
        if abs(float(prices[i]) - line_price) <= retest_tol_pts:
            return i
    return None


def simulate_path_to_outcome(prices, ts_ns, fill_idx, fill_price, direction,
                             target_pts, stop_pts, eod_cutoff_ns):
    """Walk fill_idx forward until target/stop hit, or EOD. Returns pts."""
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
    return 0.0  # timeout/EOD


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

    # Step 1: enrich pickle trades with slippage-modeled outcomes (so we know
    # which were real losses under current deploy)
    print("\nStep 1: identifying real losses (slippage-modeled)...", flush=True)
    losing_trades = []
    n_skipped = 0
    n_fill_miss = 0
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        if r["date"] not in caches:
            n_skipped += 1
            continue
        dc = caches[r["date"]]
        buf = PER_LEVEL_BUFFER.get(r["level"], 1.0)
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=buf, latency_ms=100.0,
        )
        if fill_idx is None:
            n_fill_miss += 1
            continue
        # Check if this is a real loss (path: stop hit before target)
        eod = _eod_cutoff_ns(r["date"])
        sign = 1 if r["direction"] == "up" else -1
        target_price = fill_price + sign * r["target_pts"]
        stop_price = fill_price - sign * r["stop_pts"]
        target_idx = None
        stop_idx = None
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
            losing_trades.append({
                **r,
                "fill_idx": fill_idx,
                "fill_price": fill_price,
                "stop_idx": stop_idx,
                "stop_price": stop_price,
            })
    print(f"  identified {len(losing_trades)} losses "
          f"(skipped={n_skipped}, fill_miss={n_fill_miss})")

    # Step 2: for each parameter combo, simulate retest entries
    days = len({r["date"] for r in losing_trades})
    print(f"\nStep 2: testing retest variants over {days} days "
          f"({len(RETEST_TOL_PTS_SWEEP) * len(MAX_RETEST_SECS_SWEEP)} combos)...",
          flush=True)

    sweep_results = {}
    for tol in RETEST_TOL_PTS_SWEEP:
        for max_secs in MAX_RETEST_SECS_SWEEP:
            label = f"tol{tol}pt_max{max_secs//60}min"
            n_retest = 0
            n_no_retest = 0
            outcomes_per_trade = []
            day_pnl: dict = defaultdict(float)
            outcome_counts = defaultdict(int)

            for trade in losing_trades:
                dc = caches[trade["date"]]
                eod = _eod_cutoff_ns(trade["date"])
                line = trade["line_price"]

                retest_idx = find_retest_idx(
                    dc.full_prices, dc.full_ts_ns,
                    start_idx=trade["stop_idx"], line_price=line,
                    retest_tol_pts=tol, max_secs=max_secs, eod_cutoff_ns=eod,
                )
                if retest_idx is None:
                    n_no_retest += 1
                    continue

                # Simulate fill at retest in BREAK direction (opposite of original)
                break_direction = "up" if trade["direction"] == "down" else "down"
                buf = PER_LEVEL_BUFFER.get(trade["level"], 1.0)
                # Use the line price as the limit (same as bot enters)
                # Find a fresh entry near retest
                retest_entry_ns = int(dc.full_ts_ns[retest_idx])
                fill_idx, fill_price = simulate_fill(
                    dc.full_prices, dc.full_ts_ns,
                    entry_ns=retest_entry_ns, direction=break_direction,
                    line=line, buffer=buf, latency_ms=100.0,
                )
                if fill_idx is None:
                    # Couldn't fill the new limit
                    outcome_counts["unfilled"] += 1
                    continue

                # Walk path
                pnl_pts = simulate_path_to_outcome(
                    dc.full_prices, dc.full_ts_ns,
                    fill_idx, fill_price, break_direction,
                    trade["target_pts"], trade["stop_pts"], eod,
                )
                pnl_dollars = pnl_pts * MNQ_PT
                outcomes_per_trade.append(pnl_dollars)
                day_pnl[trade["date"]] += pnl_dollars
                if pnl_pts > 0:
                    outcome_counts["win"] += 1
                elif pnl_pts < 0:
                    outcome_counts["loss"] += 1
                else:
                    outcome_counts["timeout"] += 1
                n_retest += 1

            total = sum(outcomes_per_trade)
            n = len(outcomes_per_trade)
            sweep_results[label] = {
                "tol_pts": tol,
                "max_secs": max_secs,
                "n_retest_attempts": n_retest + outcome_counts["unfilled"],
                "n_filled": n,
                "n_no_retest": n_no_retest,
                "n_unfilled": outcome_counts["unfilled"],
                "wins": outcome_counts["win"],
                "losses": outcome_counts["loss"],
                "timeouts": outcome_counts["timeout"],
                "total_pnl": total,
                "pnl_per_day": total / days if days else 0,
                "wr_pct": (outcome_counts["win"] / max(n, 1)) * 100,
                "day_pnl": dict(day_pnl),
            }
            wr = (outcome_counts["win"] / max(n, 1)) * 100
            print(f"  {label}: {n} retests filled "
                  f"({n_no_retest} no-retest, {outcome_counts['unfilled']} unfilled), "
                  f"WR {wr:.1f}%, ${total/days:+.2f}/day", flush=True)

    # Step 3: walk-forward by quarter for the best params
    print(f"\n{'='*94}\nWALK-FORWARD by quarter\n{'='*94}")
    sorted_dates = sorted({r["date"] for r in losing_trades})
    q_size = len(sorted_dates) // 4
    quarters = [
        ("Q1", sorted_dates[:q_size]),
        ("Q2", sorted_dates[q_size:2 * q_size]),
        ("Q3", sorted_dates[2 * q_size:3 * q_size]),
        ("Q4", sorted_dates[3 * q_size:]),
    ]

    print(f"  {'Variant':<22} {'Q1 $/day':>10} {'Q2 $/day':>10} "
          f"{'Q3 $/day':>10} {'Q4 $/day':>10} {'Pos quarters':>14}")
    for label, sr in sweep_results.items():
        per_q = []
        pos_q = 0
        for q_label, qdates in quarters:
            qd = set(qdates)
            q_total = sum(p for d, p in sr["day_pnl"].items() if d in qd)
            qday = q_total / len(qdates)
            per_q.append(qday)
            if qday > 0.5:
                pos_q += 1
        sr["walk_forward"] = {
            "Q1": per_q[0], "Q2": per_q[1], "Q3": per_q[2], "Q4": per_q[3],
            "positive_quarters": pos_q,
        }
        print(f"  {label:<22} ${per_q[0]:>+9.2f} ${per_q[1]:>+9.2f} "
              f"${per_q[2]:>+9.2f} ${per_q[3]:>+9.2f} {pos_q}/4")

    # Step 4: per-level breakdown for best variant (highest aggregate $/day)
    best_label = max(sweep_results.keys(),
                     key=lambda k: sweep_results[k]["pnl_per_day"])
    print(f"\n{'='*94}\nPER-LEVEL — best variant: {best_label}\n{'='*94}")
    sr_best = sweep_results[best_label]
    by_level: dict = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0,
                                          "no_retest": 0})
    # Re-walk losses with best params to get per-level
    tol_best = sr_best["tol_pts"]
    max_best = sr_best["max_secs"]
    for trade in losing_trades:
        lv = trade["level"]
        dc = caches[trade["date"]]
        eod = _eod_cutoff_ns(trade["date"])
        line = trade["line_price"]
        retest_idx = find_retest_idx(
            dc.full_prices, dc.full_ts_ns,
            start_idx=trade["stop_idx"], line_price=line,
            retest_tol_pts=tol_best, max_secs=max_best, eod_cutoff_ns=eod,
        )
        if retest_idx is None:
            by_level[lv]["no_retest"] += 1
            continue
        break_dir = "up" if trade["direction"] == "down" else "down"
        buf = PER_LEVEL_BUFFER.get(lv, 1.0)
        retest_entry_ns = int(dc.full_ts_ns[retest_idx])
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=retest_entry_ns, direction=break_dir, line=line,
            buffer=buf, latency_ms=100.0,
        )
        if fill_idx is None:
            continue
        pnl_pts = simulate_path_to_outcome(
            dc.full_prices, dc.full_ts_ns, fill_idx, fill_price, break_dir,
            trade["target_pts"], trade["stop_pts"], eod,
        )
        pnl_dollars = pnl_pts * MNQ_PT
        s = by_level[lv]
        s["n"] += 1
        if pnl_pts > 0:
            s["wins"] += 1
        s["pnl"] += pnl_dollars

    print(f"  {'Level':<22} {'n':>5} {'WR%':>6} {'$/tr':>8} {'$/day':>8} "
          f"{'no-retest':>10}")
    for lv in sorted(by_level.keys()):
        s = by_level[lv]
        wr = s["wins"] / max(s["n"], 1) * 100
        ptr = s["pnl"] / max(s["n"], 1)
        pday = s["pnl"] / days
        print(f"  {lv:<22} {s['n']:>5} {wr:>5.1f}% ${ptr:>+7.2f} "
              f"${pday:>+7.2f} {s['no_retest']:>10}")

    # Save summary JSON
    summary = {
        "generated_at": _dt.datetime.now().isoformat(),
        "n_losing_trades": len(losing_trades),
        "n_days": days,
        "sweep_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "day_pnl"}
            for k, v in sweep_results.items()
        },
        "per_level_best_variant": {
            "variant": best_label,
            "by_level": {
                lv: {"n": s["n"], "wins": s["wins"], "pnl": s["pnl"],
                     "no_retest": s["no_retest"]}
                for lv, s in by_level.items()
            },
        },
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results",
        "momentum_continuation_explore_2026_05_06.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
