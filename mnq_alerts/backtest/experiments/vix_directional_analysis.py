"""Directional VIX analysis under slippage modeling.

User hypothesis: in high VIX regimes, BUY trades (fading dips) should
be worse than SELL trades (fading rips), because:
  - High VIX = fear regime, mean-reversion breaks down
  - Dips often continue (panic) → BUYs fail more
  - Rips often fail at resistance → SELLs work
  - Aligns with negative-gamma "sell-the-rip" theme

Prior VIX test (2026-04-28) used the SLIPPAGE-BLIND simulator and
tested generic cap halving (all directions equally). All variants came
in -$0.30 to -$3.65/day. But that's a different hypothesis from this one.

This re-test under slippage:
  1. Stratify slippage-adjusted $/tr by direction × VIX bucket
  2. Look specifically for asymmetry: are BUYs much worse in high VIX
     than SELLs?
  3. If yes, walk-forward + real-sim a directional cap filter.

Same C+IBH=0.75 baseline filters: drop FIB_EXT_LO, buffer per_level dict.
"""

from __future__ import annotations

import csv
import datetime
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import (
    simulate_fill, adjusted_pnl,
)

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
VIX_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data_cache", "vix_daily.csv"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
PER_LEVEL_BUFFER = {"IBH": 0.75}


def load_vix() -> tuple[dict, dict]:
    """Returns (open_vix, close_vix) keyed by date."""
    open_vix, close_vix = {}, {}
    with open(VIX_CSV_PATH) as f:
        for row in csv.DictReader(f):
            d = datetime.date.fromisoformat(row["date"])
            open_vix[d] = float(row["open"])
            close_vix[d] = float(row["close"])
    # Build prev-close lookup (yesterday's close as of today)
    prev_close = {}
    sorted_dates = sorted(close_vix.keys())
    for i in range(1, len(sorted_dates)):
        prev_close[sorted_dates[i]] = close_vix[sorted_dates[i - 1]]
    return open_vix, prev_close


def vix_bucket(v: float) -> str:
    if v < 15: return "<15"
    if v < 18: return "15-18"
    if v < 22: return "18-22"
    if v < 27: return "22-27"
    return ">=27"


BUCKET_ORDER = ["<15", "15-18", "18-22", "22-27", ">=27"]


def main():
    print("Loading pickle + VIX + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    open_vix, prev_close_vix = load_vix()
    print(f"  pickle: {len(rows)} trades; VIX: {len(open_vix)} open, "
          f"{len(prev_close_vix)} prev_close", flush=True)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    print("\nApplying slippage + tagging with VIX...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        d = r["date"]
        if d not in open_vix:
            continue  # skip days without VIX
        dc = caches[d]
        buf = PER_LEVEL_BUFFER.get(r["level"], 1.0)
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=buf, latency_ms=100.0,
        )
        if fill_idx is None:
            continue
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        enriched.append({
            "date": d,
            "level": r["level"],
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pnl": new_pnl,
            "open_vix": open_vix[d],
            "prev_close_vix": prev_close_vix.get(d),
        })

    days = len({e["date"] for e in enriched})
    n = len(enriched)
    total_pnl = sum(e["pnl"] for e in enriched)
    print(f"\n  enriched: {n} trades over {days} days, ${total_pnl/days:+.2f}/day baseline")

    # ============================================================
    # 1. Direction × Open-VIX bucket
    # ============================================================
    for vix_field, label in [("open_vix", "OPEN VIX"), ("prev_close_vix", "PREV CLOSE VIX")]:
        print(f"\n{'='*94}\n{label} × DIRECTION ($/tr, slippage-modeled)\n{'='*94}")
        # Group by (bucket, direction)
        grp = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
        for e in enriched:
            v = e[vix_field]
            if v is None:
                continue
            bk = vix_bucket(v)
            d = e["direction"]
            s = grp[(bk, d)]
            s["n"] += 1
            s["pnl"] += e["pnl"]
            if e["outcome"] == "win":
                s["w"] += 1
        # also days per bucket
        days_per_bucket = defaultdict(set)
        for e in enriched:
            v = e[vix_field]
            if v is None:
                continue
            days_per_bucket[vix_bucket(v)].add(e["date"])

        print(f"  {'Bucket':<10} {'N_total':>7} {'#days':>5} | "
              f"{'BUY n':>6} {'BUY $/tr':>9} {'BUY WR%':>7} | "
              f"{'SELL n':>6} {'SELL $/tr':>9} {'SELL WR%':>8} | "
              f"{'Δ$/tr':>7}")
        for bk in BUCKET_ORDER:
            buy = grp.get((bk, "up"), {"n": 0, "w": 0, "pnl": 0.0})
            sell = grp.get((bk, "down"), {"n": 0, "w": 0, "pnl": 0.0})
            n_tot = buy["n"] + sell["n"]
            n_days = len(days_per_bucket[bk])
            buy_ptr = buy["pnl"]/buy["n"] if buy["n"] else 0
            sell_ptr = sell["pnl"]/sell["n"] if sell["n"] else 0
            buy_wr = buy["w"]/buy["n"]*100 if buy["n"] else 0
            sell_wr = sell["w"]/sell["n"]*100 if sell["n"] else 0
            asymmetry = sell_ptr - buy_ptr
            flag = " ⚠" if buy["n"] >= 100 and abs(asymmetry) >= 1.0 else ""
            print(f"  {bk:<10} {n_tot:>7} {n_days:>5} | "
                  f"{buy['n']:>6} {buy_ptr:>+9.2f} {buy_wr:>7.1f} | "
                  f"{sell['n']:>6} {sell_ptr:>+9.2f} {sell_wr:>8.1f} | "
                  f"{asymmetry:>+7.2f}{flag}")

    # ============================================================
    # 2. Filter candidate: drop BUY trades when VIX > X
    # ============================================================
    print(f"\n{'='*94}\nFILTER CANDIDATE: drop BUY trades when open_VIX > X\n{'='*94}")
    print(f"  baseline: ${total_pnl/days:+.2f}/day, {n} trades")
    print(f"  {'cutoff':<10} {'kept_n':>7} {'dropped':>7} {'kept_$/day':>11} "
          f"{'dropped_$/day':>14} {'dropped_$/tr':>13}")
    for cutoff in [15, 18, 20, 22, 25, 27, 30]:
        kept = [e for e in enriched if not (e["direction"] == "up" and e["open_vix"] >= cutoff)]
        dropped = [e for e in enriched if (e["direction"] == "up" and e["open_vix"] >= cutoff)]
        kept_pnl = sum(e["pnl"] for e in kept)
        dropped_pnl = sum(e["pnl"] for e in dropped)
        dropped_ptr = dropped_pnl / len(dropped) if dropped else 0
        delta = (kept_pnl - total_pnl) / days
        print(f"  VIX>={cutoff:<6} {len(kept):>7} {len(dropped):>7} "
              f"{kept_pnl/days:>+11.2f} {dropped_pnl/days:>+14.2f} "
              f"{dropped_ptr:>+13.2f}")

    # Same for SELL trades (sanity check — should be opposite direction)
    print(f"\n  Sanity: drop SELL trades when open_VIX > X (expect this hurts more)")
    for cutoff in [22, 25, 27]:
        kept = [e for e in enriched if not (e["direction"] == "down" and e["open_vix"] >= cutoff)]
        dropped = [e for e in enriched if (e["direction"] == "down" and e["open_vix"] >= cutoff)]
        kept_pnl = sum(e["pnl"] for e in kept)
        dropped_pnl = sum(e["pnl"] for e in dropped)
        dropped_ptr = dropped_pnl / len(dropped) if dropped else 0
        delta = (kept_pnl - total_pnl) / days
        print(f"  SELL VIX>={cutoff:<6} kept_n={len(kept):>5} dropped={len(dropped):>5} "
              f"kept$/day={kept_pnl/days:>+8.2f} dropped$/tr={dropped_ptr:>+8.2f}")


if __name__ == "__main__":
    main()
