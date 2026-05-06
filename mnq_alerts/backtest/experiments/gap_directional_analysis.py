"""Gap-direction × (level, direction) directional analysis.

User hypothesis: when prev day's RTH close < today's open (gap UP),
SELL trades on levels are riskier (especially LOWER levels like IBL/
FIB_0.236) because price has bullish overnight bias. Vice versa for
gap DOWN: BUYs on UPPER levels (IBH/FIB_0.764) are riskier.

Computes:
  gap = today's_open_price - prev_day's_close_price
where prev_day's close = last tick of prior trading day at ~16:00 ET,
today's open = first tick of today's data at ~09:30 ET.

Stratifies slippage-modeled $/tr per (level, direction, gap bucket).
Looks for the specific asymmetry the hypothesis predicts.

Same C+IBH=0.75 baseline filters: drop FIB_EXT_LO_1.272.
"""

from __future__ import annotations

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
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
PER_LEVEL_BUFFER = {"IBH": 0.75}


def gap_bucket(g: float) -> str:
    if g <= -15: return "<=-15  large_DOWN"
    if g <= -5:  return "-15..-5 small_DOWN"
    if g <  +5:  return "-5..+5  no_gap"
    if g < +15:  return "+5..+15 small_UP"
    return ">=+15  large_UP"


BUCKET_ORDER = [
    "<=-15  large_DOWN",
    "-15..-5 small_DOWN",
    "-5..+5  no_gap",
    "+5..+15 small_UP",
    ">=+15  large_UP",
]


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    # Compute gap per day
    print("\nComputing gap per day (today's open - prev close)...", flush=True)
    gaps = {}
    for i, d in enumerate(dates):
        if i == 0:
            continue
        prior = dates[i - 1]
        prior_dc = caches[prior]
        today_dc = caches[d]
        if len(prior_dc.full_prices) == 0 or len(today_dc.full_prices) == 0:
            continue
        prev_close = float(prior_dc.full_prices[-1])
        today_open = float(today_dc.full_prices[0])
        gaps[d] = today_open - prev_close
    print(f"  {len(gaps)} days with gap computed (1 missing for first)", flush=True)
    # Distribution
    sorted_gaps = sorted(gaps.values())
    print(f"  gap distribution (pts): "
          f"min={min(sorted_gaps):.1f}, p10={sorted_gaps[len(sorted_gaps)//10]:.1f}, "
          f"p50={sorted_gaps[len(sorted_gaps)//2]:.1f}, "
          f"p90={sorted_gaps[9*len(sorted_gaps)//10]:.1f}, max={max(sorted_gaps):.1f}")

    # Apply slippage and tag each trade with its day's gap
    print("\nApplying slippage + tagging trades with gap bucket...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        if r["date"] not in gaps:
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
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        enriched.append({
            "date": r["date"],
            "level": r["level"],
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pnl": new_pnl,
            "gap": gaps[r["date"]],
        })

    days = len({e["date"] for e in enriched})
    n = len(enriched)
    total_pnl = sum(e["pnl"] for e in enriched)
    print(f"\n  enriched: {n} trades over {days} days, "
          f"${total_pnl/days:+.2f}/day baseline", flush=True)

    # Days per bucket
    days_per_bucket = defaultdict(set)
    for e in enriched:
        days_per_bucket[gap_bucket(e["gap"])].add(e["date"])

    # === 1. Gap × Direction (level-aggregate) ===
    print(f"\n{'='*94}\nGAP BUCKET × DIRECTION  (all levels combined)\n{'='*94}")
    print(f"  {'Bucket':<22} {'#days':>5} | "
          f"{'BUY n':>5} {'BUY $/tr':>9} {'BUY WR%':>7} | "
          f"{'SELL n':>6} {'SELL $/tr':>9} {'SELL WR%':>8} | "
          f"{'Δ (S-B)':>8}")
    grp = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for e in enriched:
        bk = gap_bucket(e["gap"])
        s = grp[(bk, e["direction"])]
        s["n"] += 1
        s["pnl"] += e["pnl"]
        if e["outcome"] == "win":
            s["w"] += 1
    for bk in BUCKET_ORDER:
        buy = grp.get((bk, "up"), {"n": 0, "w": 0, "pnl": 0.0})
        sell = grp.get((bk, "down"), {"n": 0, "w": 0, "pnl": 0.0})
        n_days = len(days_per_bucket[bk])
        bp = buy["pnl"]/buy["n"] if buy["n"] else 0
        sp = sell["pnl"]/sell["n"] if sell["n"] else 0
        bw = buy["w"]/buy["n"]*100 if buy["n"] else 0
        sw = sell["w"]/sell["n"]*100 if sell["n"] else 0
        flag = " ⚠" if min(buy["n"], sell["n"]) >= 100 and abs(sp - bp) >= 1.0 else ""
        print(f"  {bk:<22} {n_days:>5} | "
              f"{buy['n']:>5} {bp:>+9.2f} {bw:>7.1f} | "
              f"{sell['n']:>6} {sp:>+9.2f} {sw:>8.1f} | "
              f"{sp-bp:>+8.2f}{flag}")

    # === 2. Per-level breakdown for the most-asymmetric buckets ===
    # Per-user hypothesis: lower levels (FIB_0.236) sell-heavy bad on gap up
    print(f"\n{'='*94}\nPER-LEVEL × DIRECTION on GAP-UP days (combined small_UP + large_UP)\n{'='*94}")
    print(f"  {'Level':<22} | {'BUY n':>5} {'BUY $/tr':>9} | {'SELL n':>6} {'SELL $/tr':>9} | {'Δ S-B':>7}")
    levels = sorted({e["level"] for e in enriched})
    for lv in levels:
        ups = [e for e in enriched if e["level"] == lv and e["gap"] > 5]
        buys = [e for e in ups if e["direction"] == "up"]
        sells = [e for e in ups if e["direction"] == "down"]
        bn = len(buys); sn = len(sells)
        bp = sum(e["pnl"] for e in buys)/bn if bn else 0
        sp = sum(e["pnl"] for e in sells)/sn if sn else 0
        flag = " ⚠ SELL bad" if sn >= 50 and sp < bp - 1.0 else ""
        print(f"  {lv:<22} | {bn:>5} {bp:>+9.2f} | {sn:>6} {sp:>+9.2f} | "
              f"{sp-bp:>+7.2f}{flag}")

    print(f"\n{'='*94}\nPER-LEVEL × DIRECTION on GAP-DOWN days (combined small_DOWN + large_DOWN)\n{'='*94}")
    print(f"  {'Level':<22} | {'BUY n':>5} {'BUY $/tr':>9} | {'SELL n':>6} {'SELL $/tr':>9} | {'Δ S-B':>7}")
    for lv in levels:
        downs = [e for e in enriched if e["level"] == lv and e["gap"] < -5]
        buys = [e for e in downs if e["direction"] == "up"]
        sells = [e for e in downs if e["direction"] == "down"]
        bn = len(buys); sn = len(sells)
        bp = sum(e["pnl"] for e in buys)/bn if bn else 0
        sp = sum(e["pnl"] for e in sells)/sn if sn else 0
        flag = " ⚠ BUY bad" if bn >= 50 and bp < sp - 1.0 else ""
        print(f"  {lv:<22} | {bn:>5} {bp:>+9.2f} | {sn:>6} {sp:>+9.2f} | "
              f"{sp-bp:>+7.2f}{flag}")

    # === 3. Filter sweep candidates ===
    print(f"\n{'='*94}\nFILTER CANDIDATE: drop SELL when gap >= X (test cutoffs)\n{'='*94}")
    print(f"  baseline ${total_pnl/days:+.2f}/day across {n} trades")
    for cutoff in [3, 5, 10, 15, 20]:
        kept = [e for e in enriched if not (e["direction"] == "down" and e["gap"] >= cutoff)]
        dropped = [e for e in enriched if (e["direction"] == "down" and e["gap"] >= cutoff)]
        kept_pnl = sum(e["pnl"] for e in kept)
        dropped_pnl = sum(e["pnl"] for e in dropped)
        dropped_ptr = dropped_pnl/len(dropped) if dropped else 0
        delta = (kept_pnl - total_pnl) / days
        print(f"  drop SELL when gap >= +{cutoff}: "
              f"kept_n={len(kept):>5}  dropped={len(dropped):>4}  "
              f"kept_$/day=${kept_pnl/days:>+7.2f}  "
              f"dropped_$/tr=${dropped_ptr:>+6.2f}  "
              f"Δ=${delta:>+6.2f}/day")

    print(f"\n  drop BUY when gap <= X (vice versa)")
    for cutoff in [-3, -5, -10, -15, -20]:
        kept = [e for e in enriched if not (e["direction"] == "up" and e["gap"] <= cutoff)]
        dropped = [e for e in enriched if (e["direction"] == "up" and e["gap"] <= cutoff)]
        kept_pnl = sum(e["pnl"] for e in kept)
        dropped_pnl = sum(e["pnl"] for e in dropped)
        dropped_ptr = dropped_pnl/len(dropped) if dropped else 0
        delta = (kept_pnl - total_pnl) / days
        print(f"  drop BUY when gap <= {cutoff:>+3}: "
              f"kept_n={len(kept):>5}  dropped={len(dropped):>4}  "
              f"kept_$/day=${kept_pnl/days:>+7.2f}  "
              f"dropped_$/tr=${dropped_ptr:>+6.2f}  "
              f"Δ=${delta:>+6.2f}/day")


if __name__ == "__main__":
    main()
