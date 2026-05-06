"""Day-of-week and hour-of-day stratification under slippage modeling.

Same C-config filters (buffer=1.0pt, drop FIB_EXT_LO_1.272). Looks
for buckets where slippage-adjusted $/tr or $/day differ meaningfully
from the level mean — would suggest a temporal filter could help.

The earlier "loss timing by hour" analysis (no-slippage) showed flat
WR across 10-15 ET. Under slippage, fill quality may vary (e.g., low-
volume afternoons might have worse fills, OR cleaner setups during
specific windows).
"""

from __future__ import annotations

import datetime
import os
import pickle
import sys
import time
from collections import defaultdict

import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import (
    simulate_fill, adjusted_pnl,
)

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}
ET = pytz.timezone("America/New_York")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    print("\nApplying slippage at C config (buffer=1.0pt)...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        dc = caches[r["date"]]
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=1.0, latency_ms=100.0,
        )
        if fill_idx is None:
            continue
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        ts = datetime.datetime.fromtimestamp(r["entry_ns"]/1e9, tz=pytz.utc)
        et = ts.astimezone(ET)
        enriched.append({
            "level": r["level"], "outcome": r["outcome"], "pnl": new_pnl,
            "weekday": et.weekday(), "hour": et.hour,
            "date": r["date"],
        })

    days = len({r["date"] for r in enriched})
    days_by_weekday = defaultdict(set)
    for t in enriched:
        days_by_weekday[t["weekday"]].add(t["date"])

    # ---------- DAY OF WEEK ----------
    print(f"\n{'='*78}\nDAY OF WEEK (slippage-modeled, buffer=1.0pt, C config)\n{'='*78}")
    print(f"  {'Day':<5} {'#days':>5} {'N':>5} {'W':>4} {'L':>4} {'WR%':>5} "
          f"{'$/tr':>7} {'$/day':>7} {'days/wk':>8}")
    by_wd = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "pnl": 0.0})
    for t in enriched:
        s = by_wd[t["weekday"]]
        s["n"] += 1
        s["pnl"] += t["pnl"]
        if t["outcome"] == "win": s["w"] += 1
        elif t["outcome"] == "loss": s["l"] += 1
    for wd in range(5):
        s = by_wd[wd]
        wd_days = len(days_by_weekday[wd])
        wr = s["w"] / s["n"] * 100 if s["n"] else 0
        ptr = s["pnl"] / s["n"] if s["n"] else 0
        # $/day on weekdays of this kind = total / wd_days
        pday = s["pnl"] / wd_days if wd_days else 0
        print(f"  {WEEKDAYS[wd]:<5} {wd_days:>5} {s['n']:>5} {s['w']:>4} {s['l']:>4} "
              f"{wr:>5.1f} {ptr:>+7.2f} {pday:>+7.2f}")

    # ---------- HOUR OF DAY ----------
    print(f"\n{'='*78}\nHOUR OF DAY (ET)\n{'='*78}")
    print(f"  {'Hour':>4} {'N':>5} {'W':>4} {'L':>4} {'WR%':>5} "
          f"{'$/tr':>7} {'$/day':>7}")
    by_hr = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "pnl": 0.0})
    for t in enriched:
        s = by_hr[t["hour"]]
        s["n"] += 1
        s["pnl"] += t["pnl"]
        if t["outcome"] == "win": s["w"] += 1
        elif t["outcome"] == "loss": s["l"] += 1
    for hr in sorted(by_hr.keys()):
        s = by_hr[hr]
        wr = s["w"] / s["n"] * 100 if s["n"] else 0
        ptr = s["pnl"] / s["n"] if s["n"] else 0
        pday = s["pnl"] / days
        print(f"  {hr:>4} {s['n']:>5} {s['w']:>4} {s['l']:>4} "
              f"{wr:>5.1f} {ptr:>+7.2f} {pday:>+7.2f}")

    # ---------- LEVEL × DAY-OF-WEEK ----------
    print(f"\n{'='*78}\nLEVEL × WEEKDAY  ($/day contribution)\n{'='*78}")
    levels = sorted({t["level"] for t in enriched})
    print(f"  {'Level':<22} ", end="")
    for wd in range(5): print(f"{WEEKDAYS[wd]:>8}", end="")
    print()
    for lv in levels:
        print(f"  {lv:<22} ", end="")
        for wd in range(5):
            wd_days = len(days_by_weekday[wd])
            pnl = sum(t["pnl"] for t in enriched if t["level"]==lv and t["weekday"]==wd)
            pday = pnl / wd_days if wd_days else 0
            print(f"{pday:>+8.2f}", end="")
        print()

    # ---------- BEST/WORST FILTER CANDIDATES ----------
    print(f"\n{'='*78}\nFILTER CANDIDATES\n{'='*78}")
    # Drop a single weekday — does $/day improve?
    total_pnl = sum(t["pnl"] for t in enriched)
    baseline_per_day = total_pnl / days
    print(f"  baseline: ${baseline_per_day:+.2f}/day across {days} days")
    print(f"  Drop weekday: $/day if that weekday is excluded (other days unchanged)")
    for wd in range(5):
        wd_days = len(days_by_weekday[wd])
        wd_pnl = sum(t["pnl"] for t in enriched if t["weekday"]==wd)
        kept_days = days - wd_days
        kept_pnl = total_pnl - wd_pnl
        kept_per_day = kept_pnl / kept_days if kept_days else 0
        print(f"    drop {WEEKDAYS[wd]:<5}: dropping {wd_days} days, "
              f"kept ${kept_per_day:+.2f}/day  (Δ ${kept_per_day - baseline_per_day:+.2f})")
    print(f"  Drop hour: $/day if that hour is excluded")
    for hr in sorted(by_hr.keys()):
        hr_pnl = by_hr[hr]["pnl"]
        kept_pnl = total_pnl - hr_pnl
        kept_per_day = kept_pnl / days
        print(f"    drop {hr:>2}:00 ET: ${kept_per_day:+.2f}/day  "
              f"(Δ ${kept_per_day - baseline_per_day:+.2f}, "
              f"removes n={by_hr[hr]['n']})")


if __name__ == "__main__":
    main()
