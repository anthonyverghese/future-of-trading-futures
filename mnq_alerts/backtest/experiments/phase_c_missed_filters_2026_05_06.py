"""phase_c_missed_filters_2026_05_06.py — pickle-based filter discovery.

Post-process on the existing 5,634-trade pickle (slippage-applied) to
look for buckets with consistently negative $/tr that the filter audit
missed. Tests:
  1. IB-range regime (small/medium/large IB days)
  2. Hour-of-day finer buckets (30-min slices)
  3. Time-since-last-trade buckets

For each candidate filter that shows aggregate negative $/tr, runs a
per-quarter walk-forward to verify regime robustness (3-4/4 quarters
required to flag for real-sim).

Wall clock: ~5 min (cache load + post-process).
"""

from __future__ import annotations

import datetime as _dt
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


def main() -> None:
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    # Compute IB range per day
    ib_ranges = {d: caches[d].ibh - caches[d].ibl for d in dates_all
                 if caches[d].ibh and caches[d].ibl}

    # Apply slippage and tag
    print("\nApplying slippage + tagging trades...", flush=True)
    enriched = []
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
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        # Hour bucket
        et_dt = _dt.datetime.fromtimestamp(r["entry_ns"] / 1e9)
        # Note: entry_ns is UTC; we'll bucket by UTC hour for simplicity
        # (consistent across days). Alternative is ET-aware but harder.
        hour_min = et_dt.hour * 60 + et_dt.minute
        enriched.append({
            "date": r["date"],
            "level": r["level"],
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pnl": new_pnl,
            "ib_range": ib_ranges.get(r["date"], 0),
            "entry_ns": r["entry_ns"],
            "hour_min": hour_min,
        })

    n = len(enriched)
    days = len({e["date"] for e in enriched})
    total = sum(e["pnl"] for e in enriched)
    print(f"  enriched: {n} trades over {days} days, "
          f"${total/days:+.2f}/day baseline", flush=True)

    sorted_dates = sorted({e["date"] for e in enriched})
    n_days = len(sorted_dates)
    q_size = n_days // 4
    quarters = [
        ("Q1", sorted_dates[:q_size]),
        ("Q2", sorted_dates[q_size:2 * q_size]),
        ("Q3", sorted_dates[2 * q_size:3 * q_size]),
        ("Q4", sorted_dates[3 * q_size:]),
    ]
    q_set = {q: set(d) for q, d in quarters}

    # ============= Filter 1: IB range regime =============
    print(f"\n{'='*94}")
    print("FILTER 1: IB-range regime")
    print(f"{'='*94}")
    print(f"  IB range distribution (pts): "
          f"min={min(ib_ranges.values()):.0f}, "
          f"p25={sorted(ib_ranges.values())[len(ib_ranges)//4]:.0f}, "
          f"p50={sorted(ib_ranges.values())[len(ib_ranges)//2]:.0f}, "
          f"p75={sorted(ib_ranges.values())[3*len(ib_ranges)//4]:.0f}, "
          f"max={max(ib_ranges.values()):.0f}")

    # Bucket by IB range
    def ib_bucket(ib: float) -> str:
        if ib < 40:
            return "small <40pt"
        if ib < 80:
            return "medium 40-80pt"
        return "large >=80pt"

    bucket_pnl = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for e in enriched:
        bk = ib_bucket(e["ib_range"])
        s = bucket_pnl[bk]
        s["n"] += 1
        s["pnl"] += e["pnl"]
        if e["outcome"] == "win":
            s["wins"] += 1
    print(f"\n  Aggregate per IB-range bucket:")
    print(f"  {'Bucket':<20} {'n':>5} {'$/tr':>8} {'WR%':>6} {'$total':>9}")
    for bk in ["small <40pt", "medium 40-80pt", "large >=80pt"]:
        s = bucket_pnl[bk]
        if s["n"] == 0:
            continue
        print(f"  {bk:<20} {s['n']:>5} {s['pnl']/s['n']:>+8.2f} "
              f"{s['wins']/s['n']*100:>5.1f}% {s['pnl']:>+9.0f}")

    # Per-quarter walk-forward for any negative bucket
    print(f"\n  Per-quarter $/tr by bucket:")
    print(f"  {'Bucket':<20} {'Q1 (n,$/tr)':>16} {'Q2':>16} {'Q3':>16} {'Q4':>16}")
    for bk in ["small <40pt", "medium 40-80pt", "large >=80pt"]:
        per_q = []
        for q_label, qdates in quarters:
            qd = q_set[q_label]
            in_b = [e for e in enriched
                    if e["date"] in qd and ib_bucket(e["ib_range"]) == bk]
            if in_b:
                per_q.append((len(in_b), sum(e["pnl"] for e in in_b) / len(in_b)))
            else:
                per_q.append((0, 0.0))
        cells = " ".join(f"({n},{p:>+6.2f})" for n, p in per_q).ljust(60)
        print(f"  {bk:<20} {cells}")

    # ============= Filter 2: hour-of-day fine bins =============
    print(f"\n{'='*94}")
    print("FILTER 2: hour-of-day (30-min UTC buckets)")
    print(f"{'='*94}")
    # Note: entry_ns is in nanoseconds since epoch UTC.
    # We bucket by UTC time. RTH 09:30-16:00 ET = 13:30-20:00 UTC (during DST).
    print(f"  Note: UTC time. RTH 09:30 ET ≈ 13:30 UTC during DST.")

    def hr_bucket(hm: int) -> str:
        h = hm // 60
        m = (hm % 60) // 30 * 30
        return f"{h:02d}:{m:02d}-{h+(m+30)//60:02d}:{(m+30)%60:02d}"

    bucket_pnl = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for e in enriched:
        bk = hr_bucket(e["hour_min"])
        s = bucket_pnl[bk]
        s["n"] += 1
        s["pnl"] += e["pnl"]
        if e["outcome"] == "win":
            s["wins"] += 1

    print(f"\n  Aggregate per 30-min bucket (n>=50 only):")
    print(f"  {'Bucket':<14} {'n':>5} {'$/tr':>8} {'WR%':>6} {'$total':>9} {'$/day':>9}")
    sorted_bks = sorted(bucket_pnl.keys())
    flagged = []
    for bk in sorted_bks:
        s = bucket_pnl[bk]
        if s["n"] < 50:
            continue
        ptr = s["pnl"] / s["n"]
        flag = " ⚠" if ptr < -0.5 else ""
        print(f"  {bk:<14} {s['n']:>5} {ptr:>+8.2f} "
              f"{s['wins']/s['n']*100:>5.1f}% {s['pnl']:>+9.0f} "
              f"{s['pnl']/days:>+9.2f}{flag}")
        if ptr < -0.5:
            flagged.append((bk, s))

    if flagged:
        print(f"\n  Per-quarter walk-forward on flagged hour buckets:")
        for bk, _ in flagged:
            print(f"\n    {bk}:")
            for q_label, qdates in quarters:
                qd = q_set[q_label]
                in_b = [e for e in enriched
                        if e["date"] in qd and hr_bucket(e["hour_min"]) == bk]
                if in_b:
                    pnl = sum(e["pnl"] for e in in_b)
                    print(f"      {q_label}: n={len(in_b)} "
                          f"$/tr={pnl/len(in_b):+6.2f} "
                          f"$tot={pnl:+7.0f}")

    # ============= Filter 3: time-since-last-trade =============
    print(f"\n{'='*94}")
    print("FILTER 3: time-since-last-trade-this-day")
    print(f"{'='*94}")

    # For each trade, find the most recent prior trade THAT day
    by_day = defaultdict(list)
    for e in enriched:
        by_day[e["date"]].append(e)
    for d in by_day:
        by_day[d].sort(key=lambda x: x["entry_ns"])
        prev_ns = None
        for e in by_day[d]:
            e["secs_since_prev"] = (
                (e["entry_ns"] - prev_ns) / 1e9 if prev_ns else None
            )
            prev_ns = e["entry_ns"]

    def gap_bucket(secs) -> str:
        if secs is None:
            return "first_trade"
        if secs < 60:
            return "<60s"
        if secs < 300:
            return "60-300s (1-5min)"
        if secs < 900:
            return "300-900s (5-15min)"
        if secs < 1800:
            return "900-1800s (15-30min)"
        return ">1800s (>30min)"

    bucket_pnl = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for d in by_day.values():
        for e in d:
            bk = gap_bucket(e["secs_since_prev"])
            s = bucket_pnl[bk]
            s["n"] += 1
            s["pnl"] += e["pnl"]
            if e["outcome"] == "win":
                s["wins"] += 1

    print(f"\n  Aggregate by gap-since-prev-trade-this-day:")
    print(f"  {'Bucket':<22} {'n':>5} {'$/tr':>8} {'WR%':>6} {'$total':>9}")
    order = ["first_trade", "<60s", "60-300s (1-5min)", "300-900s (5-15min)",
             "900-1800s (15-30min)", ">1800s (>30min)"]
    flagged = []
    for bk in order:
        s = bucket_pnl[bk]
        if s["n"] == 0:
            continue
        ptr = s["pnl"] / s["n"]
        flag = " ⚠" if ptr < -0.5 else ""
        print(f"  {bk:<22} {s['n']:>5} {ptr:>+8.2f} "
              f"{s['wins']/s['n']*100:>5.1f}% {s['pnl']:>+9.0f}{flag}")
        if ptr < -0.5 and s["n"] >= 100:
            flagged.append(bk)

    if flagged:
        print(f"\n  Per-quarter walk-forward on flagged gap buckets:")
        for bk in flagged:
            print(f"\n    {bk}:")
            for q_label, qdates in quarters:
                qd = q_set[q_label]
                in_b = []
                for d in by_day:
                    if d not in qd:
                        continue
                    for e in by_day[d]:
                        if gap_bucket(e["secs_since_prev"]) == bk:
                            in_b.append(e)
                if in_b:
                    pnl = sum(e["pnl"] for e in in_b)
                    print(f"      {q_label}: n={len(in_b)} "
                          f"$/tr={pnl/len(in_b):+6.2f}")

    print(f"\n{'='*94}\nDONE\n{'='*94}")


if __name__ == "__main__":
    main()
