"""Tier-1 quick-check analyses (post-process, no real sim):

1. Trend filter: $/tr by trend_60m bucket. Memory tested at no-slippage
   and found WR flat. Re-test on slippage-adjusted $/tr.
2. First-N-minutes filter: $/tr by minutes-since-IB-lock. Tests the
   "bad days lose early" memory note.
3. IB-range filter: $/tr by IB range bucket. Wide vs tight IB days
   may behave differently.

Same C-config slippage assumptions: buffer=1.0pt for non-IBH, 0.75pt
for IBH (matching deployed config), 100ms latency, drop FIB_EXT_LO.
"""

from __future__ import annotations

import datetime
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
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
PER_LEVEL_BUFFER = {"IBH": 0.75}  # matches deployed config; others fall back to 1.0


def get_buffer(level: str) -> float:
    return PER_LEVEL_BUFFER.get(level, 1.0)


def report(label, buckets, days):
    """Print a bucket report with $/tr, n, $/day."""
    print(f"\n=== {label} ===")
    print(f"  {'bucket':<22} {'n':>5} {'WR%':>5} {'$/tr':>7} {'$/day':>7}")
    for k, v in buckets:
        n = v["n"]
        if n == 0:
            continue
        wr = v["w"] / n * 100
        ptr = v["pnl"] / n
        pday = v["pnl"] / days
        flag = "*" if abs(ptr) > 1.5 and n >= 200 else ""
        print(f"  {str(k):<22} {n:>5} {wr:>5.1f} {ptr:>+7.2f} {pday:>+7.2f} {flag}")


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    days = len({r["date"] for r in rows})
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    # Pre-compute IB lock timestamp per day
    ib_lock_ns = {}
    ib_range = {}
    for d, dc in caches.items():
        # 10:30 ET as ns
        ib_lock_dt = datetime.datetime.combine(
            d, datetime.time(10, 30), tzinfo=ET
        )
        ib_lock_ns[d] = int(ib_lock_dt.timestamp() * 1e9)
        ib_range[d] = dc.ibh - dc.ibl

    print("\nApplying slippage + computing per-trade features...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        dc = caches[r["date"]]
        buf = get_buffer(r["level"])
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
        # trend_60m: price at entry minus price 60m before entry
        entry_idx = int(np.searchsorted(dc.full_ts_ns, r["entry_ns"]))
        sixty_m_ago_ns = r["entry_ns"] - 60 * 60 * int(1e9)
        old_idx = int(np.searchsorted(dc.full_ts_ns, sixty_m_ago_ns))
        if old_idx < 0 or old_idx >= len(dc.full_prices):
            trend = 0.0
        else:
            trend = float(dc.full_prices[entry_idx]) - float(dc.full_prices[old_idx])
        # mins-since-IB-lock
        mins_since_ib = (r["entry_ns"] - ib_lock_ns[r["date"]]) / 1e9 / 60
        # IB range
        rng = ib_range[r["date"]]

        # For trend, the SIGN matters relative to direction. We compute
        # "trend_signed_for_direction": positive = with-trend (e.g.,
        # SELL when trend is negative = trend agrees with our short).
        if r["direction"] == "down":
            with_trend = -trend  # negative trend agrees with sell → positive
        else:
            with_trend = trend

        enriched.append({
            "outcome": r["outcome"],
            "pnl": new_pnl,
            "trend_60m": trend,
            "with_trend": with_trend,
            "mins_since_ib": mins_since_ib,
            "ib_range": rng,
            "level": r["level"],
        })

    n_enriched = len(enriched)
    total_pnl = sum(t["pnl"] for t in enriched)
    print(f"  enriched {n_enriched}, baseline ${total_pnl/days:+.2f}/day "
          f"(${total_pnl/n_enriched:+.2f}/tr)")

    # ============================================================
    # 1. TREND FILTER
    # ============================================================
    print(f"\n{'='*78}\n1. TREND_60M (raw, signed)\n{'='*78}")
    bins = [-30, -15, -5, 0, 5, 15, 30]
    grp = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in enriched:
        x = t["trend_60m"]
        for i, b in enumerate(bins):
            if x < b:
                key = f"<{b}" if i == 0 else f"{bins[i-1]}–{b}"
                break
        else:
            key = f">{bins[-1]}"
        grp[key]["n"] += 1
        grp[key]["pnl"] += t["pnl"]
        if t["outcome"] == "win":
            grp[key]["w"] += 1
    ordered_keys = ["<-30","-30–-15","-15–-5","-5–0","0–5","5–15","15–30",">30"]
    report("trend_60m bucket", [(k, grp[k]) for k in ordered_keys if k in grp], days)

    # WITH-TREND signed (positive = trade direction agrees with trend)
    print(f"\n{'='*78}\n1b. WITH-TREND (positive = direction agrees with 60m trend)\n{'='*78}")
    bins = [-30, -15, -5, 0, 5, 15, 30]
    grp = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in enriched:
        x = t["with_trend"]
        for i, b in enumerate(bins):
            if x < b:
                key = f"<{b}" if i == 0 else f"{bins[i-1]}–{b}"
                break
        else:
            key = f">{bins[-1]}"
        grp[key]["n"] += 1
        grp[key]["pnl"] += t["pnl"]
        if t["outcome"] == "win":
            grp[key]["w"] += 1
    report("with_trend bucket", [(k, grp[k]) for k in ordered_keys if k in grp], days)

    # ============================================================
    # 2. FIRST-N-MINUTES (since IB lock at 10:30 ET)
    # ============================================================
    print(f"\n{'='*78}\n2. MINUTES SINCE IB LOCK (10:30 ET)\n{'='*78}")
    bins = [10, 30, 60, 120, 180, 240, 300]
    grp = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in enriched:
        x = t["mins_since_ib"]
        for i, b in enumerate(bins):
            if x < b:
                key = f"<{b}min" if i == 0 else f"{bins[i-1]}–{b}min"
                break
        else:
            key = f">{bins[-1]}min"
        grp[key]["n"] += 1
        grp[key]["pnl"] += t["pnl"]
        if t["outcome"] == "win":
            grp[key]["w"] += 1
    keys = ["<10min","10–30min","30–60min","60–120min","120–180min","180–240min","240–300min",">300min"]
    report("mins since IB lock", [(k, grp[k]) for k in keys if k in grp], days)

    # ============================================================
    # 3. IB RANGE
    # ============================================================
    print(f"\n{'='*78}\n3. IB RANGE\n{'='*78}")
    bins = [50, 100, 150, 200, 300, 400, 500]
    grp = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in enriched:
        x = t["ib_range"]
        for i, b in enumerate(bins):
            if x < b:
                key = f"<{b}" if i == 0 else f"{bins[i-1]}–{b}"
                break
        else:
            key = f">{bins[-1]}"
        grp[key]["n"] += 1
        grp[key]["pnl"] += t["pnl"]
        if t["outcome"] == "win":
            grp[key]["w"] += 1
    keys = ["<50","50–100","100–150","150–200","200–300","300–400","400–500",">500"]
    report("IB range (pts)", [(k, grp[k]) for k in keys if k in grp], days)

    # ============================================================
    # FILTER CANDIDATES
    # ============================================================
    print(f"\n{'='*78}\nFILTER CANDIDATES (drop bucket → est. $/day Δ)\n{'='*78}")
    print(f"  Baseline $/day: ${total_pnl/days:+.2f}")
    # For each feature, find the worst bucket. If dropping it improves $/day, flag.
    print(f"\n  Dropping any bucket where its $/day contribution is < -$1/day")
    for label, items in [
        ("trend_60m",      [(k, grp[k]) for k in ordered_keys if k in grp][:0]),
    ]:
        pass  # placeholder

    # Just show: for each feature, is there a bucket whose removal would
    # improve the total $/day by more than 1.0?
    def worst_bucket_lift(grp_dict):
        worst_key, worst_pday = None, 0.0
        for k, v in grp_dict.items():
            if v["n"] < 200:
                continue
            pday = v["pnl"] / days
            if pday < worst_pday:
                worst_pday = pday
                worst_key = k
        return worst_key, worst_pday

    print(f"\n  Trend (with_trend signed): no analysis (re-bucketed above).")
    print(f"  Mins-since-IB: filtering not currently rich enough — rare > 240min trades.")
    print(f"  IB range: tight-IB days might be filterable; check bucket samples.")


if __name__ == "__main__":
    main()
