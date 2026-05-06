"""q3q4_diagnostic_2026_05_06.py — diagnose what specifically degrades in
Q3 and Q4 vs Q1 and Q2.

V0 baseline per-quarter: Q1 +$29.74, Q2 +$18.93, Q3 +$6.15, Q4 +$10.09
V6 (deployed): Q1 +$33.36, Q2 +$18.39, Q3 +$8.81, Q4 +$10.91

Q3+Q4 is 2-4x weaker than Q1+Q2. This script stratifies the 5,164-trade
pickle (with slippage applied + FIB_EXT_LO excluded per V6) to find:

  1. Per-level: which level's $/tr decayed most?
  2. Per-(level, direction): which combos broke?
  3. Per-hour-of-day: did certain bucket stop working?
  4. Per-entry-count-bucket: late-in-day entries failing?
  5. Per-day-of-week: DOW pattern shift?

Output identifies candidates for targeted fixes.
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
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272", "FIB_0.5"}  # V6/V0 deploy excludes
PER_LEVEL_BUFFER = {"IBH": 0.75}
MNQ_PT = 2.0


def _et_hour(ns):
    """Extract ET hour from ns timestamp."""
    import pytz
    et = pytz.timezone("America/New_York")
    return _dt.datetime.fromtimestamp(ns / 1e9, tz=et).hour


def _et_minute_bucket(ns):
    """30-min bucket label in ET."""
    import pytz
    et = pytz.timezone("America/New_York")
    dt = _dt.datetime.fromtimestamp(ns / 1e9, tz=et)
    bucket_start = (dt.minute // 30) * 30
    return f"{dt.hour:02d}:{bucket_start:02d}"


def main() -> None:
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    t0 = time.time()
    dates_all, caches = load_all_days()
    print(f"  caches in {time.time()-t0:.0f}s", flush=True)

    # Apply slippage and tag with quarter
    print("\nApplying slippage + tagging...", flush=True)
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
        enriched.append({
            "date": r["date"],
            "level": r["level"],
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pnl": new_pnl,
            "entry_ns": r["entry_ns"],
            "et_hour": _et_hour(r["entry_ns"]),
            "et_bucket": _et_minute_bucket(r["entry_ns"]),
            "weekday": r["date"].weekday(),
        })

    sorted_dates = sorted({e["date"] for e in enriched})
    n_days = len(sorted_dates)
    q_size = n_days // 4
    quarters = [
        ("Q1", set(sorted_dates[:q_size])),
        ("Q2", set(sorted_dates[q_size:2 * q_size])),
        ("Q3", set(sorted_dates[2 * q_size:3 * q_size])),
        ("Q4", set(sorted_dates[3 * q_size:])),
    ]
    quarter_days = {q_label: q_size if q_label != "Q4"
                    else len(sorted_dates) - 3 * q_size
                    for q_label, _ in quarters}

    n = len(enriched)
    days = len({e["date"] for e in enriched})
    print(f"  {n} trades over {days} days, "
          f"baseline ${sum(e['pnl'] for e in enriched)/days:+.2f}/day", flush=True)

    def by_q(items, key=lambda e: e["pnl"]):
        """Return dict {q: list of values for items in that quarter}."""
        out = defaultdict(list)
        for e in items:
            for q_label, qd in quarters:
                if e["date"] in qd:
                    out[q_label].append(key(e))
                    break
        return out

    # ==========================================================
    # 1. PER-LEVEL DECAY
    # ==========================================================
    print(f"\n{'='*94}\n1. PER-LEVEL Q1-Q4 PROGRESSION\n{'='*94}")
    print(f"  {'Level':<22} {'n':>5} {'Q1 $/tr':>9} {'Q2 $/tr':>9} {'Q3 $/tr':>9} "
          f"{'Q4 $/tr':>9} {'Q1 $/d':>8} {'Q4 $/d':>8} {'decay':>7}")
    by_level = defaultdict(list)
    for e in enriched:
        by_level[e["level"]].append(e)
    decay_ranking = []
    for lv in sorted(by_level.keys()):
        rows_lv = by_level[lv]
        per_q = {}
        per_q_pday = {}
        for q_label, qd in quarters:
            in_q = [e for e in rows_lv if e["date"] in qd]
            if in_q:
                per_q[q_label] = sum(e["pnl"] for e in in_q) / len(in_q)
                per_q_pday[q_label] = sum(e["pnl"] for e in in_q) / quarter_days[q_label]
            else:
                per_q[q_label] = 0
                per_q_pday[q_label] = 0
        decay = per_q["Q4"] - per_q["Q1"]
        decay_ranking.append((lv, decay, per_q_pday))
        print(f"  {lv:<22} {len(rows_lv):>5} ${per_q['Q1']:>+8.2f} "
              f"${per_q['Q2']:>+8.2f} ${per_q['Q3']:>+8.2f} ${per_q['Q4']:>+8.2f} "
              f"${per_q_pday['Q1']:>+7.2f} ${per_q_pday['Q4']:>+7.2f} "
              f"${decay:>+6.2f}")
    decay_ranking.sort(key=lambda x: x[1])  # most negative decay first
    print(f"\n  WORST decay (Q4-Q1 $/tr): {decay_ranking[0][0]} "
          f"({decay_ranking[0][1]:+.2f})")
    print(f"  BEST hold (Q4-Q1 $/tr): {decay_ranking[-1][0]} "
          f"({decay_ranking[-1][1]:+.2f})")

    # ==========================================================
    # 2. PER-(LEVEL, DIRECTION)
    # ==========================================================
    print(f"\n{'='*94}\n2. PER-(LEVEL, DIRECTION) Q1-Q4 PROGRESSION\n{'='*94}")
    print(f"  {'Level/Dir':<26} {'n':>5} {'Q1 $/tr':>9} {'Q2 $/tr':>9} "
          f"{'Q3 $/tr':>9} {'Q4 $/tr':>9} {'Q3+Q4 $/d':>11} {'decay':>7}")
    by_lvdir = defaultdict(list)
    for e in enriched:
        by_lvdir[(e["level"], e["direction"])].append(e)
    decay_lvdir = []
    for (lv, d) in sorted(by_lvdir.keys()):
        rows_lvd = by_lvdir[(lv, d)]
        per_q = {}
        for q_label, qd in quarters:
            in_q = [e for e in rows_lvd if e["date"] in qd]
            per_q[q_label] = sum(e["pnl"] for e in in_q) / len(in_q) if in_q else 0
        q34_total = sum(
            e["pnl"] for e in rows_lvd
            if any(e["date"] in q[1] for q in quarters[2:])
        )
        q34_days = quarter_days["Q3"] + quarter_days["Q4"]
        q34_pday = q34_total / q34_days
        decay = per_q["Q4"] - per_q["Q1"]
        decay_lvdir.append((f"{lv} {d}", decay, q34_pday, len(rows_lvd)))
        print(f"  {lv} {d:<7} {len(rows_lvd):>5} ${per_q['Q1']:>+8.2f} "
              f"${per_q['Q2']:>+8.2f} ${per_q['Q3']:>+8.2f} ${per_q['Q4']:>+8.2f} "
              f"${q34_pday:>+10.2f} ${decay:>+6.2f}")

    # Find combos that are NEGATIVE in Q3+Q4
    print(f"\n  (Level, direction) combos with NEGATIVE Q3+Q4 $/day:")
    for label, decay, q34_pday, n_lvd in sorted(decay_lvdir, key=lambda x: x[2]):
        if q34_pday < 0 and n_lvd >= 50:
            print(f"    {label}: ${q34_pday:+.2f}/day in Q3+Q4 (n={n_lvd}, decay {decay:+.2f})")

    # ==========================================================
    # 3. PER-HOUR-OF-DAY × QUARTER
    # ==========================================================
    print(f"\n{'='*94}\n3. PER-HOUR-OF-DAY (ET) Q1-Q4 PROGRESSION\n{'='*94}")
    print(f"  {'ET Hour':<8} {'n':>5} {'Q1 $/tr':>9} {'Q2 $/tr':>9} "
          f"{'Q3 $/tr':>9} {'Q4 $/tr':>9} {'Q3+Q4 $/d':>11}")
    by_hour = defaultdict(list)
    for e in enriched:
        by_hour[e["et_hour"]].append(e)
    for h in sorted(by_hour.keys()):
        rs = by_hour[h]
        per_q = {}
        for q_label, qd in quarters:
            in_q = [e for e in rs if e["date"] in qd]
            per_q[q_label] = sum(e["pnl"] for e in in_q) / len(in_q) if in_q else 0
        q34_total = sum(e["pnl"] for e in rs
                        if any(e["date"] in q[1] for q in quarters[2:]))
        q34_pday = q34_total / (quarter_days["Q3"] + quarter_days["Q4"])
        flag = " ⚠" if q34_pday < -0.5 else ""
        print(f"  {h:>2}:00    {len(rs):>5} ${per_q['Q1']:>+8.2f} "
              f"${per_q['Q2']:>+8.2f} ${per_q['Q3']:>+8.2f} ${per_q['Q4']:>+8.2f} "
              f"${q34_pday:>+10.2f}{flag}")

    # ==========================================================
    # 4. PER-DAY-OF-WEEK × QUARTER
    # ==========================================================
    print(f"\n{'='*94}\n4. PER-DAY-OF-WEEK Q1-Q4 PROGRESSION\n{'='*94}")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print(f"  {'DOW':<8} {'n':>5} {'Q1 $/tr':>9} {'Q2 $/tr':>9} "
          f"{'Q3 $/tr':>9} {'Q4 $/tr':>9} {'Q3+Q4 $/d':>11}")
    by_dow = defaultdict(list)
    for e in enriched:
        if e["weekday"] < 5:
            by_dow[e["weekday"]].append(e)
    for dow in sorted(by_dow.keys()):
        rs = by_dow[dow]
        per_q = {}
        for q_label, qd in quarters:
            in_q = [e for e in rs if e["date"] in qd]
            per_q[q_label] = sum(e["pnl"] for e in in_q) / len(in_q) if in_q else 0
        q34_total = sum(e["pnl"] for e in rs
                        if any(e["date"] in q[1] for q in quarters[2:]))
        q34_pday = q34_total / (quarter_days["Q3"] + quarter_days["Q4"])
        flag = " ⚠" if q34_pday < -0.5 else ""
        print(f"  {dow_names[dow]:<8} {len(rs):>5} ${per_q['Q1']:>+8.2f} "
              f"${per_q['Q2']:>+8.2f} ${per_q['Q3']:>+8.2f} ${per_q['Q4']:>+8.2f} "
              f"${q34_pday:>+10.2f}{flag}")

    # ==========================================================
    # 5. RECENT 60-DAY (truly out-of-sample for live)
    # ==========================================================
    print(f"\n{'='*94}\n5. PER-LEVEL RECENT 60-DAY (latest live regime)\n{'='*94}")
    recent_dates = set(sorted_dates[-60:])
    print(f"  {'Level':<22} {'n':>5} {'$/tr':>8} {'$/day':>8} {'WR%':>6}")
    by_lv_recent = defaultdict(list)
    for e in enriched:
        if e["date"] in recent_dates:
            by_lv_recent[e["level"]].append(e)
    for lv in sorted(by_lv_recent.keys()):
        rs = by_lv_recent[lv]
        wins = sum(1 for e in rs if e["outcome"] == "win")
        ptr = sum(e["pnl"] for e in rs) / len(rs)
        pday = sum(e["pnl"] for e in rs) / 60
        wr = wins / len(rs) * 100
        flag = " ⚠" if pday < 1.0 else ""
        print(f"  {lv:<22} {len(rs):>5} ${ptr:>+7.2f} ${pday:>+7.2f} {wr:>5.1f}%{flag}")

    # ==========================================================
    # 6. SUMMARY: top decay candidates for fixing
    # ==========================================================
    print(f"\n{'='*94}\nSUMMARY: top candidates for Q3+Q4 fixes\n{'='*94}")
    print(f"\n  Worst Q3+Q4 (level, direction) combos (n>=50):")
    bad_combos = [(l, d, q, n_) for l, d, q, n_ in decay_lvdir if q < 1.0 and n_ >= 50]
    for l, d, q, n_ in sorted(bad_combos, key=lambda x: x[2])[:5]:
        print(f"    {l}: Q3+Q4 ${q:+.2f}/day (n={n_})")

    print(f"\n  Top decay (Q4-Q1 $/tr) by level:")
    for lv, decay, _ in decay_ranking[:3]:
        print(f"    {lv}: decay ${decay:+.2f}/tr")

    print(f"\nDONE. Use these candidates to design real-sim variants.")


if __name__ == "__main__":
    main()
