"""Per-level direction analysis under slippage modeling at buffer=1.0pt.

Tests the hypothesis: are some levels meaningfully biased toward one
direction? IBH SELL-only is already deployed (memory: IBH BUY cap=3
was -$2.59/day). What about other levels?

For each (level, direction) bucket: WR, $/tr, total $, contribution
to $/day. Flag candidates where one direction is materially worse.

Filters applied to match deployed C config:
  - buffer=1.0pt slippage modeled
  - exclude FIB_EXT_LO_1.272
  - drop trades that wouldn't fill
"""

from __future__ import annotations

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import (
    simulate_fill, adjusted_pnl,
)

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}


def main() -> None:
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    days = len({r["date"] for r in rows})
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(rows)} trades over {days} days, "
          f"caches loaded in {time.time()-t0:.0f}s", flush=True)

    # Apply slippage; collect (level, direction, slippage-adjusted pnl, outcome)
    print("\nApplying slippage model (buffer=1.0pt, latency=100ms)...", flush=True)
    enriched = []
    for r in rows:
        if r["level"] in EXCLUDED_LEVELS:
            continue
        # Note: IBH already filtered to SELL-only by deployed config.
        # For the analysis we want to see what happens *if we allowed both
        # directions*. The pickle was generated with IBH down-only, so
        # IBH BUY trades won't appear. We can still see all 4 other
        # levels in both directions.
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
        enriched.append((r["level"], r["direction"], new_pnl, r["outcome"]))

    # Group and report
    from collections import defaultdict
    grp = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "to": 0, "pnl": 0.0})
    for level, direction, pnl, outcome in enriched:
        s = grp[(level, direction)]
        s["n"] += 1
        s["pnl"] += pnl
        if outcome == "win":
            s["w"] += 1
        elif outcome == "loss":
            s["l"] += 1
        else:
            s["to"] += 1

    levels = sorted({k[0] for k in grp.keys()})
    print(f"\n{'='*82}")
    print("PER-LEVEL × DIRECTION (slippage-modeled, buffer=1.0pt)")
    print('='*82)
    print(f"  {'Level':<22} {'Dir':<5} {'N':>5} {'W':>4} {'L':>4} "
          f"{'WR%':>5} {'$/tr':>7} {'$tot':>8} {'$/day':>7}")

    candidates = []  # (delta_per_day_if_dropped, level, direction, current_pnl_per_day)
    for lv in levels:
        for d in ["up", "down"]:
            s = grp.get((lv, d))
            if s is None:
                continue
            n = s["n"]
            wr = s["w"] / n * 100 if n else 0
            ptr = s["pnl"] / n if n else 0
            pday = s["pnl"] / days
            print(f"  {lv:<22} {d:<5} {n:>5} {s['w']:>4} {s['l']:>4} "
                  f"{wr:>5.1f} {ptr:>+7.2f} {s['pnl']:>+8.0f} {pday:>+7.2f}")
            # If dropping this direction would improve the level's contribution,
            # mark it. Threshold: $/tr negative OR $/day < $0.50 (low EV)
            if pday < 0:
                candidates.append((pday, lv, d, "negative — drop?"))
            elif pday < 0.5 and ptr < 0.5:
                candidates.append((pday, lv, d, "marginal — consider"))
        # Show level totals
        s_up = grp.get((lv, "up"), {"n": 0, "pnl": 0})
        s_dn = grp.get((lv, "down"), {"n": 0, "pnl": 0})
        total_n = s_up["n"] + s_dn["n"]
        total_pnl = s_up["pnl"] + s_dn["pnl"]
        if total_n > 0:
            print(f"  {lv:<22} {'TOT':<5} {total_n:>5} {' '*15} "
                  f"{total_pnl/total_n:>+7.2f} {total_pnl:>+8.0f} "
                  f"{total_pnl/days:>+7.2f}")
        print()

    print(f"\n{'='*82}")
    print("DIRECTION-FILTER CANDIDATES")
    print('='*82)
    if not candidates:
        print("  No (level, direction) combo has $/day < $0.50 and $/tr < $0.50.")
        print("  Direction filtering is unlikely to help.")
    else:
        print("  Sorted by impact (worst first):")
        candidates.sort()
        for pday, lv, d, note in candidates:
            print(f"    {lv:<22} {d:<5} ${pday:+.2f}/day — {note}")
        # Compute the "drop-the-bad-side" total impact
        total_save = sum(-c[0] for c in candidates if c[0] < 0)
        if total_save > 0:
            print(f"\n  If all negative-$/day directions were dropped: "
                  f"+${total_save:.2f}/day (rough estimate, ignores cap-budget)")


if __name__ == "__main__":
    main()
