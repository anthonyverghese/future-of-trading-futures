"""Per-level edge under slippage modeling at buffer=1.0pt vs target/2.

Quick post-process: for each trade in the existing pickle, simulate the
fill at both buffers, recompute P&L using the simulated fill price
(outcome unchanged from original sim), then aggregate per level.

Reveals which levels' edge changes most when buffer goes from target/2
to 1.0pt — and whether any level becomes uneconomic under either.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import (
    simulate_fill, adjusted_pnl, MULTIPLIER, FEE_USD,
)

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)


def per_level(rows, caches, buffer_spec, latency_ms=100.0):
    by_level = defaultdict(lambda: {
        "n_orig": 0, "n_filled": 0, "wins": 0, "losses": 0,
        "timeouts": 0, "pnl": 0.0,
    })
    for r in rows:
        lv = r["level"]
        by_level[lv]["n_orig"] += 1
        dc = caches[r["date"]]
        if isinstance(buffer_spec, str):
            t = r["target_pts"]
            buf = t / 2.0 if buffer_spec == "target/2" else t / 4.0
        else:
            buf = float(buffer_spec)
        buf = round(buf * 4) / 4
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=buf, latency_ms=latency_ms,
        )
        if fill_idx is None:
            continue
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        s = by_level[lv]
        s["n_filled"] += 1
        s["pnl"] += new_pnl
        if r["outcome"] == "win":
            s["wins"] += 1
        elif r["outcome"] == "loss":
            s["losses"] += 1
        else:
            s["timeouts"] += 1
    return by_level


def fmt_table(stats, days, label):
    print(f"\n=== {label} ===", flush=True)
    print(
        f"  {'Level':<22} {'N_orig':>6} {'N_fill':>6} {'Fill%':>5} "
        f"{'WR%':>5} {'$/tr':>7} {'$/day':>8} {'$tot':>7}",
        flush=True,
    )
    rows_out = []
    for lv, s in stats.items():
        n_orig, n_fill = s["n_orig"], s["n_filled"]
        if n_fill == 0:
            continue
        wr = s["wins"] / n_fill * 100
        fill_rate = n_fill / n_orig * 100 if n_orig else 0
        rows_out.append((lv, n_orig, n_fill, fill_rate, wr,
                         s["pnl"] / n_fill, s["pnl"] / days, s["pnl"]))
    rows_out.sort(key=lambda x: -x[6])  # sort by $/day desc
    for lv, n_o, n_f, fr, wr, ptr, pday, ptot in rows_out:
        print(
            f"  {lv:<22} {n_o:>6d} {n_f:>6d} {fr:>5.1f} "
            f"{wr:>5.1f} {ptr:>+7.2f} {pday:>+8.2f} {ptot:>+7.0f}",
            flush=True,
        )
    total = sum(r[7] for r in rows_out)
    total_fills = sum(r[2] for r in rows_out)
    print(
        f"  {'TOTAL':<22} {sum(r[1] for r in rows_out):>6d} {total_fills:>6d} "
        f"{'':<5} {'':<5} {'':<7} {total/days:>+8.2f} {total:>+7.0f}",
        flush=True,
    )


def main():
    print("Loading pickle + caches...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    days = len({r["date"] for r in rows})
    print(f"  {len(rows)} trades over {days} days", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  caches loaded in {time.time()-t0:.0f}s", flush=True)

    print("\nComputing per-level edge under slippage...", flush=True)
    cur = per_level(rows, caches, "target/2")
    new = per_level(rows, caches, 1.0)

    fmt_table(cur, days, "CURRENT: target/2 buffer (slippage-modeled)")
    fmt_table(new, days, "PROPOSED: buffer=1.0pt (slippage-modeled)")

    print("\n=== DELTA: buffer=1.0 vs target/2 ===", flush=True)
    print(
        f"  {'Level':<22} {'N_fill_Δ':>9} {'$/tr_Δ':>7} {'$/day_Δ':>9} {'$tot_Δ':>8}",
        flush=True,
    )
    levels = sorted(set(cur.keys()) | set(new.keys()))
    for lv in levels:
        c, n = cur.get(lv, {}), new.get(lv, {})
        c_fill = c.get("n_filled", 0); n_fill = n.get("n_filled", 0)
        c_pnl = c.get("pnl", 0.0); n_pnl = n.get("pnl", 0.0)
        c_ptr = c_pnl / c_fill if c_fill else 0
        n_ptr = n_pnl / n_fill if n_fill else 0
        delta_pnl = n_pnl - c_pnl
        print(
            f"  {lv:<22} {n_fill - c_fill:>+9d} {n_ptr - c_ptr:>+7.2f} "
            f"{delta_pnl/days:>+9.2f} {delta_pnl:>+8.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
