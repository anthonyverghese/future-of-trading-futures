"""Real-sim Phase 1: add prior day's RTH close as a "gap" magnet level.

Hypothesis: yesterday's RTH close acts as a price magnet — order flow
memory pulls today's price toward it. If true, approaches to the gap
level should reverse (bounce) more often than approaches to math-
derived fib lines, giving the bot a NEW kind of edge that the prior
exhaustive testing didn't capture (all our prior added-levels were
math-derived).

Phase 1: just add the gap level. T8/S25 (matches FIB_0.236), cap=6,
buffer=1.0pt (default fallback in dict). If WR is at the structural
ceiling (~78%), $/tr should be ~+$1-2 with 1-3 trades/day = +$0.50-2/day.
If WR exceeds ceiling because magnet hypothesis is real, lift could
be larger.

Phase 2 (only if Phase 1 ≥ +$1/day): cap reduction on nearby existing
levels. Per user hypothesis: when gap is within X pts of IBH/etc.,
price often penetrates the closer level on its way to the gap, so
those nearby trades are worse and should be capped down.

Configs:
  baseline    — C + IBH=0.75 (current deployed)
  gap_level   — adds GAP_PRIOR_CLOSE level alongside existing levels

Caps are per-level and capped at 6 (matches FIB_EXT_HI). T/S is T8/S25.
"""

from __future__ import annotations

import datetime
import io
import os
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

PER_LEVEL_BUFFER = {"IBH": 0.75}

TS_BASE = {
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
CAPS_BASE = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}


def compute_gap_closes(dates, caches) -> dict:
    """For each date, find the prior trading day's last full_prices value."""
    gap = {}
    for i, d in enumerate(dates):
        if i == 0:
            continue
        prior = dates[i - 1]
        prior_dc = caches[prior]
        if len(prior_dc.full_prices) == 0:
            continue
        gap[d] = float(prior_dc.full_prices[-1])
    return gap


def run_config(label, dates, caches, *, include_gap, gap_closes,
               per_level_buffer):
    sink = io.StringIO()
    total_pnl = 0.0
    total_trades = 0
    by_level = defaultdict(lambda: [0, 0, 0.0])  # n, wins, pnl
    by_outcome = defaultdict(int)
    day_pnl = defaultdict(float)
    t0 = time.time()
    for di, date in enumerate(dates):
        dc = caches[date]
        arr = precompute_arrays(dc)
        ts = dict(TS_BASE)
        caps = dict(CAPS_BASE)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        gap_close = None
        if include_gap and date in gap_closes:
            gap_close = gap_closes[date]
            ts["GAP_PRIOR_CLOSE"] = (8, 25)
            caps["GAP_PRIOR_CLOSE"] = 12 if date.weekday() == 0 else 6
        with redirect_stdout(sink):
            trades = simulate_day_v2(
                dc, arr,
                per_level_ts=ts, per_level_caps=caps,
                exclude_levels={"FIB_0.5", "IBL", "FIB_EXT_LO_1.272"},
                direction_filter={"IBH": "down"},
                daily_loss=200.0, momentum_max=0.0,
                simulate_slippage=True,
                latency_ms=100.0,
                entry_limit_buffer_pts_override=per_level_buffer,
                gap_close=gap_close,
            )
        sink.truncate(0); sink.seek(0)
        d_pnl = sum(t.pnl_usd for t in trades)
        total_pnl += d_pnl
        total_trades += len(trades)
        day_pnl[date] = d_pnl
        for t in trades:
            s = by_level[t.level]
            s[0] += 1
            if t.pnl_usd >= 0: s[1] += 1
            s[2] += t.pnl_usd
            by_outcome[t.outcome] += 1
        if (di + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = (len(dates) - di - 1) * (elapsed / (di + 1))
            print(f"  [{label}] {di+1}/{len(dates)} | "
                  f"{total_trades} trades, ${total_pnl/(di+1):+.2f}/day so far, "
                  f"ETA {eta:.0f}s", flush=True)
    days = len(dates)
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for d in sorted(day_pnl):
        cum += day_pnl[d]
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd
    return {
        "label": label,
        "trades": total_trades,
        "days": days,
        "pnl_total": total_pnl,
        "pnl_per_day": total_pnl / days,
        "max_dd": max_dd,
        "by_level": dict(by_level),
        "by_outcome": dict(by_outcome),
    }


def fmt(r):
    print(f"\n=== {r['label']} ===", flush=True)
    print(f"  Trades: {r['trades']}  ({r['trades']/r['days']:.1f}/day)")
    print(f"  Outcomes: {r['by_outcome']}")
    print(f"  P&L/day: ${r['pnl_per_day']:+.2f}")
    print(f"  MaxDD:   ${r['max_dd']:.0f}")
    print(f"  By level (n / wins / pnl_total):")
    for lv in sorted(r["by_level"]):
        n, w, p = r["by_level"][lv]
        wr = w / n * 100 if n else 0
        print(f"    {lv:<22} n={n:>5}  WR={wr:>5.1f}%  $/tr=${p/n:+5.2f}  "
              f"$tot=${p:+7.0f}  $/day=${p/r['days']:+5.2f}")


def main():
    print("Loading day caches...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    print("\nComputing gap closes (prior day's last RTH tick)...", flush=True)
    gap_closes = compute_gap_closes(dates, caches)
    print(f"  {len(gap_closes)} gap closes (1 missing for first day)", flush=True)
    # Quick distribution check
    sample = sorted(gap_closes.items())[:3] + sorted(gap_closes.items())[-3:]
    for d, g in sample:
        ibh = caches[d].ibh; ibl = caches[d].ibl
        rng = ibh - ibl
        gap_in_range = ibl <= g <= ibh
        gap_to_ibh = abs(g - ibh)
        gap_to_ibl = abs(g - ibl)
        print(f"    {d}: gap={g:.2f}  IBH={ibh:.2f}  IBL={ibl:.2f}  "
              f"in_IB={gap_in_range}  dist_to_IBH={gap_to_ibh:.1f}  "
              f"dist_to_IBL={gap_to_ibl:.1f}")

    print("\n" + "="*70)
    print("BASELINE: C + IBH=0.75 (no gap level)")
    print("="*70)
    base = run_config("baseline", dates, caches,
                      include_gap=False, gap_closes=gap_closes,
                      per_level_buffer=PER_LEVEL_BUFFER)
    fmt(base)

    print("\n" + "="*70)
    print("VARIANT: C + IBH=0.75 + GAP level (T8/S25, cap=6)")
    print("="*70)
    gap = run_config("with_gap", dates, caches,
                     include_gap=True, gap_closes=gap_closes,
                     per_level_buffer=PER_LEVEL_BUFFER)
    fmt(gap)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  baseline         ${base['pnl_per_day']:+.2f}/day  "
          f"MaxDD ${base['max_dd']:.0f}  trades {base['trades']}")
    print(f"  with_gap         ${gap['pnl_per_day']:+.2f}/day  "
          f"MaxDD ${gap['max_dd']:.0f}  trades {gap['trades']}")
    print(f"  Δ:               ${gap['pnl_per_day'] - base['pnl_per_day']:+.2f}/day  "
          f"MaxDD ${gap['max_dd'] - base['max_dd']:+.0f}  "
          f"trades {gap['trades'] - base['trades']:+d}")
    # Highlight gap level if present
    gap_stats = gap["by_level"].get("GAP_PRIOR_CLOSE")
    if gap_stats:
        n, w, p = gap_stats
        if n > 0:
            wr = w / n * 100
            print(f"\n  GAP_PRIOR_CLOSE: n={n}  WR={wr:.1f}%  $/tr=${p/n:+.2f}  "
                  f"$/day=${p/gap['days']:+.2f}")


if __name__ == "__main__":
    main()
