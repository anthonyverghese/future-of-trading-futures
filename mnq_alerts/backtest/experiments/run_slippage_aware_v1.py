"""Run real slippage-aware re-simulations for buffer=1.0 (target deploy)
and a buffer=1.0 + drop FIB_EXT_LO variant. Compare to:
  - the slippage-blind baseline (current backtest behavior, +$49.60/day)
  - the buffer-sweep post-process numbers (+$14.96/day at target/2,
    +$17.89/day at buffer=1.0)

The real re-sim differs from post-processing because:
  * Failed fills don't lock the bot's cap slot, so later zone
    triggers can take the slot instead.
  * Outcomes are computed from the actual fill_idx, not the original
    fire_idx.
  * Failed-fill cooldown (60s per level) bounds rapid retries.

Two configs run end-to-end:
  A. Deployed levels @ buffer=1.0pt (slippage-aware) — the target
  B. Same minus FIB_EXT_LO_1.272 — V1 redux at the new baseline
Plus a control:
  C. Deployed @ buffer=1.0pt with simulate_slippage=False — should
     match the +$49.60 slippage-blind baseline.
"""

from __future__ import annotations

import io
import os
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

TS = {
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}


def run(
    label: str,
    dates,
    caches,
    *,
    exclude_levels: set[str],
    simulate_slippage: bool,
    latency_ms: float = 100.0,
    entry_limit_buffer_pts_override: float | None = None,
) -> dict:
    sink = io.StringIO()
    total_pnl = 0.0
    total_trades = 0
    days_with_trades = 0
    by_level = defaultdict(lambda: [0, 0, 0.0])  # [n, wins, pnl]
    by_outcome = defaultdict(int)
    day_pnl = defaultdict(float)
    t0 = time.time()
    for di, date in enumerate(dates):
        dc = caches[date]
        arr = precompute_arrays(dc)
        caps = dict(BASE_CAPS)
        for lv in exclude_levels:
            caps.pop(lv, None)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        with redirect_stdout(sink):
            trades = simulate_day_v2(
                dc, arr,
                per_level_ts=TS, per_level_caps=caps,
                exclude_levels={"FIB_0.5", "IBL"} | exclude_levels,
                direction_filter={"IBH": "down"},
                daily_loss=200.0, momentum_max=0.0,
                simulate_slippage=simulate_slippage,
                latency_ms=latency_ms,
                entry_limit_buffer_pts_override=entry_limit_buffer_pts_override,
            )
        sink.truncate(0); sink.seek(0)
        d_pnl = sum(t.pnl_usd for t in trades)
        total_pnl += d_pnl
        total_trades += len(trades)
        if trades:
            days_with_trades += 1
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
            print(
                f"  [{label}] {di+1}/{len(dates)} | "
                f"{total_trades} trades, ${total_pnl/(di+1):+.2f}/day so far, "
                f"ETA {eta:.0f}s",
                flush=True,
            )
    days = len(dates)
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for d in sorted(day_pnl):
        cum += day_pnl[d]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return {
        "label": label,
        "trades": total_trades,
        "days": days,
        "pnl_total": total_pnl,
        "pnl_per_day": total_pnl / days,
        "max_dd": max_dd,
        "by_level": {k: tuple(v) for k, v in by_level.items()},
        "by_outcome": dict(by_outcome),
        "elapsed_secs": time.time() - t0,
    }


def fmt_result(r: dict) -> None:
    print(f"\n=== {r['label']} ===", flush=True)
    print(f"  Trades: {r['trades']}  ({r['trades']/r['days']:.1f}/day)")
    print(f"  Outcomes: {r['by_outcome']}")
    print(f"  P&L/day: ${r['pnl_per_day']:+.2f}")
    print(f"  MaxDD:   ${r['max_dd']:.0f}")
    print(f"  By level (n / wins / pnl_total):")
    for lv in sorted(r["by_level"]):
        n, w, p = r["by_level"][lv]
        wr = w / n * 100 if n else 0
        print(f"    {lv:<22} n={n:>5}  WR={wr:>5.1f}%  "
              f"$/tr=${p/n:+5.2f}  $tot=${p:+7.0f}  $/day=${p/r['days']:+5.2f}")


def main():
    print("Loading day caches (~3 min)...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    print("\n" + "=" * 70)
    print("CTRL: slippage-blind, buffer=1.0 (control — matches old baseline)")
    print("=" * 70)
    control = run("CTRL_no_slippage", dates, caches,
                  exclude_levels=set(), simulate_slippage=False)
    fmt_result(control)

    print("\n" + "=" * 70)
    print("A: target/2 buffer (current production), SLIPPAGE-AWARE")
    print("=" * 70)
    # entry_limit_buffer_pts_override=0.0 triggers the legacy target_pts/2
    # path in bot_trader, exactly matching the production behavior pre-buffer-change.
    a = run("A_target_half", dates, caches,
            exclude_levels=set(), simulate_slippage=True,
            entry_limit_buffer_pts_override=0.0)
    fmt_result(a)

    print("\n" + "=" * 70)
    print("B: buffer=1.0pt (proposed change), SLIPPAGE-AWARE")
    print("=" * 70)
    b = run("B_buffer1", dates, caches,
            exclude_levels=set(), simulate_slippage=True,
            entry_limit_buffer_pts_override=1.0)
    fmt_result(b)

    print("\n" + "=" * 70)
    print("C: buffer=1.0pt + drop FIB_EXT_LO_1.272, SLIPPAGE-AWARE")
    print("=" * 70)
    c = run("C_buffer1_drop_fib_ext_lo", dates, caches,
            exclude_levels={"FIB_EXT_LO_1.272"}, simulate_slippage=True,
            entry_limit_buffer_pts_override=1.0)
    fmt_result(c)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Slippage-blind control:               ${control['pnl_per_day']:+.2f}/day  MaxDD ${control['max_dd']:.0f}")
    print(f"A: target/2 (current production):     ${a['pnl_per_day']:+.2f}/day  MaxDD ${a['max_dd']:.0f}")
    print(f"B: buffer=1.0 (proposed):             ${b['pnl_per_day']:+.2f}/day  MaxDD ${b['max_dd']:.0f}")
    print(f"C: buffer=1.0 + drop FIB_EXT_LO:      ${c['pnl_per_day']:+.2f}/day  MaxDD ${c['max_dd']:.0f}")
    print()
    print(f"Slippage cost (A vs control):  ${a['pnl_per_day'] - control['pnl_per_day']:+.2f}/day")
    print(f"Buffer change effect (B - A):  ${b['pnl_per_day'] - a['pnl_per_day']:+.2f}/day")
    print(f"Drop FIB_EXT_LO (C - B):       ${c['pnl_per_day'] - b['pnl_per_day']:+.2f}/day")


if __name__ == "__main__":
    main()
