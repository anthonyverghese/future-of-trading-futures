"""Test whether a post-IB cooldown and/or scoring improves bot P&L.

Hypothesis: the first 30 minutes after IB locks (10:31-11:01 ET) are
choppy and produce whipsaw losses. Adding a cooldown or scoring filter
might improve results.

Variants tested:
  1. Baseline: current config (trade from 10:31, unscored)
  2. 30min cooldown: skip 10:31-11:01 ET
  3. 15min cooldown: skip 10:31-10:46 ET
  4. Scoring (min_score=0): use bot_entry_score, skip negative scores
  5. 30min cooldown + scoring (min_score=0)

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/test_post_ib_cooldown.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.bot_trader import bot_entry_score

PER_LEVEL_TS = {
    "IBH": (6, 20),
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.5": (10, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
}

# Scoring weights from bot_entry_score (same as live bot).
# We pass weights=None and handle scoring via min_score parameter
# since simulate_day uses the backtest scoring module, not bot_entry_score.
# For simplicity, we just use min_score with the default scoring in simulate_day.

VARIANTS = [
    {
        "name": "Baseline (current)",
        "extra_suppressed": None,
        "min_score": -99,
    },
    {
        "name": "15min cooldown (10:31-10:46)",
        "extra_suppressed": [(10 * 60 + 31, 10 * 60 + 46)],
        "min_score": -99,
    },
    {
        "name": "30min cooldown (10:31-11:01)",
        "extra_suppressed": [(10 * 60 + 31, 11 * 60 + 1)],
        "min_score": -99,
    },
    {
        "name": "45min cooldown (10:31-11:16)",
        "extra_suppressed": [(10 * 60 + 31, 11 * 60 + 16)],
        "min_score": -99,
    },
    {
        "name": "60min cooldown (10:31-11:31)",
        "extra_suppressed": [(10 * 60 + 31, 11 * 60 + 31)],
        "min_score": -99,
    },
]


def run_variant(dates, caches, arrays_cache, variant):
    total_pnl = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    winning_days = 0
    losing_days = 0
    streak = (0, 0)

    for date in dates:
        dc = caches[date]
        arrays = arrays_cache[date]

        trades, streak = simulate_day(
            dc, arrays,
            zone_factory=lambda name, price, drifts: BotZoneTradeReset(price, drifts),
            target_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[1],
            max_per_level=12,
            include_ibl=False,
            include_vwap=False,
            min_score=variant["min_score"],
            extra_suppressed=variant["extra_suppressed"],
        )

        day_pnl = sum(t.pnl_usd for t in trades)
        total_pnl += day_pnl
        total_trades += len(trades)
        for t in trades:
            if t.pnl_usd >= 0:
                wins += 1
            else:
                losses += 1
        if day_pnl >= 0:
            winning_days += 1
        else:
            losing_days += 1

    wr = wins / total_trades * 100 if total_trades > 0 else 0
    return {
        "name": variant["name"],
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "wr": wr,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(dates) if dates else 0,
        "winning_days": winning_days,
        "losing_days": losing_days,
        "trades_per_day": total_trades / len(dates) if dates else 0,
    }


def main():
    print("Loading data...")
    dates, caches = load_all_days()
    print(f"Loaded {len(dates)} days")

    print("Precomputing arrays...")
    arrays_cache = {}
    for date in dates:
        arrays_cache[date] = precompute_arrays(caches[date])
    print("Done.\n")

    results = []
    for v in VARIANTS:
        print(f"Running: {v['name']}...")
        r = run_variant(dates, caches, arrays_cache, v)
        results.append(r)
        print(f"  {r['trades']} trades, {r['wr']:.1f}% WR, ${r['avg_pnl']:.2f}/day")

    print()
    print("=" * 90)
    print(f"{'Variant':<35} {'Trades':>6} {'T/day':>5} {'WR%':>5} {'$/day':>8} {'Total $':>10} {'W days':>6} {'L days':>6}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['name']:<35} {r['trades']:>6} {r['trades_per_day']:>5.1f} "
            f"{r['wr']:>5.1f} ${r['avg_pnl']:>7.2f} ${r['total_pnl']:>9.2f} "
            f"{r['winning_days']:>6} {r['losing_days']:>6}"
        )

    # Highlight best
    best = max(results, key=lambda r: r["avg_pnl"])
    print()
    print(f"Best: {best['name']} at ${best['avg_pnl']:.2f}/day")


if __name__ == "__main__":
    main()
