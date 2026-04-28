"""Test entry filters to reduce losing days and improve win rate.

Forks workers AFTER loading data so each child inherits the parent's
memory via copy-on-write (macOS/Linux). Only the simulation state
differs per worker, keeping memory usage low.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/test_filters.py
"""
import os, sys, time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset

PER_LEVEL_TS = {
    "IBH": (6, 20),
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.5": (10, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
}

# Data-driven per-level caps (from WR-by-entry-count analysis).
PER_LEVEL_CAPS = {
    "FIB_0.236": 12,
    "FIB_0.5": 4,
    "FIB_0.618": 3,
    "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 12,
    "FIB_EXT_LO_1.272": 6,
    "IBH": 7,
}

VARIANTS = [
    # Control
    {"name": "Baseline (current)", "kwargs": {}},
    # Individual filters (most promising)
    {"name": "Per-level caps", "kwargs": {"max_per_level_map": PER_LEVEL_CAPS}},
    {"name": "Per-level caps, no FIB_0.5", "kwargs": {
        "max_per_level_map": PER_LEVEL_CAPS, "exclude_levels": {"FIB_0.5"},
    }},
    {"name": "No repeat loss combo", "kwargs": {"no_repeat_loss_combo": True}},
    {"name": "Max 3 consec wins/level", "kwargs": {"max_wins_per_level": 3}},
    {"name": "5min level cooldown", "kwargs": {"level_cooldown_secs": 300}},
    # Combos
    {"name": "Caps + no-repeat + 5min cd", "kwargs": {
        "max_per_level_map": PER_LEVEL_CAPS, "no_repeat_loss_combo": True,
        "level_cooldown_secs": 300,
    }},
    {"name": "Caps + no FIB_0.5 + no-repeat", "kwargs": {
        "max_per_level_map": PER_LEVEL_CAPS, "exclude_levels": {"FIB_0.5"},
        "no_repeat_loss_combo": True,
    }},
    {"name": "Kitchen sink (no 0.5+nr+3w+5m)", "kwargs": {
        "max_per_level_map": PER_LEVEL_CAPS, "exclude_levels": {"FIB_0.5"},
        "no_repeat_loss_combo": True, "max_wins_per_level": 3,
        "level_cooldown_secs": 300,
    }},
]

# Loaded once in the parent; children inherit via fork COW.
_DATES = None
_CACHES = None
_ARRAYS = None


def _run_one(args):
    """Worker: simulate one variant using inherited global data."""
    name, kwargs = args
    total_pnl = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    winning_days = 0
    losing_days = 0
    days_below_neg100 = 0
    streak = (0, 0)

    for date in _DATES:
        trades, streak = simulate_day(
            _CACHES[date], _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[1],
            include_ibl=False,
            include_vwap=False,
            **kwargs,
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
            if day_pnl <= -100:
                days_below_neg100 += 1

    n = len(_DATES)
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    return {
        "name": name,
        "trades": total_trades,
        "tpd": total_trades / n if n else 0,
        "wr": wr,
        "avg_pnl": total_pnl / n if n else 0,
        "total_pnl": total_pnl,
        "w_days": winning_days,
        "l_days": losing_days,
        "l100_days": days_below_neg100,
        "w_day_pct": winning_days / n * 100 if n else 0,
    }


def main():
    global _DATES, _CACHES, _ARRAYS

    t0 = time.time()
    print("Loading data...", flush=True)
    _DATES, _CACHES = load_all_days()
    t1 = time.time()
    print(f"Loaded {len(_DATES)} days in {t1-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    _ARRAYS = {d: precompute_arrays(_CACHES[d]) for d in _DATES}
    t2 = time.time()
    print(f"Precomputed in {t2-t1:.0f}s", flush=True)

    n_variants = len(VARIANTS)
    # Fork after data is loaded — children inherit via COW.
    n_workers = min(3, n_variants)
    print(f"Running {n_variants} variants across {n_workers} workers...", flush=True)

    args = [(v["name"], v["kwargs"]) for v in VARIANTS]

    with Pool(n_workers) as pool:
        results = pool.map(_run_one, args)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 100)
    print(f"{'Variant':<38} {'Trades':>6} {'T/d':>4} {'WR%':>5} {'$/day':>7} {'W%days':>6} {'-$100d':>6} {'Total$':>9}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['name']:<38} {r['trades']:>6} {r['tpd']:>4.0f} "
            f"{r['wr']:>5.1f} {r['avg_pnl']:>+7.2f} "
            f"{r['w_day_pct']:>5.1f}% {r['l100_days']:>6} ${r['total_pnl']:>8.2f}"
        )

    best_pnl = max(results, key=lambda r: r["avg_pnl"])
    fewest_bad = min(results, key=lambda r: r["l100_days"])
    best_wr = max(results, key=lambda r: r["wr"])
    print()
    print(f"Best $/day:        {best_pnl['name']} at ${best_pnl['avg_pnl']:.2f}/day")
    print(f"Best WR:           {best_wr['name']} at {best_wr['wr']:.1f}%")
    print(f"Fewest -$100 days: {fewest_bad['name']} with {fewest_bad['l100_days']} days")


if __name__ == "__main__":
    main()
