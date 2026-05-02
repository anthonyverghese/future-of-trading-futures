"""Test profit-stop variants: stop trading after hitting a daily profit target.

Hypothesis: on good days, the bot makes most of its money early and then
gives some back. Stopping after hitting a profit target could lock in gains.
Counter-hypothesis: good days keep getting better, and capping upside
reduces total P&L.

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_profit_stop_v1.py
"""
import os, sys, time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.results import compute_stats

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}
BASE_EXCLUDE = {"FIB_0.5", "IBL"}

_DATES = None
_CACHES = None
_ARRAYS = None

# (name, profit_stop)  — 0 means no profit stop
VARIANTS = [
    ("Baseline (no profit stop)", 0),
    ("Profit stop $25", 25),
    ("Profit stop $50", 50),
    ("Profit stop $75", 75),
    ("Profit stop $100", 100),
    ("Profit stop $125", 125),
    ("Profit stop $150", 150),
    ("Profit stop $200", 200),
]


def _run_one(args):
    name, profit_stop = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        dc = _CACHES[date]
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        trades, streak = simulate_day(
            dc, _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=30,
            direction_filter={"IBH": "down"},
            momentum_max=5.0,
            momentum_lookback_ticks=1000,
        )

        if profit_stop > 0:
            # Post-filter: keep trades until cumulative P&L hits the target.
            # Once hit, stop taking trades for the rest of the day.
            kept = []
            cum_pnl = 0.0
            for t in trades:
                if cum_pnl >= profit_stop:
                    break
                kept.append(t)
                cum_pnl += t.pnl_usd
            all_trades.extend(kept)
        else:
            all_trades.extend(trades)

    stats = compute_stats(all_trades, len(_DATES), list(_DATES))
    stats["name"] = name
    return stats


def main():
    global _DATES, _CACHES, _ARRAYS
    t0 = time.time()

    print("Loading data...", flush=True)
    _DATES, _CACHES = load_all_days()
    print(f"Loaded {len(_DATES)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    _ARRAYS = {d: precompute_arrays(_CACHES[d]) for d in _DATES}

    n_variants = len(VARIANTS)
    print(f"Running {n_variants} variants across 3 workers...", flush=True)

    with Pool(3) as pool:
        results = pool.map(_run_one, VARIANTS)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    baseline = results[0]
    b_pnl = baseline["pnl_per_day"]

    print("=" * 140)
    print(f"{'Variant':<32} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 140)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<32} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<32} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"profit_stop_v1_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
