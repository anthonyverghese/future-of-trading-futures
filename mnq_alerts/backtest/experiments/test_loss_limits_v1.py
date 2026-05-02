"""Test daily loss limit variants: fixed and variable.

Current: $100 fixed daily loss limit.
Tests: different fixed limits, and variable limits that tighten
after reaching a profit threshold (lock in gains).

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_loss_limits_v1.py
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

# (name, daily_loss, variable_config)
# variable_config: None for fixed, or "trailing" with (threshold, buffer)
# Trailing: once peak P&L >= threshold, floor = peak - buffer.
# Floor rises with peak but never drops.
VARIANTS = [
    # Fixed loss limits
    ("Loss limit $75", 75, None),
    ("Loss limit $100 (current)", 100, None),
    ("Loss limit $150", 150, None),
    ("Loss limit $200", 200, None),
    # Trailing floor: once profit >= threshold, floor = peak - buffer
    # $25 buffer (tight — lock in almost everything)
    ("$100 limit, trail $25 after $50", 100, ("trailing", 50, 25)),
    ("$100 limit, trail $25 after $75", 100, ("trailing", 75, 25)),
    ("$100 limit, trail $25 after $100", 100, ("trailing", 100, 25)),
    # $50 buffer (looser — room for one loss)
    ("$100 limit, trail $50 after $50", 100, ("trailing", 50, 50)),
    ("$100 limit, trail $50 after $75", 100, ("trailing", 75, 50)),
    # Higher initial limit + trailing
    ("$150 limit, trail $25 after $50", 150, ("trailing", 50, 25)),
    ("$150 limit, trail $50 after $75", 150, ("trailing", 75, 50)),
    ("$200 limit, trail $25 after $50", 200, ("trailing", 50, 25)),
]


def _run_one(args):
    name, daily_loss, variable_config = args
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
            daily_loss=999,  # disable built-in limit, we apply our own
        )

        kept = []
        cum_pnl = 0.0
        peak_pnl = 0.0
        current_floor = -daily_loss  # stop if cum_pnl <= floor

        for t in trades:
            if cum_pnl <= current_floor:
                break
            kept.append(t)
            cum_pnl += t.pnl_usd
            if cum_pnl > peak_pnl:
                peak_pnl = cum_pnl

            # Trailing floor: once peak >= threshold, floor = peak - buffer
            if variable_config is not None:
                _, threshold, buffer = variable_config
                if peak_pnl >= threshold:
                    trail_floor = peak_pnl - buffer
                    if trail_floor > current_floor:
                        current_floor = trail_floor

        all_trades.extend(kept)

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

    baseline = [r for r in results if "current" in r["name"].lower()][0]
    b_pnl = baseline["pnl_per_day"]

    print("=" * 145)
    print(f"{'Variant':<40} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 145)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<40} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<40} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"loss_limits_v1_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
