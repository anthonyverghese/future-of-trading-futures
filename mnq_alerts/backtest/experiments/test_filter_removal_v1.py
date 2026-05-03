"""Test removing remaining defensive filters and varying position timeout.

Filters to test:
- Per-level caps: currently FIB_0.236=18, FIB_0.618=3, FIB_0.764=5, etc.
  What if we raise or remove them?
- Volatility filter: skip when 30m range < 0.15% of price. Does this help?
- Position timeout: currently 15 min. Try 10, 20, 30 min and no timeout.

Base config: $200 limit, no cooldown, no suppress, no adaptive caps,
momentum filter on (the best config from adaptive_caps_v2).

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_filter_removal_v1.py
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
CURRENT_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}
# Higher caps — double everything
DOUBLE_CAPS = {k: v * 2 for k, v in CURRENT_CAPS.items()}
# Very high caps — effectively uncapped
HIGH_CAPS = {k: 99 for k in CURRENT_CAPS}
# Loosen just the tight ones (FIB_0.618=3 and FIB_0.764=5 are very tight)
LOOSENED_CAPS = dict(CURRENT_CAPS)
LOOSENED_CAPS["FIB_0.618"] = 6
LOOSENED_CAPS["FIB_0.764"] = 10

BASE_EXCLUDE = {"FIB_0.5", "IBL"}

_DATES = None
_CACHES = None
_ARRAYS = None

# (name, caps, vol_filter_pct, timeout_secs)
VARIANTS = [
    # Baseline: current best config
    ("Baseline (current best)", CURRENT_CAPS, 0.0015, 900),

    # Per-level caps variations
    ("Double all caps", DOUBLE_CAPS, 0.0015, 900),
    ("Uncapped (99)", HIGH_CAPS, 0.0015, 900),
    ("Loosen 0.618=6, 0.764=10", LOOSENED_CAPS, 0.0015, 900),

    # Volatility filter removal
    ("No vol filter", CURRENT_CAPS, 0.0, 900),

    # Position timeout variations
    ("Timeout 10 min", CURRENT_CAPS, 0.0015, 600),
    ("Timeout 20 min", CURRENT_CAPS, 0.0015, 1200),
    ("Timeout 30 min", CURRENT_CAPS, 0.0015, 1800),
    ("Timeout 5 min", CURRENT_CAPS, 0.0015, 300),

    # Combos
    ("Loosen caps + no vol", LOOSENED_CAPS, 0.0, 900),
    ("Loosen caps + timeout 5m", LOOSENED_CAPS, 0.0015, 300),
    ("Loosen caps + no vol + 5m", LOOSENED_CAPS, 0.0, 300),
]


def _run_one(args):
    name, caps, vol_pct, timeout = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        dc = _CACHES[date]
        day_caps = dict(caps)
        if date.weekday() == 0:
            day_caps = {k: v * 2 for k, v in day_caps.items()}

        trades, streak = simulate_day(
            dc, _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=day_caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=0,
            direction_filter={"IBH": "down"},
            momentum_max=5.0,
            momentum_lookback_ticks=1000,
            daily_loss=200.0,
            suppress_1330=False,
            adaptive_caps=False,
            timeout_secs=timeout,
            vol_filter_pct=vol_pct,
        )
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

    print("=" * 145)
    print(f"{'Variant':<34} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 145)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<34} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<34} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"filter_removal_v1_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
