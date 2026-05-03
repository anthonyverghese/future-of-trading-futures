"""Test higher loss limits + removing defensive filters that may not make
sense with more room to recover.

Defensive filters designed for $100 loss budget:
- Adaptive caps: halves caps first 30 min after IB
- 30s global cooldown after loss
- 13:30-14:00 suppression (weak time window)

These might be counterproductive at $250-350 because they prevent
recovery trades that the higher limit would allow.

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_loss_limits_v2.py
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

# (name, daily_loss, cooldown_secs, suppress_1330, momentum_max)
# adaptive caps are handled in post-filter since simulate_day doesn't have that param
VARIANTS = [
    # Baselines at different loss limits
    ("$100 limit (old baseline)", 100, 30, True, 5.0),
    ("$200 limit (current)", 200, 30, True, 5.0),
    ("$250 limit", 250, 30, True, 5.0),
    ("$300 limit", 300, 30, True, 5.0),
    ("$350 limit", 350, 30, True, 5.0),
    # $250 with filters removed
    ("$250, no cooldown", 250, 0, True, 5.0),
    ("$250, no 13:30 suppress", 250, 30, False, 5.0),
    ("$250, no cooldown + no suppress", 250, 0, False, 5.0),
    # $300 with filters removed
    ("$300, no cooldown", 300, 0, True, 5.0),
    ("$300, no 13:30 suppress", 300, 30, False, 5.0),
    ("$300, no cooldown + no suppress", 300, 0, False, 5.0),
    # $300, no momentum (see if momentum still helps at higher limits)
    ("$300, no momentum", 300, 30, True, 0.0),
    # $300, all filters removed except momentum
    ("$300, only momentum filter", 300, 0, False, 5.0),
]


def _run_one(args):
    name, daily_loss, cooldown_secs, suppress_1330, momentum_max = args
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
            global_cooldown_after_loss_secs=cooldown_secs,
            direction_filter={"IBH": "down"},
            momentum_max=momentum_max,
            momentum_lookback_ticks=1000,
            daily_loss=float(daily_loss),
            suppress_1330=suppress_1330,
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

    baseline = results[1]  # $200 current
    b_pnl = baseline["pnl_per_day"]

    print("=" * 145)
    print(f"{'Variant':<38} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$200d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs $200':>7}")
    print("-" * 145)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<38} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<38} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"loss_limits_v2_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
