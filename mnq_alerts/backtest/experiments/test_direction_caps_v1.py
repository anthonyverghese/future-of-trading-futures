"""Test direction caps + momentum filter with all levels.

Instead of excluding levels entirely (IBL, VWAP, FIB_0.5) or blocking
directions (IBH BUY), use per-direction caps to limit the weaker side
while keeping the stronger side. Combined with momentum filter which
blocks through-level entries.

Hypothesis: levels excluded for low WR might have value in one direction,
and the momentum filter handles the "blasting through" cases that drag
down WR. Per-direction caps limit remaining exposure on weak sides.

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_direction_caps_v1.py
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

# T/S for ALL levels (including previously excluded ones).
ALL_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
    "IBH": (6, 20), "IBL": (6, 20),
    "VWAP": (8, 25), "FIB_0.5": (10, 25),
}

_DATES = None
_CACHES = None
_ARRAYS = None

# Each variant is: (name, level_caps, direction_filter, direction_caps, exclude_levels,
#                    include_ibl, include_vwap, momentum_max)
VARIANTS = [
    # 0. Current deployed config (baseline)
    ("Current deployed", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
    }, {"IBH": "down"}, None, {"FIB_0.5", "IBL"}, False, False, 5.0),

    # 1. Current + unblock IBH BUY (was 81.1% WR, +$528)
    ("+ IBH BUY cap=3", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 10,
    }, None, {("IBH", "up"): 3, ("IBH", "down"): 7}, {"FIB_0.5", "IBL"}, False, False, 5.0),

    # 2. Current + add IBL (UP only, cap=3)
    ("+ IBL UP cap=3", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7, "IBL": 3,
    }, {"IBH": "down"}, {("IBL", "down"): 0}, {"FIB_0.5"}, True, False, 5.0),

    # 3. Current + add IBL (both dirs, cap=3 each)
    ("+ IBL cap=3/3", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7, "IBL": 6,
    }, {"IBH": "down"}, {("IBL", "up"): 3, ("IBL", "down"): 3}, {"FIB_0.5"}, True, False, 5.0),

    # 4. Current + add VWAP (UP only, cap=3)
    ("+ VWAP UP cap=3", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7, "VWAP": 3,
    }, {"IBH": "down"}, {("VWAP", "down"): 0}, {"FIB_0.5", "IBL"}, False, True, 5.0),

    # 5. Current + add VWAP (both dirs, cap=3 each)
    ("+ VWAP cap=3/3", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7, "VWAP": 6,
    }, {"IBH": "down"}, {("VWAP", "up"): 3, ("VWAP", "down"): 3}, {"FIB_0.5", "IBL"}, False, True, 5.0),

    # 6. All 9 levels, conservative direction caps
    ("All 9 levels conservative", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
        "IBH": 10, "IBL": 4, "VWAP": 4, "FIB_0.5": 3,
    }, None, {
        ("IBH", "up"): 3, ("IBH", "down"): 7,
        ("IBL", "up"): 3, ("IBL", "down"): 1,
        ("VWAP", "up"): 3, ("VWAP", "down"): 1,
        ("FIB_0.5", "up"): 0, ("FIB_0.5", "down"): 3,
    }, set(), True, True, 5.0),

    # 7. All 9 levels, aggressive (higher caps)
    ("All 9 levels aggressive", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
        "IBH": 10, "IBL": 6, "VWAP": 6, "FIB_0.5": 5,
    }, None, {
        ("IBH", "up"): 5, ("IBH", "down"): 7,
        ("IBL", "up"): 5, ("IBL", "down"): 3,
        ("VWAP", "up"): 5, ("VWAP", "down"): 3,
        ("FIB_0.5", "up"): 1, ("FIB_0.5", "down"): 5,
    }, set(), True, True, 5.0),

    # 8. All 9 levels, no direction caps (just momentum filter)
    ("All 9 levels no dir caps", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
        "IBH": 7, "IBL": 7, "VWAP": 12, "FIB_0.5": 5,
    }, None, None, set(), True, True, 5.0),

    # 9. IBH BUY + IBL UP + VWAP UP (best individual adds)
    ("IBH BUY+IBL UP+VWAP UP", {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
        "IBH": 10, "IBL": 3, "VWAP": 3,
    }, None, {
        ("IBH", "up"): 3, ("IBH", "down"): 7,
        ("IBL", "down"): 0,
        ("VWAP", "down"): 0,
    }, {"FIB_0.5"}, True, True, 5.0),
]


def _run_one(args):
    name, level_caps, dir_filter, dir_caps, exclude, inc_ibl, inc_vwap, mom = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        dc = _CACHES[date]
        caps = dict(level_caps)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        # Monday double for direction caps too
        dcaps = None
        if dir_caps:
            dcaps = dict(dir_caps)
            if date.weekday() == 0:
                dcaps = {k: v * 2 for k, v in dcaps.items()}

        trades, streak = simulate_day(
            dc, _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: ALL_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: ALL_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=exclude,
            include_ibl=inc_ibl, include_vwap=inc_vwap,
            global_cooldown_after_loss_secs=30,
            direction_filter=dir_filter,
            direction_caps=dcaps,
            momentum_max=mom,
            momentum_lookback_ticks=1000,
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

    # Per-level breakdown for the best variant
    print()
    best = max(results, key=lambda r: r["pnl_per_day"])
    print(f"Best variant: {best['name']}")
    if "per_level" in best:
        print(f"  {'Level':<20} {'Trades':>7} {'WR%':>6} {'$/day':>7}")
        for lv, stats in sorted(best["per_level"].items()):
            print(f"  {lv:<20} {stats['trades']:>7} {stats['wr']:>5.1f}% {stats['pnl_per_day']:>+7.2f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"direction_caps_v1_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
