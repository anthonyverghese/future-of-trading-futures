"""Test per-level direction filters (BUY only / SELL only).

Tests whether restricting to one direction per level improves P&L,
including adding back excluded levels with direction filtering.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_direction_filter.py
"""
import os, sys, time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.results import compute_stats

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
    # T/S for excluded levels (use defaults from prior config)
    "IBH": (6, 20), "IBL": (6, 20), "FIB_0.5": (10, 25),
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
    "IBH": 7, "IBL": 6, "FIB_0.5": 4,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH", "IBL"}

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    # Baseline
    ("Baseline", {"exclude": BASE_EXCLUDE, "dir_filter": None, "ibl": False}),

    # Phase 1: Add back excluded levels (both directions)
    ("+IBH both", {"exclude": BASE_EXCLUDE - {"IBH"}, "dir_filter": None, "ibl": False}),
    ("+IBL both", {"exclude": BASE_EXCLUDE - {"IBL"}, "dir_filter": None, "ibl": True}),
    ("+FIB_0.5 both", {"exclude": BASE_EXCLUDE - {"FIB_0.5"}, "dir_filter": None, "ibl": False}),

    # Phase 2: Excluded levels with direction filter
    ("+IBH BUY only", {"exclude": BASE_EXCLUDE - {"IBH"}, "dir_filter": {"IBH": "up"}, "ibl": False}),
    ("+IBH SELL only", {"exclude": BASE_EXCLUDE - {"IBH"}, "dir_filter": {"IBH": "down"}, "ibl": False}),
    ("+IBL BUY only", {"exclude": BASE_EXCLUDE - {"IBL"}, "dir_filter": {"IBL": "up"}, "ibl": True}),
    ("+IBL SELL only", {"exclude": BASE_EXCLUDE - {"IBL"}, "dir_filter": {"IBL": "down"}, "ibl": True}),
    ("+FIB_0.5 BUY only", {"exclude": BASE_EXCLUDE - {"FIB_0.5"}, "dir_filter": {"FIB_0.5": "up"}, "ibl": False}),
    ("+FIB_0.5 SELL only", {"exclude": BASE_EXCLUDE - {"FIB_0.5"}, "dir_filter": {"FIB_0.5": "down"}, "ibl": False}),

    # Phase 3: Active levels direction filter
    ("0.236 BUY only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_0.236": "up"}, "ibl": False}),
    ("0.236 SELL only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_0.236": "down"}, "ibl": False}),
    ("0.618 BUY only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_0.618": "up"}, "ibl": False}),
    ("0.618 SELL only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_0.618": "down"}, "ibl": False}),
    ("0.764 BUY only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_0.764": "up"}, "ibl": False}),
    ("0.764 SELL only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_0.764": "down"}, "ibl": False}),
    ("EXT_HI BUY only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_EXT_HI_1.272": "up"}, "ibl": False}),
    ("EXT_HI SELL only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_EXT_HI_1.272": "down"}, "ibl": False}),
    ("EXT_LO BUY only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_EXT_LO_1.272": "up"}, "ibl": False}),
    ("EXT_LO SELL only", {"exclude": BASE_EXCLUDE, "dir_filter": {"FIB_EXT_LO_1.272": "down"}, "ibl": False}),
]


def _run_one(args):
    name, cfg = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        caps = dict(BASE_CAPS)
        # Monday double caps (deployed)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        trades, streak = simulate_day(
            _CACHES[date], _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=cfg["exclude"],
            include_ibl=cfg.get("ibl", False),
            include_vwap=False,
            direction_filter=cfg["dir_filter"],
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

    print("=" * 115)
    print(f"{'Variant':<25} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'vs base':>7}")
    print("-" * 115)

    sections = [
        ("BASELINE", [0]),
        ("ADD BACK EXCLUDED (both dir)", [1, 2, 3]),
        ("EXCLUDED LEVELS (dir filter)", [4, 5, 6, 7, 8, 9]),
        ("ACTIVE LEVELS (dir filter)", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    ]
    for section_name, indices in sections:
        print(f"\n  --- {section_name} ---")
        for i in indices:
            r = results[i]
            diff = r["pnl_per_day"] - b_pnl
            r60 = r.get("recent_60d_pnl_per_day", 0)
            r30 = r.get("recent_30d_pnl_per_day", 0)
            l100 = r.get("days_below_neg100", 0)
            print(
                f"{r['name']:<25} {r['trades']:>6} "
                f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
                f"{r['max_dd']:>6.0f} {l100:>6} "
                f"{r60:>+7.2f} {r30:>+7.2f} {diff:>+7.2f}"
            )

    # Find winners
    winners = [r for r in results[1:] if r["pnl_per_day"] > b_pnl]
    if winners:
        print(f"\n  Variants that beat baseline (${b_pnl:.2f}/day):")
        for r in sorted(winners, key=lambda x: x["pnl_per_day"], reverse=True):
            diff = r["pnl_per_day"] - b_pnl
            print(f"    {r['name']:<25} ${r['pnl_per_day']:>+.2f}/day ({diff:>+.2f})")
    else:
        print(f"\n  No variants beat baseline (${b_pnl:.2f}/day)")

    # Save results
    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"direction_filters_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
