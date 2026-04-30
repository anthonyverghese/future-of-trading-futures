"""Test offensive + defensive combos.

Instead of just cutting trades (always costs P&L), ADD profitable
trades while managing risk with conditional limits.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_offensive_combos.py
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
    "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
    "IBH": 7,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH", "IBL"}

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    ("Baseline", {
        "exclude": {"FIB_0.5", "IBH", "IBL"},
        "caps": {"FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
                 "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6},
        "dir_filter": None,
        "limit": "100",
        "friday_double": False,
    }),
    # Offensive + defensive
    ("IBH SELL + $75 cond", {
        "exclude": {"FIB_0.5", "IBL"},
        "caps": {"FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
                 "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7},
        "dir_filter": {"IBH": "down"},
        "limit": "75_cond",
        "friday_double": False,
    }),
    ("IBH SELL + $60 1st hr", {
        "exclude": {"FIB_0.5", "IBL"},
        "caps": {"FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
                 "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7},
        "dir_filter": {"IBH": "down"},
        "limit": "60_first_hr",
        "friday_double": False,
    }),
    ("IBH SELL + 0.236 cap=18", {
        "exclude": {"FIB_0.5", "IBL"},
        "caps": {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
                 "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7},
        "dir_filter": {"IBH": "down"},
        "limit": "100",
        "friday_double": False,
    }),
    ("0.236 cap=18 + $75 cond", {
        "exclude": {"FIB_0.5", "IBH", "IBL"},
        "caps": {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
                 "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6},
        "dir_filter": None,
        "limit": "75_cond",
        "friday_double": False,
    }),
    ("IBH SELL+0.236=18+$75c", {
        "exclude": {"FIB_0.5", "IBL"},
        "caps": {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
                 "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7},
        "dir_filter": {"IBH": "down"},
        "limit": "75_cond",
        "friday_double": False,
    }),
    ("Fri double + $75 cond", {
        "exclude": {"FIB_0.5", "IBH", "IBL"},
        "caps": {"FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
                 "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6},
        "dir_filter": None,
        "limit": "75_cond",
        "friday_double": True,
    }),
    ("All caps+50% + $75 cond", {
        "exclude": {"FIB_0.5", "IBH", "IBL"},
        "caps": {"FIB_0.236": 18, "FIB_0.618": 5, "FIB_0.764": 8,
                 "FIB_EXT_HI_1.272": 9, "FIB_EXT_LO_1.272": 9},
        "dir_filter": None,
        "limit": "75_cond",
        "friday_double": False,
    }),
]


def _apply_limit(trades, limit_type):
    """Apply loss limit post-filter."""
    if limit_type == "100":
        return trades

    filtered = []
    cum_pnl = 0.0
    peak_pnl = 0.0
    first_et = None

    for t in trades:
        et = t.get('et_mins', 0)
        if first_et is None:
            first_et = et

        if limit_type == "75_cond":
            limit = 75 if peak_pnl < 20 else 100
            if cum_pnl <= -limit:
                break

        elif limit_type == "60_first_hr":
            minutes = et - first_et if first_et else 0
            limit = 60 if minutes <= 60 else 100
            if cum_pnl <= -limit:
                break

        filtered.append(t)
        cum_pnl += t['pnl_usd']
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl

    return filtered


def _run_one(args):
    name, cfg = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        dc = _CACHES[date]
        caps = dict(cfg["caps"])

        # Monday double caps (deployed)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        # Friday double caps
        if cfg.get("friday_double") and date.weekday() == 4:
            caps = {k: v * 2 for k, v in caps.items()}

        trades, streak = simulate_day(
            dc, _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=cfg["exclude"],
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=30,
            direction_filter=cfg.get("dir_filter"),
        )

        # Apply loss limit post-filter
        if cfg["limit"] != "100":
            trade_dicts = [{
                'level': t.level, 'pnl_usd': t.pnl_usd,
                'et_mins': t.factors.et_mins if t.factors else 0,
            } for t in trades]
            filtered_dicts = _apply_limit(trade_dicts, cfg["limit"])
            trades = trades[:len(filtered_dicts)]

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

    print("=" * 130)
    print(f"{'Variant':<28} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 130)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<28} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<28} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"offensive_combos_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
