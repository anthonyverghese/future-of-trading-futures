"""Test variants targeting consistent P&L across quarters.

Key finding: IBH went from +$1.69/trade (Q1) to -$1.20/trade (Q4).
FIB_EXT_HI weakening too. FIB_0.618 is improving.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/test_consistency.py
"""
import os, sys, time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
import numpy as np

PER_LEVEL_TS = {
    "IBH": (6, 20),
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
}

BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 12, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}

VARIANTS = [
    # Current best
    {"name": "Baseline (caps, no 0.5)", "kwargs": {
        "max_per_level_map": BASE_CAPS,
    }},
    # IBH treatments
    {"name": "Exclude IBH", "kwargs": {
        "max_per_level_map": BASE_CAPS,
        "exclude_levels": {"IBH"},
    }},
    {"name": "IBH cap=3", "kwargs": {
        "max_per_level_map": {**BASE_CAPS, "IBH": 3},
    }},
    # FIB_EXT_HI weakening
    {"name": "Exclude IBH + FIB_EXT_HI", "kwargs": {
        "max_per_level_map": BASE_CAPS,
        "exclude_levels": {"IBH", "FIB_EXT_HI_1.272"},
    }},
    {"name": "FIB_EXT_HI cap=6", "kwargs": {
        "max_per_level_map": {**BASE_CAPS, "FIB_EXT_HI_1.272": 6},
    }},
    # FIB_0.618 is improving — raise its cap
    {"name": "FIB_0.618 cap=5", "kwargs": {
        "max_per_level_map": {**BASE_CAPS, "FIB_0.618": 5},
    }},
    {"name": "FIB_0.618 cap=5 + no IBH", "kwargs": {
        "max_per_level_map": {**BASE_CAPS, "FIB_0.618": 5},
        "exclude_levels": {"IBH"},
    }},
    # Only consistently strong levels
    {"name": "Strong only (no IBH, no EXT_HI)", "kwargs": {
        "max_per_level_map": {**BASE_CAPS, "FIB_0.618": 5},
        "exclude_levels": {"IBH", "FIB_EXT_HI_1.272"},
    }},
]

_DATES = None
_CACHES = None
_ARRAYS = None


def _run_one(args):
    name, kwargs = args
    daily = []
    streak = (0, 0)

    for date in _DATES:
        trades, streak = simulate_day(
            _CACHES[date], _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[1],
            include_ibl=False, include_vwap=False,
            exclude_levels=kwargs.get("exclude_levels", set()) | {"FIB_0.5"},
            **{k: v for k, v in kwargs.items() if k != "exclude_levels"},
        )
        day_pnl = sum(t.pnl_usd for t in trades)
        w = sum(1 for t in trades if t.pnl_usd >= 0)
        l = sum(1 for t in trades if t.pnl_usd < 0)
        daily.append((date, day_pnl, len(trades), w, l))

    n = len(daily)
    q = n // 4

    def stats(subset):
        pnls = [p for _, p, _, _, _ in subset]
        trades = sum(t for _, _, t, _, _ in subset)
        wins = sum(w for _, _, _, w, _ in subset)
        losses = sum(l_val for _, _, _, _, l_val in subset)
        wr = wins / trades * 100 if trades > 0 else 0
        avg = np.mean(pnls)
        w_days = sum(1 for p in pnls if p >= 0)
        l100 = sum(1 for p in pnls if p <= -100)
        return avg, wr, w_days / len(subset) * 100, l100

    full = stats(daily)
    q1 = stats(daily[:q])
    q2 = stats(daily[q:2*q])
    q3 = stats(daily[2*q:3*q])
    q4 = stats(daily[3*q:])
    last60 = stats(daily[-60:])

    # Consistency: stdev of quarterly $/day
    q_avgs = [q1[0], q2[0], q3[0], q4[0]]
    consistency = np.std(q_avgs)
    worst_q = min(q_avgs)

    return {
        "name": name,
        "full_pnl": full[0], "full_wr": full[1], "full_wdays": full[2], "full_l100": full[3],
        "q1": q1[0], "q2": q2[0], "q3": q3[0], "q4": q4[0],
        "last60": last60[0],
        "consistency": consistency,
        "worst_q": worst_q,
    }


def main():
    global _DATES, _CACHES, _ARRAYS
    t0 = time.time()

    print("Loading data...", flush=True)
    _DATES, _CACHES = load_all_days()
    print(f"Loaded {len(_DATES)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    _ARRAYS = {d: precompute_arrays(_CACHES[d]) for d in _DATES}
    print(f"Running {len(VARIANTS)} variants across 3 workers...", flush=True)

    args = [(v["name"], v["kwargs"]) for v in VARIANTS]
    with Pool(3) as pool:
        results = pool.map(_run_one, args)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 120)
    print(f"{'Variant':<35} {'$/day':>6} {'WR%':>5} {'W%d':>4} {'-100d':>5} {'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'L60d':>7} {'StDev':>6} {'WorstQ':>7}")
    print("-" * 120)
    for r in results:
        print(
            f"{r['name']:<35} {r['full_pnl']:>+6.1f} {r['full_wr']:>5.1f} "
            f"{r['full_wdays']:>3.0f}% {r['full_l100']:>5} "
            f"{r['q1']:>+7.1f} {r['q2']:>+7.1f} {r['q3']:>+7.1f} {r['q4']:>+7.1f} "
            f"{r['last60']:>+7.1f} {r['consistency']:>6.1f} {r['worst_q']:>+7.1f}"
        )

    print()
    best_pnl = max(results, key=lambda r: r["full_pnl"])
    most_consistent = min(results, key=lambda r: r["consistency"])
    best_worst_q = max(results, key=lambda r: r["worst_q"])
    best_recent = max(results, key=lambda r: r["last60"])
    print(f"Best $/day:        {best_pnl['name']} at ${best_pnl['full_pnl']:.2f}/day")
    print(f"Most consistent:   {most_consistent['name']} (stdev=${most_consistent['consistency']:.1f})")
    print(f"Best worst quarter:{best_worst_q['name']} (worst Q=${best_worst_q['worst_q']:.1f})")
    print(f"Best recent 60d:   {best_recent['name']} at ${best_recent['last60']:.2f}/day")


if __name__ == "__main__":
    main()
