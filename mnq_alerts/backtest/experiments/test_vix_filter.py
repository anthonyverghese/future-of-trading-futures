"""Test VIX-based parameter adjustments.

Tests low VIX (<15) and high VIX (>25, >30) using prev close and today open.
All variants include Monday double caps (deployed config).

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_vix_filter.py
"""
import os, sys, time, datetime
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
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}

_DATES = None
_CACHES = None
_ARRAYS = None
_PREV_VIX = None  # previous day's close
_OPEN_VIX = None  # today's open


VARIANTS = [
    ("Baseline (incl Mon 2x)", "baseline"),
    # Low VIX
    ("Prev VIX<15: vol 0.25%", "prev_lt15_vol"),
    ("Open VIX<15: vol 0.25%", "open_lt15_vol"),
    # High VIX
    ("Prev VIX>30: halve caps", "prev_gt30_halve"),
    ("Open VIX>30: halve caps", "open_gt30_halve"),
    ("Prev VIX>25: halve caps", "prev_gt25_halve"),
    ("Open VIX>25: halve caps", "open_gt25_halve"),
    # Combos
    ("Prev: <15 vol + >30 halve", "prev_combo_30"),
    ("Open: <15 vol + >30 halve", "open_combo_30"),
    ("Prev: <15 vol + >25 halve", "prev_combo_25"),
]


def _run_one(args):
    name, variant = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        caps = dict(BASE_CAPS)
        vol = 0.0015

        # Monday double caps (deployed)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        prev = _PREV_VIX.get(date, 20)
        opn = _OPEN_VIX.get(date, 20)

        if variant == "prev_lt15_vol":
            if prev < 15: vol = 0.0025
        elif variant == "open_lt15_vol":
            if opn < 15: vol = 0.0025
        elif variant == "prev_gt30_halve":
            if prev > 30: caps = {k: max(1, v // 2) for k, v in caps.items()}
        elif variant == "open_gt30_halve":
            if opn > 30: caps = {k: max(1, v // 2) for k, v in caps.items()}
        elif variant == "prev_gt25_halve":
            if prev > 25: caps = {k: max(1, v // 2) for k, v in caps.items()}
        elif variant == "open_gt25_halve":
            if opn > 25: caps = {k: max(1, v // 2) for k, v in caps.items()}
        elif variant == "prev_combo_30":
            if prev < 15: vol = 0.0025
            if prev > 30: caps = {k: max(1, v // 2) for k, v in caps.items()}
        elif variant == "open_combo_30":
            if opn < 15: vol = 0.0025
            if opn > 30: caps = {k: max(1, v // 2) for k, v in caps.items()}
        elif variant == "prev_combo_25":
            if prev < 15: vol = 0.0025
            if prev > 25: caps = {k: max(1, v // 2) for k, v in caps.items()}

        trades, streak = simulate_day(
            _CACHES[date], _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels={"FIB_0.5", "IBH"},
            include_ibl=False, include_vwap=False,
            vol_filter_pct=vol,
        )
        all_trades.extend(trades)

    stats = compute_stats(all_trades, len(_DATES), list(_DATES))
    stats["name"] = name
    return stats


def main():
    global _DATES, _CACHES, _ARRAYS, _PREV_VIX, _OPEN_VIX
    t0 = time.time()

    print("Loading data...", flush=True)
    _DATES, _CACHES = load_all_days()
    print(f"Loaded {len(_DATES)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    _ARRAYS = {d: precompute_arrays(_CACHES[d]) for d in _DATES}

    print("Loading VIX...", flush=True)
    import yfinance as yf
    vix_df = yf.download("^VIX", start="2024-12-01", end="2026-05-01", progress=False)
    vix_close = {d.date(): float(row["Close"].iloc[0]) for d, row in vix_df.iterrows()}
    _OPEN_VIX = {d.date(): float(row["Open"].iloc[0]) for d, row in vix_df.iterrows()}

    # Build prev-day close lookup
    _PREV_VIX = {}
    sorted_dates = sorted(vix_close.keys())
    for i in range(1, len(sorted_dates)):
        _PREV_VIX[sorted_dates[i]] = vix_close[sorted_dates[i-1]]

    # Count triggers
    prev15 = sum(1 for d in _DATES if _PREV_VIX.get(d, 20) < 15)
    open15 = sum(1 for d in _DATES if _OPEN_VIX.get(d, 20) < 15)
    prev30 = sum(1 for d in _DATES if _PREV_VIX.get(d, 20) > 30)
    open30 = sum(1 for d in _DATES if _OPEN_VIX.get(d, 20) > 30)
    prev25 = sum(1 for d in _DATES if _PREV_VIX.get(d, 20) > 25)
    open25 = sum(1 for d in _DATES if _OPEN_VIX.get(d, 20) > 25)
    print(f"Trigger counts: VIX<15 prev={prev15}/open={open15}, VIX>25 prev={prev25}/open={open25}, VIX>30 prev={prev30}/open={open30}", flush=True)

    print(f"Running {len(VARIANTS)} variants across 3 workers...", flush=True)

    with Pool(3) as pool:
        results = pool.map(_run_one, VARIANTS)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    baseline = results[0]
    b_pnl = baseline["pnl_per_day"]
    print("=" * 115)
    print(f"{'Variant':<35} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'vs base':>7}")
    print("-" * 115)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        print(
            f"{r['name']:<35} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {diff:>+7.2f}"
        )

    # Save results
    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"vix_filters_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
