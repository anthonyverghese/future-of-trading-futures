"""Test momentum-based scoring variants.

Key finding from analyze_scoring_v2.py:
- "With momentum" trades (price moving in trade direction at entry): 65-73% WR
- "Against momentum" trades (price approaching level from other side): 79-82% WR
- This is a ~10-15% WR gap — largest signal found across all factor analysis

"With momentum" means momentum is carrying price THROUGH the level.
"Against momentum" means price approached from the far side and may bounce.

Example: SELL at resistance. If price was falling (with momentum for SELL),
it's blasting through — bad. If price was rising toward resistance (against
momentum for SELL), it's a classic bounce setup — good.

Variants:
1. Baseline (unfiltered)
2. Skip with-momentum (1 min, >0pts)
3. Skip with-momentum (5 min, >0pts)
4. Skip strong with-momentum (1 min, >5pts)
5. Skip strong with-momentum (5 min, >5pts)
6. Skip strong with-momentum (5 min, >10pts)
7. Half caps for with-momentum trades
8. Skip with-momentum on weak levels only (FIB_0.618, FIB_0.764)
9. Skip with-momentum first hour only
10. Best combo: skip strong with-momentum 5min + adaptive caps

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_momentum_v1.py
"""
import os, sys, time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
import numpy as np
from collections import defaultdict

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
IB_SET = 630

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    ("Baseline", "baseline"),
    ("Skip with-mom 1min >0", "skip_with_1min_0"),
    ("Skip with-mom 5min >0", "skip_with_5min_0"),
    ("Skip with-mom 1min >5pt", "skip_with_1min_5"),
    ("Skip with-mom 5min >5pt", "skip_with_5min_5"),
    ("Skip with-mom 5min >10pt", "skip_with_5min_10"),
    ("Half caps with-mom 5min", "half_caps_with_5min"),
    ("Skip with-mom weak levels", "skip_with_weak_levels"),
    ("Skip with-mom first hr", "skip_with_first_hr"),
    ("Skip with-mom 5m>5 + adap caps", "skip_with_5min_5_adaptive"),
]


def _compute_momentum(dc, entry_idx, direction, lookback_ticks):
    """Compute momentum relative to trade direction.

    Positive = price moving WITH trade direction (bad signal).
    Negative = price moving AGAINST trade direction (good signal).
    """
    fp = dc.full_prices
    start_idx = max(0, entry_idx - lookback_ticks)
    if start_idx >= len(fp) or entry_idx >= len(fp):
        return 0.0
    momentum = float(fp[entry_idx]) - float(fp[start_idx])
    if direction == "down":
        momentum = -momentum
    return momentum


def _apply_filters(trades, variant, dc):
    if variant == "baseline":
        return trades

    filtered = []
    cum_pnl = 0.0
    stopped = False
    level_counts = defaultdict(int)

    # Adaptive caps state (for combined variant)
    adaptive_caps_restored = False
    adaptive_caps_until_et = IB_SET + 30
    adaptive_accepted = 0
    adaptive_any_loss = False

    for t in trades:
        if stopped:
            break

        et = t.get("et_mins", 0)
        pnl = t["pnl_usd"]
        level = t["level"]
        direction = t["direction"]
        entry_idx = t["entry_idx"]

        # $100 daily loss limit
        if cum_pnl <= -100:
            stopped = True
            break

        # Compute momentum
        mom_1min = _compute_momentum(dc, entry_idx, direction, 200)
        mom_5min = _compute_momentum(dc, entry_idx, direction, 1000)

        # --- Variant-specific checks ---

        if variant == "skip_with_1min_0":
            if mom_1min > 0:
                continue

        elif variant == "skip_with_5min_0":
            if mom_5min > 0:
                continue

        elif variant == "skip_with_1min_5":
            if mom_1min > 5:
                continue

        elif variant == "skip_with_5min_5":
            if mom_5min > 5:
                continue

        elif variant == "skip_with_5min_10":
            if mom_5min > 10:
                continue

        elif variant == "half_caps_with_5min":
            if mom_5min > 0:
                day_caps = dict(BASE_CAPS)
                if dc.date.weekday() == 0:
                    day_caps = {k: v * 2 for k, v in day_caps.items()}
                half_cap = max(1, day_caps.get(level, 99) // 2)
                if level_counts[level] >= half_cap:
                    continue

        elif variant == "skip_with_weak_levels":
            # Only skip with-momentum on levels with worst WR split
            if mom_5min > 0 and level in ("FIB_0.618", "FIB_0.764"):
                continue

        elif variant == "skip_with_first_hr":
            if mom_5min > 0 and et < IB_SET + 60:
                continue

        elif variant == "skip_with_5min_5_adaptive":
            # Skip strong with-momentum
            if mom_5min > 5:
                continue
            # Adaptive caps: half caps first 30 min, restore on 3 wins
            if not adaptive_caps_restored:
                if et < adaptive_caps_until_et:
                    day_caps = dict(BASE_CAPS)
                    if dc.date.weekday() == 0:
                        day_caps = {k: v * 2 for k, v in day_caps.items()}
                    half_cap = max(1, day_caps.get(level, 99) // 2)
                    if level_counts[level] >= half_cap:
                        continue
                else:
                    adaptive_caps_restored = True

        # --- Accept trade ---
        filtered.append(t)
        level_counts[level] += 1
        cum_pnl += pnl
        adaptive_accepted += 1

        # --- Post-trade updates for adaptive caps ---
        if variant == "skip_with_5min_5_adaptive":
            if pnl < 0:
                if not adaptive_caps_restored:
                    adaptive_any_loss = True
                    adaptive_caps_until_et = et + 30
            else:
                if not adaptive_caps_restored and adaptive_accepted >= 3 and not adaptive_any_loss:
                    adaptive_caps_restored = True

    return filtered


def _run_one(args):
    name, variant = args
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
        )

        trade_dicts = [{
            "level": t.level, "direction": t.direction,
            "pnl_usd": t.pnl_usd, "outcome": t.outcome,
            "et_mins": t.factors.et_mins if t.factors else 0,
            "entry_ns": t.entry_ns, "entry_idx": t.entry_idx,
            "entry_count": t.factors.entry_count if t.factors else 0,
            "_idx": i,
        } for i, t in enumerate(trades)]

        filtered_dicts = _apply_filters(trade_dicts, variant, dc)
        filtered_trades = [trades[fd["_idx"]] for fd in filtered_dicts]
        all_trades.extend(filtered_trades)

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
    print(f"{'Variant':<36} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 140)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<36} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<36} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"momentum_v1_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
