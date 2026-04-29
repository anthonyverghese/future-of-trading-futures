"""Test scoring filters inspired by 2026-04-29 losing trades.

Today's losses: 3 consecutive stops on FIB_0.764/FIB_0.618 in 40 min.
- Trade 2: same level same dir as trade 1, 2 min later → stopped
- Trade 3: reverse direction on same level → stopped (level broken)
- Trade 4: different level, 16 sec after trade 3 → stopped (momentum)

Filters tested:
1. Global cooldown after any loss (60s, 120s across ALL levels)
2. No reverse direction after loss on same level
3. Tick rate filter (skip when > 2500, 3000)
4. Combos

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_scoring_filters.py
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
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH", "IBL"}

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    ("Baseline", {}),
    # Global cooldown after loss
    ("Global 30s cooldown", {"global_cooldown_after_loss_secs": 30}),
    ("Global 60s cooldown", {"global_cooldown_after_loss_secs": 60}),
    ("Global 120s cooldown", {"global_cooldown_after_loss_secs": 120}),
    ("Global 300s cooldown", {"global_cooldown_after_loss_secs": 300}),
    # No reverse after loss
    ("No reverse after loss", {"no_reverse_after_loss": True}),
    # Tick rate filter
    ("Tick rate < 2500", {"max_tick_rate": 2500}),
    ("Tick rate < 3000", {"max_tick_rate": 3000}),
    ("Tick rate < 3500", {"max_tick_rate": 3500}),
    # Combos
    ("60s cool + no reverse", {
        "global_cooldown_after_loss_secs": 60,
        "no_reverse_after_loss": True,
    }),
    ("60s cool + tick<3000", {
        "global_cooldown_after_loss_secs": 60,
        "max_tick_rate": 3000,
    }),
    ("No rev + tick<3000", {
        "no_reverse_after_loss": True,
        "max_tick_rate": 3000,
    }),
    ("All three (60s+norev+3000)", {
        "global_cooldown_after_loss_secs": 60,
        "no_reverse_after_loss": True,
        "max_tick_rate": 3000,
    }),
]


def _run_one(args):
    name, extra_kwargs = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        trades, streak = simulate_day(
            _CACHES[date], _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            **extra_kwargs,
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
    print(f"{'Variant':<30} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'vs base':>7}")
    print("-" * 115)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        print(
            f"{r['name']:<30} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {diff:>+7.2f}"
        )

    winners = [r for r in results[1:] if r["pnl_per_day"] > b_pnl]
    if winners:
        print(f"\n  Variants that beat baseline (${b_pnl:.2f}/day):")
        for r in sorted(winners, key=lambda x: x["pnl_per_day"], reverse=True):
            diff = r["pnl_per_day"] - b_pnl
            print(f"    {r['name']:<30} ${r['pnl_per_day']:>+.2f}/day ({diff:>+.2f})")
    else:
        print(f"\n  No variants beat baseline (${b_pnl:.2f}/day)")

    # Save results
    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"scoring_filters_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
