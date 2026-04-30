"""Test smart loss limit variants.

Based on analysis: 78 days hit -$75, 64% kept losing to -$100+,
29% recovered. Only 19% of "straight loss" days recover vs 35%
of "peaked then crashed" days. 8 false stops cost $3.12/day.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_smart_limits.py
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
    ("Baseline ($100 limit)", "baseline"),
    ("$75 limit", "limit_75"),
    ("$75 if no peak>$20", "limit_75_no_peak"),
    ("$75, extend $100 if peak>$20", "limit_75_extend"),
    ("$75 + pace 4/30min", "limit_75_pace"),
    ("$75 + 0.236 cap=6", "limit_75_fib236"),
    ("$60 first 60min, $100 after", "limit_60_first_hour"),
    ("Stop after 3 consec losses", "stop_3consec"),
    ("$75 + 0.236 cap=6 + pace", "combo_all"),
]


def _apply_filters(trades, variant):
    """Apply smart limit filters to a day's trades."""
    if variant == "baseline":
        return trades

    filtered = []
    cum_pnl = 0.0
    peak_pnl = 0.0
    consec_losses = 0
    trade_times = []  # entry_ns for pace tracking
    fib236_loss_count = 0
    fib236_total_after_first_loss = 0
    fib236_had_loss = False
    stopped = False
    first_trade_et = None

    for t in trades:
        if stopped:
            break

        et = t.get('et_mins', 0)
        entry_ns = t.get('entry_ns', 0)
        pnl = t['pnl_usd']
        level = t['level']

        if first_trade_et is None:
            first_trade_et = et

        # --- Pre-trade checks ---

        # Dynamic pace: pause 10min after 4+ trades in 30min
        if variant in ("limit_75_pace", "combo_all"):
            cutoff_30 = entry_ns - 30 * 60 * 1_000_000_000
            cutoff_10 = entry_ns - 10 * 60 * 1_000_000_000
            recent_30 = sum(1 for tns in trade_times if tns >= cutoff_30)
            recent_10 = sum(1 for tns in trade_times if tns >= cutoff_10)
            if recent_30 >= 4 and recent_10 >= 4:
                continue

        # FIB_0.236 reactive cap
        if variant in ("limit_75_fib236", "combo_all"):
            if level == "FIB_0.236" and fib236_had_loss:
                if fib236_total_after_first_loss >= 6:
                    continue

        # Loss limit checks
        if variant == "limit_75":
            if cum_pnl <= -75:
                stopped = True
                break

        elif variant == "limit_75_no_peak":
            # $75 limit only if never peaked >$20; otherwise $100
            limit = 75 if peak_pnl < 20 else 100
            if cum_pnl <= -limit:
                stopped = True
                break

        elif variant == "limit_75_extend":
            # Start at $75, extend to $100 once peaked >$20
            limit = 100 if peak_pnl >= 20 else 75
            if cum_pnl <= -limit:
                stopped = True
                break

        elif variant in ("limit_75_pace", "limit_75_fib236"):
            if cum_pnl <= -75:
                stopped = True
                break

        elif variant == "limit_60_first_hour":
            minutes_since_start = et - first_trade_et if first_trade_et else 0
            if minutes_since_start <= 60:
                limit = 60
            else:
                limit = 100
            if cum_pnl <= -limit:
                stopped = True
                break

        elif variant == "stop_3consec":
            if consec_losses >= 3:
                stopped = True
                break
            if cum_pnl <= -100:  # still have the $100 overall limit
                stopped = True
                break

        elif variant == "combo_all":
            if cum_pnl <= -75:
                stopped = True
                break

        # --- Accept trade ---
        filtered.append(t)
        cum_pnl += pnl
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        trade_times.append(entry_ns)

        # --- Post-trade state updates ---
        if pnl < 0:
            consec_losses += 1
            if level == "FIB_0.236":
                fib236_had_loss = True
        else:
            consec_losses = 0

        if level == "FIB_0.236" and fib236_had_loss:
            fib236_total_after_first_loss += 1

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
        )

        trade_dicts = []
        for t in trades:
            trade_dicts.append({
                'level': t.level,
                'direction': t.direction,
                'pnl_usd': t.pnl_usd,
                'outcome': t.outcome,
                'et_mins': t.factors.et_mins if t.factors else 0,
                'entry_ns': t.entry_ns,
            })

        filtered_dicts = _apply_filters(trade_dicts, variant)

        # Map back to TradeRecords
        filtered_trades = trades[:len(filtered_dicts)]
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

    print("=" * 130)
    print(f"{'Variant':<30} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 130)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<30} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<30} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"smart_limits_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
