"""Test defensive factors on the new deployed config (IBH SELL + 0.236 cap=18).

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_defensive_v2.py
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

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    ("Baseline", "baseline"),
    ("IBH after 11:30 only", "ibh_after_1130"),
    ("IBH retest from above", "ibh_retest"),
    ("IBH cap=3", "ibh_cap3"),
    ("0.236 cap=6 after loss", "fib236_reactive"),
    ("Pace: pause 5m after 10/hr", "dynamic_pace"),
    ("$60 first hour limit", "limit_60_first_hr"),
    ("IBH 11:30 + 0.236 react", "combo_ibh_fib"),
    ("IBH retest + $60 1hr + pace", "combo_retest_limit_pace"),
    ("IBH cap3 + 0.236 react + $60", "combo_all"),
]


def _was_above_ibh_recently(dc, entry_idx, ibh, lookback_ticks=5000):
    """Check if price was above IBH in the recent past (within ~30 min of ticks)."""
    fp = dc.full_prices
    start = max(0, entry_idx - lookback_ticks)
    for i in range(start, entry_idx):
        if float(fp[i]) > ibh + 1.0:  # was meaningfully above IBH
            return True
    return False


def _apply_filters(trades, variant, dc):
    if variant == "baseline":
        return trades

    filtered = []
    cum_pnl = 0.0
    first_et = None
    trade_times = []  # entry_ns for pace
    fib236_had_loss = False
    fib236_post_loss_count = 0
    stopped = False

    ibh = dc.ibh

    for t in trades:
        if stopped:
            break

        et = t.get("et_mins", 0)
        entry_ns = t.get("entry_ns", 0)
        entry_idx = t.get("entry_idx", 0)
        pnl = t["pnl_usd"]
        level = t["level"]

        if first_et is None:
            first_et = et

        # --- Pre-trade checks ---

        # IBH after 11:30 only (et_mins >= 690)
        if variant in ("ibh_after_1130", "combo_ibh_fib"):
            if level == "IBH" and et < 690:
                continue

        # IBH retest from above
        if variant in ("ibh_retest", "combo_retest_limit_pace"):
            if level == "IBH":
                if not _was_above_ibh_recently(dc, entry_idx, ibh):
                    continue

        # IBH cap=3
        if variant in ("ibh_cap3", "combo_all"):
            if level == "IBH":
                ibh_count = sum(1 for f in filtered if f["level"] == "IBH")
                if ibh_count >= 3:
                    continue

        # FIB_0.236 reactive cap
        if variant in ("fib236_reactive", "combo_ibh_fib", "combo_all"):
            if level == "FIB_0.236" and fib236_had_loss:
                if fib236_post_loss_count >= 6:
                    continue

        # Dynamic pace: pause 5 min after 10+ trades in last hour
        if variant in ("dynamic_pace", "combo_retest_limit_pace"):
            one_hr_ns = 60 * 60 * 1_000_000_000
            five_min_ns = 5 * 60 * 1_000_000_000
            recent_1hr = sum(1 for tns in trade_times if tns >= entry_ns - one_hr_ns)
            if recent_1hr >= 10:
                recent_5min = sum(1 for tns in trade_times if tns >= entry_ns - five_min_ns)
                if recent_5min > 0:  # still within 5 min pause
                    continue

        # $60 first hour loss limit
        if variant in ("limit_60_first_hr", "combo_retest_limit_pace", "combo_all"):
            minutes = et - first_et if first_et else 0
            if minutes <= 60:
                if cum_pnl <= -60:
                    # Don't stop entirely — just skip until first hour ends
                    continue
            else:
                if cum_pnl <= -100:
                    stopped = True
                    break

        # Regular $100 limit for variants without special limit
        if variant not in ("limit_60_first_hr", "combo_retest_limit_pace", "combo_all"):
            if cum_pnl <= -100:
                stopped = True
                break

        # --- Accept trade ---
        filtered.append(t)
        cum_pnl += pnl
        trade_times.append(entry_ns)

        # --- Post-trade updates ---
        if pnl < 0:
            if level == "FIB_0.236":
                fib236_had_loss = True
                fib236_post_loss_count = 0
        if level == "FIB_0.236" and fib236_had_loss:
            fib236_post_loss_count += 1

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

    print("=" * 130)
    print(f"{'Variant':<32} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 130)
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

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"defensive_v2_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
