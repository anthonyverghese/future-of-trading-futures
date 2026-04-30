"""Test risk management improvements: trend filter, VWAP filter,
split budget, confirmation bounce.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_risk_management.py
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
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH", "IBL"}
INTERIOR = {"FIB_0.236", "FIB_0.618", "FIB_0.764"}

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    ("Baseline", {}),
    # Trend-aligned (session move direction)
    ("Trend: block counter", {"trend_filter": "block"}),
    ("Trend: ext only counter", {"trend_filter": "ext_only"}),
    ("Trend: halve counter caps", {"trend_filter": "halve"}),
    # VWAP directional filter
    ("VWAP: block counter", {"vwap_filter": "block"}),
    ("VWAP: ext only counter", {"vwap_filter": "ext_only"}),
    ("VWAP: halve counter caps", {"vwap_filter": "halve"}),
    # Confirmation bounce (post-filter)
    ("Confirm 2pt bounce", {"confirm_pts": 2.0}),
    ("Confirm 4pt bounce", {"confirm_pts": 4.0}),
    ("Confirm 2pt counter only", {"confirm_pts": 2.0, "confirm_counter_only": True}),
    # Split budget
    ("Budget $60 AM / $40 PM", {"split_budget": (60, 40)}),
    ("Budget $40 AM / $60 PM", {"split_budget": (40, 60)}),
]


def _check_bounce(dc, entry_idx, level_price, bounce_pts):
    """Check if price bounced bounce_pts away from level before entry.

    Looks at the 300 ticks before entry_idx for a tick where price was
    within 1pt of the level, then moved bounce_pts away, then came back.
    """
    fp = dc.full_prices
    start = max(0, entry_idx - 300)
    # Find first touch (within 1pt)
    first_touch = -1
    for i in range(start, entry_idx):
        if abs(float(fp[i]) - level_price) <= 1.0:
            first_touch = i
            break
    if first_touch < 0:
        return True  # no prior touch found, allow entry
    # Check if price moved bounce_pts away after first touch
    max_dist = 0.0
    for i in range(first_touch, entry_idx):
        dist = abs(float(fp[i]) - level_price)
        if dist > max_dist:
            max_dist = dist
    return max_dist >= bounce_pts


def _run_one(args):
    name, cfg = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        sim_kwargs = {
            "global_cooldown_after_loss_secs": 30,
        }
        if "trend_filter" in cfg:
            sim_kwargs["trend_filter"] = cfg["trend_filter"]
        if "vwap_filter" in cfg:
            sim_kwargs["vwap_filter"] = cfg["vwap_filter"]
        if "split_budget" in cfg:
            sim_kwargs["split_budget"] = cfg["split_budget"]

        trades, streak = simulate_day(
            _CACHES[date], _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            **sim_kwargs,
        )

        # Post-filter: confirmation bounce.
        confirm_pts = cfg.get("confirm_pts", 0)
        confirm_counter = cfg.get("confirm_counter_only", False)
        if confirm_pts > 0 and trades:
            dc = _CACHES[date]
            filtered = []
            for t in trades:
                # Determine if counter-trend.
                session_mv = t.factors.session_move
                trend_dir = "down" if session_mv < 0 else "up"
                is_counter = (t.direction != trend_dir)

                if confirm_counter and not is_counter:
                    filtered.append(t)  # no confirmation needed for with-trend
                    continue

                # Check for bounce in ticks before entry.
                level_price = dc.full_prices[t.entry_idx]
                # Approximate level price from zone (find closest fixed level)
                ib_range = dc.ibh - dc.ibl
                level_map = {
                    "FIB_0.236": dc.ibl + 0.236 * ib_range,
                    "FIB_0.618": dc.ibl + 0.618 * ib_range,
                    "FIB_0.764": dc.ibl + 0.764 * ib_range,
                    "FIB_EXT_HI_1.272": dc.fib_hi,
                    "FIB_EXT_LO_1.272": dc.fib_lo,
                }
                lp = level_map.get(t.level, level_price)

                if _check_bounce(dc, t.entry_idx, lp, confirm_pts):
                    filtered.append(t)
            trades = filtered

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

    print("=" * 120)
    print(f"{'Variant':<30} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 120)
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

    winners = [r for r in results[1:] if r["pnl_per_day"] > b_pnl]
    if winners:
        print(f"\n  Variants that beat baseline (${b_pnl:.2f}/day):")
        for r in sorted(winners, key=lambda x: x["pnl_per_day"], reverse=True):
            diff = r["pnl_per_day"] - b_pnl
            print(f"    {r['name']:<30} ${r['pnl_per_day']:>+.2f}/day ({diff:>+.2f})")
    else:
        print(f"\n  No variants beat baseline (${b_pnl:.2f}/day)")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"risk_management_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
