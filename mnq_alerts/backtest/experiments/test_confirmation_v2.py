"""Test confirmation and pace-limiting tweaks.

Explores:
- Max trades per level per time window (standalone)
- 1 confirm tweaks (easier bounce, interior fibs only)
- Combos of confirmation + pace limiting

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_confirmation_v2.py
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
INTERIOR = {"FIB_0.236", "FIB_0.618", "FIB_0.764"}

TOLERANCE = 20.0
BOUNCE_WINDOW_NS = 5 * 60 * 1_000_000_000

_DATES = None
_CACHES = None
_ARRAYS = None


def _get_level_prices(dc):
    ib_range = dc.ibh - dc.ibl
    return {
        "FIB_0.236": dc.ibl + 0.236 * ib_range,
        "FIB_0.618": dc.ibl + 0.618 * ib_range,
        "FIB_0.764": dc.ibl + 0.764 * ib_range,
        "FIB_EXT_HI_1.272": dc.fib_hi,
        "FIB_EXT_LO_1.272": dc.fib_lo,
    }


def _find_bounces(dc, level_price, bounce_pts=6.0):
    """Find bounce timestamps. See test_confirmation.py for full docs."""
    fp = dc.full_prices
    ft = dc.full_ts_ns
    n = len(fp)
    bounces = []
    ENTRY_THRESHOLD = 1.0

    i = 0
    while i < n:
        p = float(fp[i])
        if abs(p - level_price) > ENTRY_THRESHOLD:
            i += 1
            continue

        visit_start_ns = int(ft[i])
        visit_min = p
        visit_max = p

        j = i + 1
        while j < n:
            pj = float(fp[j])
            tj = int(ft[j])

            if tj - visit_start_ns > BOUNCE_WINDOW_NS:
                break
            if abs(pj - level_price) > TOLERANCE:
                break

            if pj < visit_min: visit_min = pj
            if pj > visit_max: visit_max = pj

            if pj >= level_price + bounce_pts and visit_min <= level_price - 2.0:
                bounces.append(tj)
                break
            if pj <= level_price - bounce_pts and visit_max >= level_price + 2.0:
                bounces.append(tj)
                break

            j += 1

        i = j + 1

    return bounces


VARIANTS = [
    ("Baseline", {"confirm": None, "max_per_window": None}),
    # Standalone pace limiting
    ("Max 1/lvl/30min", {"confirm": None, "max_per_window": (1, 30)}),
    ("Max 2/lvl/30min", {"confirm": None, "max_per_window": (2, 30)}),
    ("Max 3/lvl/30min", {"confirm": None, "max_per_window": (3, 30)}),
    ("Max 2/lvl/20min", {"confirm": None, "max_per_window": (2, 20)}),
    # 1 confirm tweaks
    ("1 confirm (2 hours)", {"confirm": {"required": 1, "bounce": 6.0, "levels": "all", "lookback_min": 120}, "max_per_window": None}),
    ("1 confirm interior only", {"confirm": {"required": 1, "bounce": 6.0, "levels": "interior"}, "max_per_window": None}),
    # Combos
    ("1 confirm + max 2/30m", {"confirm": {"required": 1, "bounce": 6.0, "levels": "all"}, "max_per_window": (2, 30)}),
    ("1 confirm + max 3/30m", {"confirm": {"required": 1, "bounce": 6.0, "levels": "all"}, "max_per_window": (3, 30)}),
    ("1 confirm int + max 2/30m", {"confirm": {"required": 1, "bounce": 6.0, "levels": "interior"}, "max_per_window": (2, 30)}),
]


def _run_one(args):
    name, cfg = args
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

        # Pre-compute bounces if needed.
        confirm_cfg = cfg.get("confirm")
        level_bounces = {}
        if confirm_cfg:
            level_prices = _get_level_prices(dc)
            bounce_pts = confirm_cfg.get("bounce", 6.0)
            for lv, lp in level_prices.items():
                level_bounces[lv] = _find_bounces(dc, lp, bounce_pts)

        # Track trade times per level for pace limiting.
        level_trade_times: dict[str, list[int]] = defaultdict(list)
        max_per_window = cfg.get("max_per_window")

        filtered = []
        for t in trades:
            entry_ns = t.entry_ns

            # Confirmation filter.
            if confirm_cfg:
                required = confirm_cfg["required"]
                levels_scope = confirm_cfg["levels"]

                # Skip confirmation for extensions if scope is "interior".
                needs_confirm = True
                if levels_scope == "interior" and t.level not in INTERIOR:
                    needs_confirm = False

                if needs_confirm:
                    prior_bounces = [b for b in level_bounces.get(t.level, []) if b < entry_ns]
                    lookback_min = confirm_cfg.get("lookback_min")
                    if lookback_min:
                        lookback_ns = lookback_min * 60 * 1_000_000_000
                        prior_bounces = [b for b in prior_bounces if b >= entry_ns - lookback_ns]
                    if len(prior_bounces) < required:
                        continue

            # Pace limiting: max trades per level per time window.
            if max_per_window:
                max_trades, window_min = max_per_window
                window_ns = window_min * 60 * 1_000_000_000
                cutoff = entry_ns - window_ns
                recent = sum(1 for tt in level_trade_times[t.level] if tt >= cutoff)
                if recent >= max_trades:
                    continue

            filtered.append(t)
            level_trade_times[t.level].append(entry_ns)

        all_trades.extend(filtered)

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

    print("=" * 125)
    print(f"{'Variant':<28} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 125)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        q = r.get("quarterly_pnl_per_day", {})
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
    path = results_dir / f"confirmation_v2_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
