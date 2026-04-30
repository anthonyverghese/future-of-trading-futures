"""Test confirmation-based entry filters.

A bounce = price comes within 20pts of level, stays within tolerance,
and reverses 6pts past the line in the bounced direction, all within
5 minutes.

Variants:
1. 1 confirmation (any time today)
2. 1 confirmation from past 30 min
3. 2 confirmations (any time today)
4. 2 confirmations from past hour
5. 1 confirmation in last 30 min + max 2 trades per level per 30 min

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_confirmation.py
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

TOLERANCE = 20.0  # pts from level to be considered "near"
BOUNCE_PTS = 6.0  # pts past the line in bounced direction
BOUNCE_WINDOW_NS = 5 * 60 * 1_000_000_000  # 5 minutes in nanoseconds

_DATES = None
_CACHES = None
_ARRAYS = None


def _get_level_prices(dc):
    """Get level prices for current day."""
    ib_range = dc.ibh - dc.ibl
    return {
        "FIB_0.236": dc.ibl + 0.236 * ib_range,
        "FIB_0.618": dc.ibl + 0.618 * ib_range,
        "FIB_0.764": dc.ibl + 0.764 * ib_range,
        "FIB_EXT_HI_1.272": dc.fib_hi,
        "FIB_EXT_LO_1.272": dc.fib_lo,
    }


def _find_bounces(dc, level_price):
    """Find all bounce timestamps for a level.

    A bounce starts when price first enters within 1pt of the level
    (same threshold as trade entry). Then within 5 minutes:
    - Price can go up to TOLERANCE pts past the level
    - If price exceeds TOLERANCE → visit ends, no bounce
    - If price reverses to BOUNCE_PTS past the line in the opposite
      direction → bounce confirmed

    Returns list of bounce timestamps (ns since epoch).
    """
    fp = dc.full_prices
    ft = dc.full_ts_ns
    n = len(fp)
    bounces = []
    ENTRY_THRESHOLD = 1.0  # same as bot zone entry

    i = 0
    while i < n:
        p = float(fp[i])

        # Only start watching when price is within 1pt of level.
        if abs(p - level_price) > ENTRY_THRESHOLD:
            i += 1
            continue

        # Price touched the level. Track this visit.
        visit_start_ns = int(ft[i])
        visit_min = p
        visit_max = p

        j = i + 1
        while j < n:
            pj = float(fp[j])
            tj = int(ft[j])

            # Check if still within time window.
            if tj - visit_start_ns > BOUNCE_WINDOW_NS:
                break

            # Check if exceeded tolerance (level broke).
            if abs(pj - level_price) > TOLERANCE:
                break

            if pj < visit_min:
                visit_min = pj
            if pj > visit_max:
                visit_max = pj

            # Bounce UP: price dipped below level, then recovered
            # to BOUNCE_PTS above the line. The dip must have gone
            # at least 2pts below the level (not just entry noise).
            if pj >= level_price + BOUNCE_PTS and visit_min <= level_price - 2.0:
                bounces.append(tj)
                break

            # Bounce DOWN: price spiked above level, then dropped
            # to BOUNCE_PTS below the line. The spike must have gone
            # at least 2pts above the level.
            if pj <= level_price - BOUNCE_PTS and visit_max >= level_price + 2.0:
                bounces.append(tj)
                break

            j += 1

        # Skip past this visit to avoid double-counting.
        i = j + 1

    return bounces


def _count_confirmations(bounces, current_ns, lookback_ns=None):
    """Count bounces within the lookback window."""
    if lookback_ns is None:
        return len(bounces)
    cutoff = current_ns - lookback_ns
    return sum(1 for b in bounces if b >= cutoff)


def _count_recent_trades(trade_times_ns, current_ns, window_ns):
    """Count trades within the time window."""
    cutoff = current_ns - window_ns
    return sum(1 for t in trade_times_ns if t >= cutoff)


THIRTY_MIN_NS = 30 * 60 * 1_000_000_000
ONE_HOUR_NS = 60 * 60 * 1_000_000_000

VARIANTS = [
    ("Baseline", "baseline"),
    ("1 confirm (today)", "1_today"),
    ("1 confirm (30 min)", "1_30min"),
    ("2 confirms (today)", "2_today"),
    ("2 confirms (1 hour)", "2_1hour"),
    ("1 confirm 30m + max2/30m", "1_30min_max2"),
]


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

        if variant == "baseline":
            all_trades.extend(trades)
            continue

        # Pre-compute bounces for each level.
        level_prices = _get_level_prices(dc)
        level_bounces = {
            lv: _find_bounces(dc, lp)
            for lv, lp in level_prices.items()
        }

        # Track trade times per level for max-trades-per-window filter.
        level_trade_times: dict[str, list[int]] = {lv: [] for lv in level_prices}

        # Post-filter trades based on confirmation.
        filtered = []
        for t in trades:
            entry_ns = t.entry_ns
            bounces = level_bounces.get(t.level, [])
            # Only count bounces that happened BEFORE this entry.
            prior_bounces = [b for b in bounces if b < entry_ns]

            if variant == "1_today":
                if len(prior_bounces) < 1:
                    continue

            elif variant == "1_30min":
                if _count_confirmations(prior_bounces, entry_ns, THIRTY_MIN_NS) < 1:
                    continue

            elif variant == "2_today":
                if len(prior_bounces) < 2:
                    continue

            elif variant == "2_1hour":
                if _count_confirmations(prior_bounces, entry_ns, ONE_HOUR_NS) < 2:
                    continue

            elif variant == "1_30min_max2":
                if _count_confirmations(prior_bounces, entry_ns, THIRTY_MIN_NS) < 1:
                    continue
                recent_trades = _count_recent_trades(
                    level_trade_times.get(t.level, []),
                    entry_ns, THIRTY_MIN_NS
                )
                if recent_trades >= 2:
                    continue

            filtered.append(t)
            level_trade_times.setdefault(t.level, []).append(entry_ns)

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

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"confirmation_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
