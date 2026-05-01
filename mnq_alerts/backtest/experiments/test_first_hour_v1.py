"""Test first-hour defensive variants based on bad-day signal analysis.

Key findings driving these variants:
- Bad days: first loss within 7 min median (100% within 30 min)
- Bad days: avg 1.3 wins before first loss vs 4.6 on good days
- Bad days: 81% break IB range in first hour vs 64% good days
- Bad days: 46% above VWAP at IB set vs 60% good days
- Bad days: FIB_0.236 WR 57% vs 87% good days in first hour
- Simple kill switch after first loss destroys good-day recovery

Strategy: use COMBINATIONS of signals to throttle only when multiple bad-day
indicators align, preserving good-day recovery.

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. conda run -n mnq python -u mnq_alerts/backtest/experiments/test_first_hour_v1.py
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

IB_SET = 630         # 10:30 AM ET in et_mins
FIRST_HOUR_END = 690  # 11:30 AM ET

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    ("Baseline", "baseline"),
    ("Early loss+IB breakout", "early_loss_ib_breakout"),
    ("Early loss+below VWAP", "early_loss_below_vwap"),
    ("Graduated response", "graduated_response"),
    ("Win-streak gate", "win_streak_gate"),
    ("Combined 3-signal", "combined_3signal"),
    ("Interior fib skip 1hr", "interior_fib_skip"),
    ("Adaptive caps 1hr", "adaptive_caps"),
]


def _ib_broken_in_first_hour(trades, dc):
    """Check if any trade in the first hour has entry price above IBH or below IBL."""
    for t in trades:
        et = t.get("et_mins", 0)
        if IB_SET <= et < FIRST_HOUR_END:
            entry_price = dc.full_prices[t["entry_idx"]]
            if entry_price > dc.ibh or entry_price < dc.ibl:
                return True
    return False


def _price_above_vwap_at_ib(dc):
    """Check if price was above VWAP when IB period ended."""
    if len(dc.post_ib_vwaps) == 0:
        return True  # default to optimistic
    price_at_ib = dc.post_ib_prices[0]
    vwap_at_ib = dc.post_ib_vwaps[0]
    return price_at_ib >= vwap_at_ib


def _apply_filters(trades, variant, dc):
    if variant == "baseline":
        return trades

    filtered = []
    cum_pnl = 0.0
    wins_so_far = 0
    losses_so_far = 0
    first_loss_et = None       # et_mins of first loss
    consecutive_wins_after_loss = 0
    has_lost = False
    pause_until_et = 0         # pause trading until this et_mins
    half_caps_until_et = 0     # halve caps until this et_mins
    half_caps_rest_of_day = False
    full_caps_restored = False  # for adaptive caps
    adaptive_half_until_et = IB_SET + 30  # first 30 min for adaptive
    adaptive_loss_seen = False
    stopped = False

    # Pre-scan for IB breakout (needed by some variants)
    ib_broken = _ib_broken_in_first_hour(trades, dc)
    above_vwap = _price_above_vwap_at_ib(dc)

    # Per-level counters for cap enforcement
    level_counts = defaultdict(int)
    # Effective caps (may be halved)
    base_caps = {}
    for t in trades:
        # Reconstruct what the day's caps were
        # (Monday doubling already applied in simulate_day, so caps here are post-Monday)
        pass
    # We'll track per-level counts ourselves since we're filtering

    for t in trades:
        if stopped:
            break

        et = t.get("et_mins", 0)
        pnl = t["pnl_usd"]
        level = t["level"]
        in_first_hour = IB_SET <= et < FIRST_HOUR_END

        # --- Check pauses ---
        if et < pause_until_et:
            continue

        # --- Variant-specific pre-trade checks ---

        if variant == "early_loss_ib_breakout":
            # After first loss within 15 min of IB AND IB broken, pause 10 min
            if has_lost and first_loss_et is not None:
                if first_loss_et <= IB_SET + 15 and ib_broken:
                    if et < first_loss_et + 10:
                        continue

        elif variant == "early_loss_below_vwap":
            # After first loss within 15 min of IB AND below VWAP at IB set,
            # halve caps for 30 min after first loss
            if has_lost and first_loss_et is not None:
                if first_loss_et <= IB_SET + 15 and not above_vwap:
                    if et < first_loss_et + 30:
                        # Half caps: skip if this level already hit half its cap
                        day_caps = dict(BASE_CAPS)
                        if dc.date.weekday() == 0:
                            day_caps = {k: v * 2 for k, v in day_caps.items()}
                        half_cap = max(1, day_caps.get(level, 99) // 2)
                        if level_counts[level] >= half_cap:
                            continue

        elif variant == "graduated_response":
            # After first loss in first hour: require 2 consecutive wins
            # before taking full-risk trades. Until then, skip FIB_0.236 and FIB_0.618.
            if has_lost and in_first_hour:
                if consecutive_wins_after_loss < 2:
                    if level in ("FIB_0.236", "FIB_0.618"):
                        continue

        elif variant == "win_streak_gate":
            # Don't allow more than 2 trades in first hour until 2 wins.
            # If first 2 trades include a loss, pause 10 min from that loss.
            if in_first_hour:
                first_hour_count = len([f for f in filtered
                                         if IB_SET <= f.get("et_mins", 0) < FIRST_HOUR_END])
                if wins_so_far < 2 and first_hour_count >= 2:
                    continue
                if has_lost and first_loss_et is not None and first_hour_count < 2:
                    if et < first_loss_et + 10:
                        continue

        elif variant == "combined_3signal":
            # Loss within 15 min AND <3 wins AND IB broken:
            # stop 20 min + halve caps rest of day
            if has_lost and first_loss_et is not None:
                if (first_loss_et <= IB_SET + 15
                        and wins_so_far < 3
                        and ib_broken):
                    if et < first_loss_et + 20:
                        continue
                    # Halve caps for rest of day
                    if not half_caps_rest_of_day:
                        half_caps_rest_of_day = True
            if half_caps_rest_of_day:
                day_caps = dict(BASE_CAPS)
                if dc.date.weekday() == 0:
                    day_caps = {k: v * 2 for k, v in day_caps.items()}
                half_cap = max(1, day_caps.get(level, 99) // 2)
                if level_counts[level] >= half_cap:
                    continue

        elif variant == "interior_fib_skip":
            # Skip FIB_0.236 and FIB_0.618 in first 30 min after IB
            if et < IB_SET + 30:
                if level in ("FIB_0.236", "FIB_0.618"):
                    continue

        elif variant == "adaptive_caps":
            # Start with half caps for first 30 min.
            # If first 3 trades all wins -> restore full caps.
            # If any loss -> keep half caps for another 30 min.
            if not full_caps_restored and et < adaptive_half_until_et:
                day_caps = dict(BASE_CAPS)
                if dc.date.weekday() == 0:
                    day_caps = {k: v * 2 for k, v in day_caps.items()}
                half_cap = max(1, day_caps.get(level, 99) // 2)
                if level_counts[level] >= half_cap:
                    continue
            elif not full_caps_restored and et >= adaptive_half_until_et:
                # Timer expired, restore full caps
                full_caps_restored = True

        # --- Standard $100 daily loss limit ---
        if cum_pnl <= -100:
            stopped = True
            break

        # --- Accept trade ---
        filtered.append(t)
        level_counts[level] += 1
        cum_pnl += pnl

        # --- Post-trade state updates ---
        if pnl < 0:
            losses_so_far += 1
            if not has_lost:
                has_lost = True
                first_loss_et = et
            consecutive_wins_after_loss = 0

            # Adaptive caps: loss extends half-cap period
            if variant == "adaptive_caps" and not full_caps_restored:
                adaptive_loss_seen = True
                adaptive_half_until_et = et + 30
        else:
            wins_so_far += 1
            if has_lost:
                consecutive_wins_after_loss += 1

            # Adaptive caps: 3 consecutive wins from start -> restore
            if variant == "adaptive_caps" and not full_caps_restored:
                total_accepted = len(filtered)
                if total_accepted >= 3 and not adaptive_loss_seen:
                    full_caps_restored = True

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
    path = results_dir / f"first_hour_v1_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
