"""Test condition-based parameter adjustments for losing day patterns.

Instead of binary skip/don't-skip, adjusts trading parameters based
on market conditions: tighter caps, fewer levels, higher vol filter.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_pattern_filters.py
"""
import os, sys, time, datetime
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.results import compute_stats, save_trades_data

# Baseline config
BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH"}
INTERIOR_FIBS = {"FIB_0.236", "FIB_0.618", "FIB_0.764"}

# Global data
_DATES = None
_CACHES = None
_ARRAYS = None
_VIX = None
_IB_AVGS = None
_GAPS = None  # {date: {gap_pct, gap_pts, unfilled_at_ib}}
_ECON = None  # {date: "FOMC"/"CPI"/"NFP"}

FOMC_DATES = {datetime.date.fromisoformat(d) for d in [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
]}


def _load_vix():
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start="2024-12-01", end="2026-05-01", progress=False)
        return {d.date(): float(row["Close"].iloc[0]) for d, row in vix.iterrows()}
    except Exception as e:
        print(f"  WARNING: Could not load VIX: {e}")
        return {}


def _compute_ib_avgs(dates, caches):
    ib_ranges = []
    avgs = {}
    for i, d in enumerate(dates):
        ib_range = caches[d].ibh - caches[d].ibl
        ib_ranges.append(ib_range)
        avgs[d] = np.mean(ib_ranges[max(0,i-20):i]) if i > 0 else ib_range
    return avgs


def _compute_gaps(dates, caches):
    """Compute gap size and whether it was filled during IB."""
    gaps = {}
    prev_close = None
    for date in dates:
        dc = caches[date]
        today_open = float(dc.full_prices[0])
        if prev_close is not None and prev_close > 0:
            gap_pts = today_open - prev_close
            gap_pct = gap_pts / prev_close * 100
            ib_range = dc.ibh - dc.ibl
            # Gap unfilled: gap up → IB low never reached prev close.
            # Gap down → IB high never reached prev close.
            if gap_pts > 0:
                unfilled = dc.ibl > prev_close  # IB low stayed above prev close
            elif gap_pts < 0:
                unfilled = dc.ibh < prev_close  # IB high stayed below prev close
            else:
                unfilled = False
            gaps[date] = {
                "gap_pct": gap_pct,
                "gap_pts": gap_pts,
                "abs_gap_pct": abs(gap_pct),
                "unfilled": unfilled,
                "gap_vs_ib": abs(gap_pts) / ib_range if ib_range > 0 else 0,
            }
        today_close = float(dc.full_prices[-1])
        prev_close = today_close
    return gaps


def _halve_caps(caps):
    return {k: max(1, v // 2) for k, v in caps.items()}


def _get_day_params(date, variant_name):
    """Return (exclude_levels, max_per_level_map, vol_filter_override) for a day.

    Returns baseline params if no condition matches, or adjusted params
    based on the variant's rules.
    """
    dc = _CACHES[date]
    ib_range = dc.ibh - dc.ibl
    vix = _VIX.get(date)
    dow = date.weekday()
    ib_avg = _IB_AVGS.get(date, ib_range)

    exclude = set(BASE_EXCLUDE)
    caps = dict(BASE_CAPS)
    vol_override = None  # None = use default 0.0015

    if variant_name == "baseline":
        pass

    # --- VIX < 15 adjustments ---
    elif variant_name == "vix15_raise_vol":
        if vix is not None and vix < 15:
            vol_override = 0.0025  # 0.25% instead of 0.15%

    elif variant_name == "vix15_halve_caps":
        if vix is not None and vix < 15:
            caps = _halve_caps(caps)

    elif variant_name == "vix15_ext_only":
        if vix is not None and vix < 15:
            exclude = exclude | INTERIOR_FIBS

    # --- Narrow IB adjustments ---
    elif variant_name == "ib100_halve_caps":
        if ib_range < 100:
            caps = _halve_caps(caps)

    elif variant_name == "ib100_ext_only":
        if ib_range < 100:
            exclude = exclude | INTERIOR_FIBS

    elif variant_name == "ib100_raise_vol":
        if ib_range < 100:
            vol_override = 0.0025

    # --- Thursday ---
    elif variant_name == "thu_halve_caps":
        if dow == 3:
            caps = _halve_caps(caps)

    # --- Combos ---
    elif variant_name == "vix15_vol_ib100_caps":
        if vix is not None and vix < 15:
            vol_override = 0.0025
        if ib_range < 100:
            caps = _halve_caps(caps)

    elif variant_name == "vix15_ext_ib100_ext":
        if vix is not None and vix < 15:
            exclude = exclude | INTERIOR_FIBS
        if ib_range < 100:
            exclude = exclude | INTERIOR_FIBS

    # --- Opening gap adjustments ---
    elif variant_name == "gap05_halve_caps":
        gap = _GAPS.get(date)
        if gap and gap["abs_gap_pct"] > 0.5:
            caps = _halve_caps(caps)

    elif variant_name == "gap05_raise_vol":
        gap = _GAPS.get(date)
        if gap and gap["abs_gap_pct"] > 0.5:
            vol_override = 0.0025

    elif variant_name == "gap1_ext_only":
        gap = _GAPS.get(date)
        if gap and gap["abs_gap_pct"] > 1.0:
            exclude = exclude | INTERIOR_FIBS

    elif variant_name == "gap_unfilled_halve":
        gap = _GAPS.get(date)
        if gap and gap["unfilled"] and gap["abs_gap_pct"] > 0.3:
            caps = _halve_caps(caps)

    elif variant_name == "gap_unfilled_ext_only":
        gap = _GAPS.get(date)
        if gap and gap["unfilled"] and gap["abs_gap_pct"] > 0.3:
            exclude = exclude | INTERIOR_FIBS

    elif variant_name == "gap_gt_ib":
        gap = _GAPS.get(date)
        if gap and gap["gap_vs_ib"] > 1.0:
            caps = _halve_caps(caps)

    # --- First loss / FOMC ---
    elif variant_name in ("first_loss_cap1", "first_loss_raise_vol"):
        pass  # handled as post-filter in _run_one

    elif variant_name == "fomc_halve":
        if date in FOMC_DATES:
            caps = _halve_caps(caps)

    # --- Aggressive: increase on good conditions ---
    elif variant_name == "mon_double_caps":
        if dow == 0:  # Monday
            caps = {k: v * 2 for k, v in caps.items()}

    elif variant_name == "ib_sweet_double":
        if 150 <= ib_range <= 250:
            caps = {k: v * 2 for k, v in caps.items()}

    # --- Combos ---
    elif variant_name == "vix15_vol_ib100_ext":
        if vix is not None and vix < 15:
            vol_override = 0.0025
        if ib_range < 100:
            exclude = exclude | INTERIOR_FIBS

    elif variant_name == "gap_unfilled_ib100":
        gap = _GAPS.get(date)
        if gap and gap["unfilled"] and gap["abs_gap_pct"] > 0.3:
            caps = _halve_caps(caps)
        if ib_range < 100:
            caps = _halve_caps(caps)

    elif variant_name == "gap_unfilled_vix15":
        gap = _GAPS.get(date)
        if gap and gap["unfilled"] and gap["abs_gap_pct"] > 0.3:
            caps = _halve_caps(caps)
        if vix is not None and vix < 15:
            vol_override = 0.0025

    elif variant_name == "unfilled_narrow_ib":
        gap = _GAPS.get(date)
        if gap and gap["unfilled"] and gap["abs_gap_pct"] > 0.3 and ib_range < 100:
            exclude = exclude | INTERIOR_FIBS  # most dangerous combo

    elif variant_name == "adr15x_steep60":
        pass  # both handled as post-filters in _run_one

    elif variant_name == "all_defensive":
        if vix is not None and vix < 15:
            vol_override = 0.0025
        if ib_range < 100:
            exclude = exclude | INTERIOR_FIBS
        gap = _GAPS.get(date)
        if gap and gap["unfilled"] and gap["abs_gap_pct"] > 0.3:
            caps = _halve_caps(caps)
        if date in FOMC_DATES:
            caps = _halve_caps(caps)

    elif variant_name == "all_def_agg":
        # Defensive
        if vix is not None and vix < 15:
            vol_override = 0.0025
        if ib_range < 100:
            exclude = exclude | INTERIOR_FIBS
        gap = _GAPS.get(date)
        if gap and gap["unfilled"] and gap["abs_gap_pct"] > 0.3:
            caps = _halve_caps(caps)
        if date in FOMC_DATES:
            caps = _halve_caps(caps)
        # Aggressive
        if dow == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        if 150 <= ib_range <= 250:
            caps = {k: min(v * 2, 12) for k, v in caps.items()}

    return exclude, caps, vol_override


def _get_adr_threshold(variant_name):
    """Returns ADR multiplier threshold for mid-day stop, or None."""
    if variant_name in ("adr_15x_stop_interior", "adr15x_steep60",
                         "all_defensive", "all_def_agg"):
        return 1.5
    return None


def _get_steep_move_pct(variant_name):
    """Returns session move threshold as fraction of IB range, or None."""
    if variant_name in ("steep_60pct", "adr15x_steep60",
                         "all_defensive", "all_def_agg"):
        return 0.6
    return None


VARIANTS = [
    ("Baseline", "baseline"),
    # Defensive: reduce on bad conditions
    ("VIX<15: vol 0.25%", "vix15_raise_vol"),
    ("IB<100: halve caps", "ib100_halve_caps"),
    ("IB<100: ext only", "ib100_ext_only"),
    ("ADR>1.5x: stop interior", "adr_15x_stop_interior"),
    ("Steep>60%: skip interior", "steep_60pct"),
    ("Gap>0.5%: halve caps", "gap05_halve_caps"),
    ("Gap>1%: ext only", "gap1_ext_only"),
    ("Gap unfilled: halve caps", "gap_unfilled_halve"),
    ("Gap unfilled: ext only", "gap_unfilled_ext_only"),
    ("Gap > IB range: halve", "gap_gt_ib"),
    ("1st loss: caps to 1", "first_loss_cap1"),
    ("1st loss: raise vol 0.25%", "first_loss_raise_vol"),
    ("FOMC: halve caps", "fomc_halve"),
    # Aggressive: increase on good conditions
    ("Mon: double caps", "mon_double_caps"),
    ("IB 150-250: double caps", "ib_sweet_double"),
    # Combos
    ("VIX<15 vol + IB<100 ext", "vix15_vol_ib100_ext"),
    ("Gap unfilled + IB<100", "gap_unfilled_ib100"),
    ("Unfilled + narrow IB", "unfilled_narrow_ib"),
    ("ADR>1.5x + steep>60%", "adr15x_steep60"),
    ("All defensive", "all_defensive"),
    ("All defensive + aggressive", "all_def_agg"),
]


def _run_one(args):
    name, variant = args
    all_trades = []
    streak = (0, 0)
    adr_threshold = _get_adr_threshold(variant)
    steep_pct = _get_steep_move_pct(variant)

    for date in _DATES:
        dc = _CACHES[date]
        exclude, caps, vol_override = _get_day_params(date, variant)
        ib_range = dc.ibh - dc.ibl

        trades, streak = simulate_day(
            dc, _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=exclude,
            include_ibl=False, include_vwap=False,
            vol_filter_pct=vol_override if vol_override is not None else 0.0015,
        )

        # Post-filter: first loss response — after first loss, limit remaining trades.
        if variant == "first_loss_cap1" and trades:
            first_loss_idx = None
            for i, t in enumerate(trades):
                if t.pnl_usd < 0:
                    first_loss_idx = i
                    break
            if first_loss_idx is not None:
                post_loss_counts = defaultdict(int)
                filtered = trades[:first_loss_idx + 1]
                for t in trades[first_loss_idx + 1:]:
                    if post_loss_counts[t.level] < 1:
                        filtered.append(t)
                        post_loss_counts[t.level] += 1
                trades = filtered

        if variant == "first_loss_raise_vol" and trades:
            first_loss_idx = None
            for i, t in enumerate(trades):
                if t.pnl_usd < 0:
                    first_loss_idx = i
                    break
            if first_loss_idx is not None:
                filtered = trades[:first_loss_idx + 1]
                for t in trades[first_loss_idx + 1:]:
                    price = dc.full_prices[t.entry_idx]
                    range_pct = t.factors.range_30m / price if price > 0 else 0
                    if range_pct >= 0.0025:
                        filtered.append(t)
                trades = filtered

        # Post-filter: ADR consumed — stop interior fib trades when session range > threshold × IB.
        if adr_threshold is not None and trades:
            filtered = []
            for t in trades:
                start = dc.post_ib_start_idx
                prices_so_far = dc.full_prices[start:t.entry_idx + 1]
                if len(prices_so_far) > 0:
                    session_range = float(np.max(prices_so_far) - np.min(prices_so_far))
                    if session_range > adr_threshold * ib_range and t.level in INTERIOR_FIBS:
                        continue
                filtered.append(t)
            trades = filtered

        # Post-filter: steep move — skip INTERIOR FIB entries where abs(session_move) > pct × IB range.
        if steep_pct is not None and trades:
            filtered = []
            for t in trades:
                if t.level in INTERIOR_FIBS and abs(t.factors.session_move) > steep_pct * ib_range:
                    continue
                filtered.append(t)
            trades = filtered

        all_trades.extend(trades)

    # Use compute_stats for comprehensive metrics.
    stats = compute_stats(all_trades, len(_DATES), list(_DATES))
    stats["name"] = name
    return stats


def main():
    global _DATES, _CACHES, _ARRAYS, _VIX, _IB_AVGS, _GAPS
    t0 = time.time()

    print("Loading data...", flush=True)
    _DATES, _CACHES = load_all_days()
    print(f"Loaded {len(_DATES)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    _ARRAYS = {d: precompute_arrays(_CACHES[d]) for d in _DATES}

    print("Loading VIX...", flush=True)
    _VIX = _load_vix()
    print(f"VIX: {len(_VIX)} days", flush=True)

    _IB_AVGS = _compute_ib_avgs(_DATES, _CACHES)

    print("Computing gaps...", flush=True)
    _GAPS = _compute_gaps(_DATES, _CACHES)
    unfilled = sum(1 for g in _GAPS.values() if g["unfilled"])
    print(f"Gaps: {len(_GAPS)} days, {unfilled} unfilled", flush=True)

    n_variants = len(VARIANTS)
    n_workers = 3
    print(f"Running {n_variants} variants across {n_workers} workers...", flush=True)

    args = list(VARIANTS)
    with Pool(n_workers) as pool:
        results = pool.map(_run_one, args)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    baseline = results[0]
    b_pnl = baseline["pnl_per_day"]

    # Summary table
    print("=" * 130)
    print(f"{'Variant':<30} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'W%days':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'Q4':>7} {'vs base':>7}")
    print("-" * 130)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        q4 = r.get("quarterly_pnl_per_day", {}).get("Q4_newest", 0)
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        w_pct = r.get("winning_days_pct", 0)
        l100 = r.get("days_below_neg100", 0)
        print(
            f"{r['name']:<30} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {w_pct:>5.1f}% {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {q4:>+7.2f} {diff:>+7.2f}"
        )

    # Per-level breakdown for top 3 variants
    print()
    best_pnl = max(results, key=lambda r: r["pnl_per_day"])
    fewest_bad = min(results, key=lambda r: r.get("days_below_neg100", 999))
    best_recent = max(results, key=lambda r: r.get("recent_60d_pnl_per_day", -999))

    print(f"Best $/day:        {best_pnl['name']} at ${best_pnl['pnl_per_day']:.2f}/day")
    print(f"Fewest -$100 days: {fewest_bad['name']} with {fewest_bad.get('days_below_neg100', '?')} days")
    print(f"Best recent 60d:   {best_recent['name']} at ${best_recent.get('recent_60d_pnl_per_day', 0):.2f}/day")

    # Save all results as JSON
    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"pattern_filters_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {path}")


if __name__ == "__main__":
    main()
