"""Backtest data validation and diagnostics.

Run before any backtest to catch data quality issues that could
silently corrupt results. Prints warnings for anything suspicious.

Usage:
    from backtest.validate import validate_all
    valid_days, day_caches = load_all_days()
    validate_all(valid_days, day_caches)
"""

from __future__ import annotations

import datetime
import os

import numpy as np
import pytz

_ET = pytz.timezone("America/New_York")

# Expected trading day count by month (approximate, excludes holidays).
_EXPECTED_DAYS_PER_MONTH = 21


def validate_all(
    valid_days: list[datetime.date],
    day_caches: dict,
    arrays: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run all validation checks. Returns a summary dict.

    Args:
        valid_days: sorted list of trading dates
        day_caches: date → DayCache mapping
        arrays: date → DayArrays mapping (optional, for factor validation)
        verbose: print warnings as they're found
    """
    issues = []
    stats = {
        "total_days": len(valid_days),
        "date_range": f"{valid_days[0]} to {valid_days[-1]}" if valid_days else "none",
        "issues": issues,
    }

    if not valid_days:
        issues.append("ERROR: No valid days loaded")
        return stats

    # 1. Check for gaps in trading days.
    _check_date_gaps(valid_days, issues, verbose)

    # 2. Validate each day's data quality.
    tiny_days = []
    bad_ib_days = []
    bad_vwap_days = []
    zero_range_days = []
    total_ticks = 0
    total_post_ib_ticks = 0

    for date in valid_days:
        dc = day_caches[date]

        # Tick count.
        n_full = len(dc.full_prices)
        n_post_ib = len(dc.post_ib_prices)
        total_ticks += n_full
        total_post_ib_ticks += n_post_ib

        # Too few ticks (< 10K is suspicious for MNQ RTH).
        if n_full < 10000:
            tiny_days.append((date, n_full))

        # IB levels: IBH should be > IBL.
        if dc.ibh <= dc.ibl:
            bad_ib_days.append((date, dc.ibh, dc.ibl))

        # IB range too small (< 5 pts is suspicious).
        ib_range = dc.ibh - dc.ibl
        if ib_range < 5.0:
            zero_range_days.append((date, ib_range))

        # VWAP should be between day low and day high.
        if dc.post_ib_vwaps is not None and len(dc.post_ib_vwaps) > 0:
            vwap_mid = float(dc.post_ib_vwaps[len(dc.post_ib_vwaps) // 2])
            day_low = float(np.min(dc.full_prices))
            day_high = float(np.max(dc.full_prices))
            if vwap_mid < day_low or vwap_mid > day_high:
                bad_vwap_days.append((date, vwap_mid, day_low, day_high))

        # post_ib_start_idx should be within bounds.
        if dc.post_ib_start_idx >= n_full:
            issues.append(
                f"ERROR: {date} post_ib_start_idx={dc.post_ib_start_idx} "
                f">= full_prices length {n_full}"
            )

        # Timestamps should be monotonically increasing.
        if n_full > 1:
            ts_diff = np.diff(dc.full_ts_ns)
            if np.any(ts_diff < 0):
                issues.append(f"ERROR: {date} has non-monotonic timestamps")

    if tiny_days:
        issues.append(
            f"WARNING: {len(tiny_days)} days with < 10K ticks: "
            + ", ".join(f"{d}({n})" for d, n in tiny_days[:5])
            + ("..." if len(tiny_days) > 5 else "")
        )

    if bad_ib_days:
        issues.append(
            f"ERROR: {len(bad_ib_days)} days with IBH <= IBL: "
            + ", ".join(f"{d}(H={h:.0f},L={l:.0f})" for d, h, l in bad_ib_days[:5])
        )

    if zero_range_days:
        issues.append(
            f"WARNING: {len(zero_range_days)} days with IB range < 5 pts: "
            + ", ".join(f"{d}({r:.1f})" for d, r in zero_range_days[:5])
        )

    if bad_vwap_days:
        issues.append(
            f"WARNING: {len(bad_vwap_days)} days with VWAP outside day range: "
            + ", ".join(f"{d}" for d, *_ in bad_vwap_days[:5])
        )

    stats["total_ticks"] = total_ticks
    stats["total_post_ib_ticks"] = total_post_ib_ticks
    stats["avg_ticks_per_day"] = total_ticks // len(valid_days)
    stats["avg_post_ib_ticks"] = total_post_ib_ticks // len(valid_days)
    stats["tiny_days"] = len(tiny_days)
    stats["bad_ib_days"] = len(bad_ib_days)

    # 3. Validate precomputed arrays if provided.
    if arrays:
        _check_arrays(valid_days, day_caches, arrays, issues, verbose)

    # 4. Check .npz cache freshness.
    _check_array_cache(valid_days, issues, verbose)

    # Print summary.
    if verbose:
        print(f"\n  === BACKTEST DATA VALIDATION ===")
        print(f"  Days: {stats['total_days']} ({stats['date_range']})")
        print(f"  Ticks: {stats['total_ticks']:,} total, {stats['avg_ticks_per_day']:,}/day avg")
        print(f"  Post-IB ticks: {stats['total_post_ib_ticks']:,} total, {stats['avg_post_ib_ticks']:,}/day avg")
        if issues:
            print(f"\n  ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"    {issue}")
        else:
            print(f"\n  All checks passed ✓")
        print()

    return stats


def _check_date_gaps(
    valid_days: list[datetime.date],
    issues: list[str],
    verbose: bool,
) -> None:
    """Check for unexpected gaps in the trading day sequence."""
    gaps = []
    for i in range(1, len(valid_days)):
        prev = valid_days[i - 1]
        curr = valid_days[i]
        delta = (curr - prev).days
        # More than 5 calendar days = possible missing data
        # (weekends = 2-3 days, holidays = up to 4)
        if delta > 5:
            gaps.append((prev, curr, delta))

    if gaps:
        issues.append(
            f"WARNING: {len(gaps)} date gap(s) > 5 days: "
            + ", ".join(f"{p}→{c} ({d}d)" for p, c, d in gaps)
        )


def _check_arrays(
    valid_days: list[datetime.date],
    day_caches: dict,
    arrays: dict,
    issues: list[str],
    verbose: bool,
) -> None:
    """Validate precomputed factor arrays."""
    for date in valid_days:
        dc = day_caches[date]
        arr = arrays.get(date)
        if arr is None:
            issues.append(f"ERROR: {date} missing precomputed arrays")
            continue

        n = len(dc.full_prices)
        for name, a in [
            ("tick_rates", arr.tick_rates),
            ("range_30m", arr.range_30m_pts),
            ("approach_speed", arr.approach_speed),
            ("tick_density", arr.tick_density),
            ("et_mins", arr.et_mins),
            ("session_move", arr.session_move),
        ]:
            if len(a) != n:
                issues.append(
                    f"ERROR: {date} {name} length {len(a)} != "
                    f"full_prices length {n}"
                )
            if np.any(np.isnan(a)):
                issues.append(f"WARNING: {date} {name} contains NaN values")
            if np.any(np.isinf(a)):
                issues.append(f"WARNING: {date} {name} contains Inf values")

        # Tick rate should be > 0 for post-IB ticks.
        s = dc.post_ib_start_idx
        post_ib_tr = arr.tick_rates[s:]
        if len(post_ib_tr) > 100 and np.all(post_ib_tr == 0):
            issues.append(f"WARNING: {date} tick_rates all zero post-IB")

        # ET minutes should be in 630-960 range (10:30-16:00 ET).
        post_ib_em = arr.et_mins[s:]
        if len(post_ib_em) > 0:
            min_em = int(np.min(post_ib_em))
            max_em = int(np.max(post_ib_em))
            if min_em < 600 or max_em > 970:
                issues.append(
                    f"WARNING: {date} ET minutes out of range: "
                    f"{min_em}-{max_em} (expected 630-960)"
                )


def _check_array_cache(
    valid_days: list[datetime.date],
    issues: list[str],
    verbose: bool,
) -> None:
    """Check if .npz cache files exist and are reasonably fresh."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".array_cache")
    if not os.path.exists(cache_dir):
        issues.append("INFO: No .npz array cache directory — arrays will be computed from scratch")
        return

    cached = set(
        f.replace(".npz", "")
        for f in os.listdir(cache_dir)
        if f.endswith(".npz")
    )
    missing = [str(d) for d in valid_days if str(d) not in cached]
    if missing:
        issues.append(
            f"INFO: {len(missing)} days missing from .npz cache "
            f"(will be computed): {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else "")
        )

    # Check if any cached files are from a different code version
    # by checking file size consistency.
    sizes = []
    for f in os.listdir(cache_dir):
        if f.endswith(".npz"):
            sizes.append(os.path.getsize(os.path.join(cache_dir, f)))
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        tiny = [s for s in sizes if s < avg_size * 0.01]  # <1% = truly corrupt
        if tiny:
            issues.append(
                f"WARNING: {len(tiny)} .npz cache files are suspiciously "
                f"small (< 1% of average). May be corrupt — consider "
                f"deleting .array_cache/ and recomputing."
            )
