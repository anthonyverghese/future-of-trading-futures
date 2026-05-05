"""For each trade in the deployed-config pickle, compute approach features
(the 6pt window between human-alert threshold ~7pt and bot-entry ~1pt)
and look for buckets where slippage-adjusted $/tr differs meaningfully.

The hypothesis is that fast approaches break through the level (poor
fills + lower WR), while slow/decelerating approaches reverse cleanly.
The same hypothesis was tested in no-slippage backtest (2026-04-26) and
showed only ±2-3% WR delta — judged "weak signal." But that was on WR.
Slippage shifts the metric to fill-adjusted $/trade, which compounds:
- Strong-rejection scenarios fill at limit (worst fill) AND have small
  wins, so any bucket with bad fills shows up in $/tr
- Slow approaches let the limit fill closer to line (better fill)

Approach features computed per trade:
- approach_duration_secs   : time from 7pt entry to 1pt entry
- approach_velocity_pps    : 6.0 / approach_duration_secs (pts per sec)
- approach_tick_count      : number of ticks during 7pt→1pt window
- approach_deceleration    : ratio of late-half velocity to early-half
                             (>1 = accelerating, <1 = decelerating)
- approach_max_retrace     : max price retrace during approach (back
                             toward 7pt) — a "clean approach" doesn't
                             retrace much

Output: per-level table of slippage-adjusted $/tr by feature bucket,
sample sizes, and a flag where the bucket shows >$1/tr difference vs
the level mean.

Filters applied to match the deployed C config:
- buffer = 1.0pt (slippage modeled)
- exclude FIB_EXT_LO_1.272
- only include trades that would have filled (87% fill rate)
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from mnq_alerts.backtest.data import load_all_days
from mnq_alerts.backtest.experiments.buffer_sweep_v1 import (
    simulate_fill, adjusted_pnl, MULTIPLIER, FEE_USD,
)

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
APPROACH_THRESHOLD = 7.0   # human alert threshold (pts from line)
ENTRY_THRESHOLD = 1.0      # bot entry threshold

EXCLUDED_LEVELS = {"FIB_EXT_LO_1.272"}  # match deployed C config


def compute_approach_features(
    full_prices: np.ndarray,
    full_ts_ns: np.ndarray,
    entry_idx: int,
    line: float,
    direction: str,
) -> dict:
    """Walk backward from entry_idx to find when price first crossed
    APPROACH_THRESHOLD pts from line on the same side as the entry.

    Returns dict of approach features, or None if the approach started
    too far back (likely a different oscillation, not a fresh approach).
    """
    if direction == "down":
        # SELL: bot entered at price near line (within 1pt above OR below);
        # the approach is the move that brought price from far-below
        # toward line OR from far-above toward line. The 7pt mark
        # depends on which side. We track the most recent crossing of
        # 7pt threshold on either side.
        target_dist = APPROACH_THRESHOLD
    else:
        target_dist = APPROACH_THRESHOLD

    # Walk back. Stop when we find a tick where |price - line| >= 7,
    # or after a max look-back.
    max_lookback_ticks = 50_000  # ~10 min at 80 ticks/sec
    start_idx = max(0, entry_idx - max_lookback_ticks)
    found_idx = None
    for i in range(entry_idx - 1, start_idx - 1, -1):
        if abs(full_prices[i] - line) >= target_dist:
            found_idx = i
            break
    if found_idx is None:
        return None

    # Fresh approach defined as: from found_idx forward to entry_idx,
    # price stays within the 7pt threshold (no exit and re-entry).
    # Actually we want the first crossing AFTER found_idx into the 7pt
    # threshold — which is at found_idx + 1 (or the next tick that
    # gets within 7pt). Use that as approach_start.
    approach_start = found_idx
    approach_dur_ns = int(full_ts_ns[entry_idx]) - int(full_ts_ns[approach_start])
    approach_dur_secs = approach_dur_ns / 1e9
    if approach_dur_secs <= 0.001:
        return None

    velocity_pps = 6.0 / approach_dur_secs
    tick_count = entry_idx - approach_start

    # Deceleration: compare second half velocity to first half
    half_idx = approach_start + tick_count // 2
    if 0 < half_idx - approach_start and 0 < entry_idx - half_idx:
        first_half_dur = (full_ts_ns[half_idx] - full_ts_ns[approach_start]) / 1e9
        second_half_dur = (full_ts_ns[entry_idx] - full_ts_ns[half_idx]) / 1e9
        first_half_pts = abs(full_prices[half_idx] - full_prices[approach_start])
        second_half_pts = abs(full_prices[entry_idx] - full_prices[half_idx])
        first_v = first_half_pts / first_half_dur if first_half_dur > 0.001 else 0
        second_v = second_half_pts / second_half_dur if second_half_dur > 0.001 else 0
        decel = second_v / first_v if first_v > 0.001 else 1.0
    else:
        decel = 1.0

    # Max retrace: during the approach, did price ever move back away
    # from the line by more than X pts? Measures messiness of approach.
    seg = full_prices[approach_start:entry_idx + 1]
    if direction == "down":
        # price moving toward higher line, retrace = drop
        running_max = np.maximum.accumulate(seg)
        retrace = float(np.max(running_max - seg))
    else:
        running_min = np.minimum.accumulate(seg)
        retrace = float(np.max(seg - running_min))

    return {
        "approach_dur_secs": approach_dur_secs,
        "approach_velocity_pps": velocity_pps,
        "approach_tick_count": tick_count,
        "approach_deceleration": decel,
        "approach_max_retrace": retrace,
    }


def main() -> None:
    print("Loading pickle...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    print(f"  {len(rows)} trades over {len({r['date'] for r in rows})} days",
          flush=True)

    print("\nLoading day caches (~3 min)...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("\nApplying slippage model + computing approach features...",
          flush=True)
    enriched: list[dict] = []
    skipped_no_fill = 0
    skipped_no_approach = 0
    skipped_excluded = 0
    t1 = time.time()
    for ri, r in enumerate(rows):
        if r["level"] in EXCLUDED_LEVELS:
            skipped_excluded += 1
            continue
        dc = caches[r["date"]]
        # Apply 1.0pt buffer slippage model
        fill_idx, fill_price = simulate_fill(
            dc.full_prices, dc.full_ts_ns,
            entry_ns=r["entry_ns"], direction=r["direction"],
            line=r["line_price"], buffer=1.0, latency_ms=100.0,
        )
        if fill_idx is None:
            skipped_no_fill += 1
            continue
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        # Compute approach features (look back to 7pt from line)
        # Use the original entry_idx (where bot fired at 1pt) as reference
        entry_idx = int(np.searchsorted(dc.full_ts_ns, r["entry_ns"]))
        feats = compute_approach_features(
            dc.full_prices, dc.full_ts_ns, entry_idx,
            r["line_price"], r["direction"],
        )
        if feats is None:
            skipped_no_approach += 1
            continue
        enriched.append({
            "level": r["level"],
            "direction": r["direction"],
            "outcome": r["outcome"],
            "pnl_slip": new_pnl,
            **feats,
        })
        if (ri + 1) % 1000 == 0:
            print(f"  {ri+1}/{len(rows)}  enriched={len(enriched)}  "
                  f"({time.time()-t1:.0f}s)", flush=True)

    print(f"\nEnriched {len(enriched)} trades  "
          f"(excluded: {skipped_excluded}, no-fill: {skipped_no_fill}, "
          f"no-approach: {skipped_no_approach})",
          flush=True)

    # Bucket analysis: per level, per feature
    levels = sorted({t["level"] for t in enriched})
    days = len({(t.get("date"), t.get("level")) for t in enriched})  # rough

    def report_buckets(feature: str, bins: list[float], unit: str) -> None:
        print(f"\n{'='*78}")
        print(f"FEATURE: {feature} ({unit})")
        print('='*78)
        bin_labels = [f"<{bins[0]}"] + [
            f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)
        ] + [f">{bins[-1]}"]
        print(f"  {'Level':<22} ", end="")
        for label in bin_labels:
            print(f"{label:>10}", end="")
        print()
        for lv in levels:
            lv_trades = [t for t in enriched if t["level"] == lv]
            if not lv_trades:
                continue
            print(f"  {lv:<22} ", end="")
            level_mean = sum(t["pnl_slip"] for t in lv_trades) / len(lv_trades)
            for i, _ in enumerate(bin_labels):
                if i == 0:
                    bucket = [t for t in lv_trades if t[feature] < bins[0]]
                elif i == len(bin_labels) - 1:
                    bucket = [t for t in lv_trades if t[feature] >= bins[-1]]
                else:
                    bucket = [t for t in lv_trades
                              if bins[i-1] <= t[feature] < bins[i]]
                if not bucket:
                    print(f"{'·':>10}", end="")
                    continue
                bucket_pnl = sum(t["pnl_slip"] for t in bucket) / len(bucket)
                marker = "*" if abs(bucket_pnl - level_mean) > 1.0 else " "
                print(f"{bucket_pnl:>+8.2f}{marker} ", end="")
            print(f"  (mean ${level_mean:+.2f}/tr, n={len(lv_trades)})")
        print(f"\n  Counts per bucket:")
        print(f"  {'Level':<22} ", end="")
        for label in bin_labels:
            print(f"{label:>10}", end="")
        print()
        for lv in levels:
            lv_trades = [t for t in enriched if t["level"] == lv]
            if not lv_trades:
                continue
            print(f"  {lv:<22} ", end="")
            for i, _ in enumerate(bin_labels):
                if i == 0:
                    n = sum(1 for t in lv_trades if t[feature] < bins[0])
                elif i == len(bin_labels) - 1:
                    n = sum(1 for t in lv_trades if t[feature] >= bins[-1])
                else:
                    n = sum(1 for t in lv_trades
                            if bins[i-1] <= t[feature] < bins[i])
                print(f"{n:>10d}", end="")
            print()

    # Run bucket analyses on each feature
    report_buckets("approach_velocity_pps", [0.05, 0.10, 0.20, 0.50, 1.0], "pts/sec")
    report_buckets("approach_dur_secs", [10, 30, 60, 120, 300], "secs")
    report_buckets("approach_tick_count", [50, 200, 500, 1000, 2000], "ticks")
    report_buckets("approach_deceleration", [0.3, 0.6, 1.0, 1.5, 2.5], "ratio")
    report_buckets("approach_max_retrace", [1, 2, 3, 5, 7], "pts")

    # Also: a simple "best filter candidate" summary — for each feature, find the
    # cutoff that maximizes total $/day if we KEEP trades on the high side.
    print(f"\n{'='*78}")
    print("BEST-FILTER CANDIDATES (keep trades meeting the threshold)")
    print('='*78)
    days_count = len({(t.get('approach_dur_secs', 0) > 0,) for t in enriched})  # rough proxy
    # actually use date count
    # Reload to get dates - simpler: count from rows
    days_count = len({r['date'] for r in rows})

    for feature, direction in [
        ("approach_velocity_pps", "below"),  # slow approach is better
        ("approach_dur_secs", "above"),       # long approach is better
        ("approach_max_retrace", "above"),    # clean approach has small retrace
        ("approach_deceleration", "below"),   # decelerating is better
    ]:
        vals = sorted(set(t[feature] for t in enriched))
        candidate_cutoffs = vals[::max(1, len(vals)//30)]  # ~30 cutoffs to test
        best_cutoff = None
        best_pnl_per_day = sum(t["pnl_slip"] for t in enriched) / days_count
        baseline = best_pnl_per_day
        for c in candidate_cutoffs:
            if direction == "below":
                kept = [t for t in enriched if t[feature] < c]
            else:
                kept = [t for t in enriched if t[feature] >= c]
            if len(kept) < 1000:  # need enough sample
                continue
            kept_per_day = sum(t["pnl_slip"] for t in kept) / days_count
            if kept_per_day > best_pnl_per_day:
                best_pnl_per_day = kept_per_day
                best_cutoff = c
        if best_cutoff is None:
            print(f"  {feature} ({direction}): no improvement found "
                  f"vs baseline ${baseline:+.2f}/day")
        else:
            kept = ([t for t in enriched if t[feature] < best_cutoff]
                    if direction == "below"
                    else [t for t in enriched if t[feature] >= best_cutoff])
            print(f"  {feature} ({direction} {best_cutoff:.3f}): "
                  f"${best_pnl_per_day:+.2f}/day (baseline ${baseline:+.2f}, "
                  f"+${best_pnl_per_day - baseline:.2f}), "
                  f"keeps {len(kept)}/{len(enriched)} trades")


if __name__ == "__main__":
    main()
