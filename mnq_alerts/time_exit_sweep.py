"""
time_exit_sweep.py — Sweep time-based exits to find the optimal hold period.

Instead of +8 target / -20 stop, exit after N minutes and measure P&L.
Tests: 1, 2, 3, 5, 7, 10, 15 minute exits.

Also tests hybrid approaches: time exit + trailing stop.

Usage:
    python time_exit_sweep.py
"""

from __future__ import annotations

import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from targeted_backtest import (
    ALERT_THRESHOLD,
    EXIT_THRESHOLD,
    IB_END,
    MARKET_CLOSE,
    MARKET_OPEN,
    _run_zone_numpy,
    load_cached_days,
    load_day,
)

# Time horizons to test (minutes)
TIME_EXITS = [1, 2, 3, 5, 7, 10, 15]

# Also test hybrid: time exit with a protective stop
HYBRID_STOPS = [10, 15, 20, 30]  # max loss before time exit

# Current production baseline
TARGET_POINTS = 8.0
STOP_POINTS = 20.0
HIT_THRESHOLD = 1.0
WINDOW_SECS = 15 * 60


def preprocess_day(df, date):
    """Minimal preprocessing — just need prices, timestamps, IB levels."""
    if df.empty:
        return None
    ib = df[(df.index.time >= MARKET_OPEN) & (df.index.time < IB_END)]
    if ib.empty:
        return None
    ibh = float(ib["price"].max())
    ibl = float(ib["price"].min())
    ib_range = ibh - ibl

    prices = df["price"].values.astype(np.float64)
    ts_ns = df.index.asi8
    sizes = df["size"].values.astype(np.int64)

    # VWAP
    cum_pv = np.cumsum(prices * sizes)
    cum_vol = np.cumsum(sizes)
    vwaps = cum_pv / cum_vol

    post_ib_mask = df.index.time >= IB_END
    post_ib = df[post_ib_mask]
    if post_ib.empty:
        return None
    post_ib_start = int(np.argmax(post_ib_mask))

    return {
        "date": date,
        "ibh": ibh,
        "ibl": ibl,
        "fib_lo": ibl - 0.272 * ib_range,
        "fib_hi": ibh + 0.272 * ib_range,
        "prices": prices,
        "ts_ns": ts_ns,
        "post_ib_prices": post_ib["price"].values.astype(np.float64),
        "post_ib_vwaps": vwaps[post_ib_mask].astype(np.float64),
        "post_ib_start": post_ib_start,
    }


def evaluate_time_exit(
    alert_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    exit_minutes: list[int],
) -> dict[int, float | None]:
    """Return P&L at each time horizon. Positive = favorable direction."""
    alert_ns = ts_ns[alert_idx]
    results = {}

    for mins in exit_minutes:
        exit_ns = alert_ns + np.int64(mins * 60 * 1_000_000_000)
        # Find the last trade at or before exit time
        exit_idx = int(np.searchsorted(ts_ns, exit_ns, side="right")) - 1
        if exit_idx <= alert_idx or exit_idx >= len(prices):
            results[mins] = None
            continue

        exit_price = prices[exit_idx]
        if direction == "up":
            pnl = exit_price - line_price
        else:
            pnl = line_price - exit_price
        results[mins] = float(pnl)

    return results


def evaluate_hybrid(
    alert_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    exit_minutes: int,
    max_loss: float,
) -> float | None:
    """Time exit with protective stop. Exit at time OR stop, whichever first."""
    alert_ns = ts_ns[alert_idx]
    exit_ns = alert_ns + np.int64(exit_minutes * 60 * 1_000_000_000)

    for i in range(alert_idx + 1, len(prices)):
        if ts_ns[i] > exit_ns:
            # Time exit
            exit_price = prices[i - 1] if i > alert_idx + 1 else prices[alert_idx]
            if direction == "up":
                return float(exit_price - line_price)
            else:
                return float(line_price - exit_price)

        # Check stop
        if direction == "up":
            pnl = prices[i] - line_price
        else:
            pnl = line_price - prices[i]

        if pnl <= -max_loss:
            return float(-max_loss)

    return None


def evaluate_target_stop(
    alert_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
) -> str:
    """Current production evaluation for baseline comparison."""
    alert_ns = ts_ns[alert_idx]
    window_ns = np.int64(WINDOW_SECS * 1_000_000_000)

    # Find hit
    start = alert_idx + 1
    end_ns = alert_ns + window_ns
    hit_idx = -1
    for i in range(start, len(prices)):
        if ts_ns[i] > end_ns:
            break
        if abs(prices[i] - line_price) <= HIT_THRESHOLD:
            hit_idx = i
            break
    if hit_idx < 0:
        return "inconclusive"

    eval_end_ns = ts_ns[hit_idx] + window_ns
    target_idx = stop_idx = -1
    if direction == "up":
        tp = line_price + TARGET_POINTS
        sl = line_price - STOP_POINTS
        for i in range(hit_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            if target_idx < 0 and prices[i] >= tp:
                target_idx = i
            if stop_idx < 0 and prices[i] <= sl:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break
    else:
        tp = line_price - TARGET_POINTS
        sl = line_price + STOP_POINTS
        for i in range(hit_idx + 1, len(prices)):
            if ts_ns[i] > eval_end_ns:
                break
            if target_idx < 0 and prices[i] <= tp:
                target_idx = i
            if stop_idx < 0 and prices[i] >= sl:
                stop_idx = i
            if target_idx >= 0 and stop_idx >= 0:
                break

    if target_idx >= 0 and stop_idx >= 0:
        return "correct" if target_idx <= stop_idx else "incorrect"
    elif target_idx >= 0:
        return "correct"
    return "incorrect"


def main() -> None:
    t0 = time.time()
    days = load_cached_days()

    print(f"{'═' * 80}")
    print(f"  TIME-BASED EXIT SWEEP  |  {days[0]} → {days[-1]}  ({len(days)} days)")
    print(f"  Testing exits at: {TIME_EXITS} minutes")
    print(f"{'═' * 80}")

    # Load all days
    print("\n  Loading data...", flush=True)
    day_data = {}
    for i, date in enumerate(days):
        try:
            df = load_day(date)
            dd = preprocess_day(df, date)
            if dd is not None:
                day_data[date] = dd
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(days)} days loaded...", flush=True)
    n_days = len(day_data)
    print(f"  {n_days} days loaded. ({time.time() - t0:.1f}s)")

    # Collect all alerts with time-exit P&L and baseline outcome
    print("\n  Simulating alerts...", flush=True)
    alerts = []

    for day_i, (date, dd) in enumerate(sorted(day_data.items())):
        prices = dd["post_ib_prices"]
        n = len(prices)

        all_levels = [
            ("IBH", np.full(n, dd["ibh"])),
            ("IBL", np.full(n, dd["ibl"])),
            ("VWAP", dd["post_ib_vwaps"]),
            ("FIB_EXT_LO_1.272", np.full(n, dd["fib_lo"])),
            ("FIB_EXT_HI_1.272", np.full(n, dd["fib_hi"])),
        ]

        for level_name, level_arr in all_levels:
            entries = _run_zone_numpy(
                prices, level_arr, ALERT_THRESHOLD, EXIT_THRESHOLD, False
            )

            for idx, entry_count, ref_price in entries:
                price = prices[idx]
                direction = "up" if price > ref_price else "down"
                full_idx = dd["post_ib_start"] + idx

                # Baseline target/stop outcome
                baseline = evaluate_target_stop(
                    full_idx, ref_price, direction, dd["ts_ns"], dd["prices"]
                )

                # Time-based exits
                time_pnls = evaluate_time_exit(
                    full_idx,
                    ref_price,
                    direction,
                    dd["ts_ns"],
                    dd["prices"],
                    TIME_EXITS,
                )

                alerts.append(
                    {
                        "date": date,
                        "level": level_name,
                        "direction": direction,
                        "entry_count": entry_count,
                        "baseline": baseline,
                        **{f"pnl_{m}m": time_pnls.get(m) for m in TIME_EXITS},
                    }
                )

        if (day_i + 1) % 50 == 0:
            print(f"  {day_i + 1}/{n_days} days processed...", flush=True)

    df = pd.DataFrame(alerts)
    print(f"  {len(df)} total alerts generated.")

    # ── Baseline (current production) ────────────────────────────────────────
    decided = df[df["baseline"].isin(["correct", "incorrect"])]
    w = (decided["baseline"] == "correct").sum()
    l = (decided["baseline"] == "incorrect").sum()
    wr = w / (w + l)
    ev = wr * TARGET_POINTS - (1 - wr) * STOP_POINTS
    print(f"\n{'─' * 80}")
    print(f"  BASELINE: +{TARGET_POINTS:.0f} target / -{STOP_POINTS:.0f} stop")
    print(f"{'─' * 80}")
    print(f"  {w}W / {l}L = {wr:.1%} WR, EV {ev:+.2f} pts, {(w+l)/n_days:.1f}/day")

    # ── Time-based exit results ──────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print(f"  TIME-BASED EXIT RESULTS")
    print(f"{'═' * 80}")
    print(
        f"  {'Exit':>6}  {'N':>6}  {'Avg P&L':>8}  {'Med P&L':>8}  {'Win%':>6}  {'W':>5}  {'L':>5}  {'/day':>5}  {'EV':>8}"
    )
    print(
        f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*8}"
    )

    for mins in TIME_EXITS:
        col = f"pnl_{mins}m"
        valid = df[df[col].notna()][col]
        if len(valid) == 0:
            continue
        avg_pnl = valid.mean()
        med_pnl = valid.median()
        wins = (valid > 0).sum()
        losses = (valid <= 0).sum()
        total = wins + losses
        wr = wins / total if total > 0 else 0
        per_day = total / n_days
        ev = avg_pnl
        print(
            f"  {mins:>4}m  {total:>6}  {avg_pnl:>+8.2f}  {med_pnl:>+8.2f}  "
            f"{wr:>5.1%}  {wins:>5}  {losses:>5}  {per_day:>5.1f}  {ev:>+8.2f}"
        )

    # ── P&L distribution at each time horizon ────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  P&L DISTRIBUTION (percentiles)")
    print(f"{'─' * 80}")
    print(
        f"  {'Exit':>6}  {'p10':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p90':>8}  {'Std':>8}"
    )
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for mins in TIME_EXITS:
        col = f"pnl_{mins}m"
        valid = df[df[col].notna()][col].values
        if len(valid) == 0:
            continue
        pcts = np.percentile(valid, [10, 25, 50, 75, 90])
        std = np.std(valid)
        print(
            f"  {mins:>4}m  {pcts[0]:>+8.2f}  {pcts[1]:>+8.2f}  {pcts[2]:>+8.2f}  "
            f"{pcts[3]:>+8.2f}  {pcts[4]:>+8.2f}  {std:>8.2f}"
        )

    # ── Per-level breakdown at best time horizon ─────────────────────────────
    # Find the time horizon with highest EV
    best_mins = None
    best_ev = -999
    for mins in TIME_EXITS:
        col = f"pnl_{mins}m"
        valid = df[df[col].notna()][col]
        if len(valid) > 0 and valid.mean() > best_ev:
            best_ev = valid.mean()
            best_mins = mins

    if best_mins is not None:
        col = f"pnl_{best_mins}m"
        print(f"\n{'─' * 80}")
        print(f"  PER-LEVEL BREAKDOWN @ {best_mins}-minute exit")
        print(f"{'─' * 80}")
        print(
            f"  {'Level':<25}  {'N':>5}  {'Avg':>8}  {'Med':>8}  {'Win%':>6}  {'/day':>5}"
        )
        print(f"  {'-'*25}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}")
        for lvl in ["IBH", "IBL", "VWAP", "FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"]:
            sub = df[(df["level"] == lvl) & df[col].notna()][col]
            if len(sub) == 0:
                continue
            w = (sub > 0).sum()
            t = len(sub)
            print(
                f"  {lvl:<25}  {t:>5}  {sub.mean():>+8.2f}  {sub.median():>+8.2f}  "
                f"{w/t:>5.1%}  {t/n_days:>5.1f}"
            )

        # Per direction
        print(f"\n  {'Direction':<25}  {'N':>5}  {'Avg':>8}  {'Med':>8}  {'Win%':>6}")
        print(f"  {'-'*25}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}")
        for d in ["up", "down"]:
            sub = df[(df["direction"] == d) & df[col].notna()][col]
            if len(sub) == 0:
                continue
            w = (sub > 0).sum()
            t = len(sub)
            print(
                f"  {d:<25}  {t:>5}  {sub.mean():>+8.2f}  {sub.median():>+8.2f}  "
                f"{w/t:>5.1%}"
            )

    # ── Hybrid: time exit + protective stop ──────────────────────────────────
    # Pre-collect alert metadata for fast hybrid evaluation
    print(f"\n  Pre-collecting alerts for hybrid evaluation...", flush=True)
    alert_meta: list[tuple[int, float, str, np.ndarray, np.ndarray]] = []
    for _, dd in sorted(day_data.items()):
        pp = dd["post_ib_prices"]
        nn = len(pp)
        all_levels = [
            ("IBH", np.full(nn, dd["ibh"])),
            ("IBL", np.full(nn, dd["ibl"])),
            ("VWAP", dd["post_ib_vwaps"]),
            ("FIB_EXT_LO_1.272", np.full(nn, dd["fib_lo"])),
            ("FIB_EXT_HI_1.272", np.full(nn, dd["fib_hi"])),
        ]
        for _, level_arr in all_levels:
            entries = _run_zone_numpy(
                pp, level_arr, ALERT_THRESHOLD, EXIT_THRESHOLD, False
            )
            for idx, _, ref_price in entries:
                price = pp[idx]
                direction = "up" if price > ref_price else "down"
                full_idx = dd["post_ib_start"] + idx
                alert_meta.append(
                    (full_idx, ref_price, direction, dd["ts_ns"], dd["prices"])
                )
    print(f"  {len(alert_meta)} alerts collected.", flush=True)

    print(f"\n{'═' * 80}")
    print(f"  HYBRID: TIME EXIT + PROTECTIVE STOP")
    print(f"{'═' * 80}")
    print(
        f"  {'Time':>6}  {'Stop':>6}  {'N':>6}  {'Avg P&L':>8}  {'Med P&L':>8}  {'Win%':>6}  {'/day':>5}"
    )
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}")

    for exit_mins in [2, 3, 5, 7, 10]:
        for max_loss in HYBRID_STOPS:
            pnls = []
            for full_idx, ref_price, direction, ts_ns_arr, prices_arr in alert_meta:
                pnl = evaluate_hybrid(
                    full_idx,
                    ref_price,
                    direction,
                    ts_ns_arr,
                    prices_arr,
                    exit_mins,
                    max_loss,
                )
                if pnl is not None:
                    pnls.append(pnl)

            if not pnls:
                continue
            arr = np.array(pnls)
            avg = arr.mean()
            med = np.median(arr)
            wr = (arr > 0).sum() / len(arr)
            per_day = len(arr) / n_days
            print(
                f"  {exit_mins:>4}m  {max_loss:>4}pt  {len(arr):>6}  "
                f"{avg:>+8.2f}  {med:>+8.2f}  {wr:>5.1%}  {per_day:>5.1f}"
            )

    print(f"\n{'═' * 80}")
    print(f"  Total runtime: {time.time() - t0:.1f}s")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
