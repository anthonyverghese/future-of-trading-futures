"""Sweep entry_limit_buffer to find slippage-vs-fill-rate optimum.

Background: the deployed bot uses entry_limit_buffer = target_pts / 2,
which on a SELL places the limit at line - target/2. Live data shows
fills clustering near the limit price (e.g., 5x IBH trades on 2026-05-05
filled at 28132-28133 with limit 28132, line 28135 — ~2-3pt slippage).
The existing v2 backtest doesn't model slippage at all (evaluate_bot_trade
uses line_price as entry), so the +$47.13/day baseline is optimistic.

Approach: post-process the existing variants_v1_trades.pkl. For each
trade, simulate the fill with a tick-data walk-forward model:
  1. Entry latency (default 100ms — matches typical Databento->IBKR path)
  2. Walk forward up to BOT_FILL_TIMEOUT_SECS (3.0s) looking for the
     first tick where the limit is satisfied.
     - SELL LIMIT at X: fills when price >= X (bid hits or exceeds X)
     - BUY LIMIT at X:  fills when price <= X
  3. If no fill within timeout, drop the trade (no entry).

Outcome (win/loss/timeout) is unchanged from the original sim — target
and stop are absolute prices derived from the line, so the eval window
is the same regardless of where in that window we filled. Only P&L
changes (it's now computed from fill_price, not line_price).

Sweep buffers: 0.0, 0.5, 1.0, 1.5, 2.0, "target/4", "target/2" (current).

Outputs a comparison table: fill rate, $/day, $/trade, MaxDD vs baseline.
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

PICKLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "variants_v1_trades.pkl"
)
MULTIPLIER = 2.0
FEE_USD = 1.24


def simulate_fill(
    prices: np.ndarray,
    ts_ns: np.ndarray,
    entry_ns: int,
    direction: str,
    line: float,
    buffer: float,
    latency_ms: float = 100.0,
    timeout_secs: float = 3.0,
) -> tuple[int, float] | tuple[None, None]:
    """Simulate a limit order fill.

    Returns (fill_idx, fill_price) or (None, None) if no fill within
    timeout. The limit price is `line - buffer` for SELL ("down")
    and `line + buffer` for BUY ("up").
    """
    if direction == "down":
        limit = line - buffer
    else:
        limit = line + buffer

    start_ns = entry_ns + int(latency_ms * 1_000_000)
    end_ns = entry_ns + int(timeout_secs * 1_000_000_000)

    start_idx = int(np.searchsorted(ts_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(ts_ns, end_ns, side="right"))
    if start_idx >= end_idx or start_idx >= len(prices):
        return None, None
    end_idx = min(end_idx, len(prices))

    seg = prices[start_idx:end_idx]
    if direction == "down":  # SELL: fill when price >= limit
        hit = np.where(seg >= limit)[0]
    else:                     # BUY: fill when price <= limit
        hit = np.where(seg <= limit)[0]

    if len(hit) == 0:
        return None, None

    fill_idx = start_idx + int(hit[0])
    return fill_idx, float(prices[fill_idx])


def adjusted_pnl(
    outcome: str,
    direction: str,
    line: float,
    fill: float,
    target_pts: float,
    stop_pts: float,
    original_pnl_usd: float,
) -> float:
    """Recompute P&L given slippage from line to fill.

    For win/loss the new P&L is computed from absolute target/stop
    prices and the fill. For timeout we shift the original (assumed
    line-based) P&L by the slippage delta — the timeout exit price
    isn't in the pickle, but the differential is the same.
    """
    if outcome == "win":
        if direction == "down":
            target_price = line - target_pts
            pnl_pts = fill - target_price
        else:
            target_price = line + target_pts
            pnl_pts = target_price - fill
        return pnl_pts * MULTIPLIER - FEE_USD

    if outcome == "loss":
        if direction == "down":
            stop_price = line + stop_pts
            pnl_pts = fill - stop_price
        else:
            stop_price = line - stop_pts
            pnl_pts = stop_price - fill
        return pnl_pts * MULTIPLIER - FEE_USD

    # timeout: original P&L assumed entry at line; adjust by slippage.
    # For SELL, slippage = (line - fill) is "missed profit" (we sold
    # cheaper than the line). Subtract from original.
    # For BUY, slippage = (fill - line) is "extra cost".
    if direction == "down":
        slippage_usd = (line - fill) * MULTIPLIER
    else:
        slippage_usd = (fill - line) * MULTIPLIER
    return original_pnl_usd - slippage_usd


def run_buffer(
    rows: list[dict], caches: dict, buffer_spec, latency_ms: float = 100.0,
) -> dict:
    """Run one buffer setting over all trades.

    `buffer_spec` is either a float (fixed buffer in pts) or the string
    "target/2" / "target/4" for target-relative buffers.
    """
    n_filled = 0
    n_dropped = 0
    pnl_total = 0.0
    pnl_per_day: dict = defaultdict(float)
    new_outcomes: dict = defaultdict(int)

    for r in rows:
        date = r["date"]
        dc = caches[date]
        prices = dc.full_prices
        ts_ns = dc.full_ts_ns

        if isinstance(buffer_spec, str):
            t = r["target_pts"]
            if buffer_spec == "target/2":
                buf = t / 2.0
            elif buffer_spec == "target/4":
                buf = t / 4.0
            else:
                raise ValueError(buffer_spec)
        else:
            buf = float(buffer_spec)

        # Round to MNQ tick (0.25)
        buf_rounded = round(buf * 4) / 4.0

        fill_idx, fill_price = simulate_fill(
            prices, ts_ns,
            entry_ns=r["entry_ns"],
            direction=r["direction"],
            line=r["line_price"],
            buffer=buf_rounded,
            latency_ms=latency_ms,
        )
        if fill_idx is None:
            n_dropped += 1
            continue

        n_filled += 1
        new_pnl = adjusted_pnl(
            outcome=r["outcome"], direction=r["direction"],
            line=r["line_price"], fill=fill_price,
            target_pts=r["target_pts"], stop_pts=r["stop_pts"],
            original_pnl_usd=r["pnl_usd"],
        )
        pnl_total += new_pnl
        pnl_per_day[date] += new_pnl
        new_outcomes[r["outcome"]] += 1

    days = len({r["date"] for r in rows})
    # MaxDD over chronologically-ordered trades
    by_date_sorted = sorted(pnl_per_day.items())
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for _, day_p in by_date_sorted:
        cum += day_p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    return {
        "n_filled": n_filled,
        "n_dropped": n_dropped,
        "fill_rate": n_filled / (n_filled + n_dropped) if (n_filled + n_dropped) else 0,
        "pnl_total": pnl_total,
        "pnl_per_day": pnl_total / days if days else 0,
        "pnl_per_trade": pnl_total / n_filled if n_filled else 0,
        "max_dd": max_dd,
        "outcomes": dict(new_outcomes),
        "days": days,
    }


def main() -> None:
    print("Loading pickle...", flush=True)
    with open(PICKLE_PATH, "rb") as f:
        rows = pickle.load(f)
    days = len({r["date"] for r in rows})
    print(f"  {len(rows)} trades over {days} days", flush=True)
    print(f"  Original (no-slippage) baseline: "
          f"${sum(r['pnl_usd'] for r in rows)/days:+.2f}/day", flush=True)

    print("\nLoading day caches (~3 min)...", flush=True)
    t0 = time.time()
    dates, caches = load_all_days()
    print(f"  {len(dates)} days loaded in {time.time()-t0:.1f}s", flush=True)

    print("\nRunning buffer sweep (latency=100ms, timeout=3s)...", flush=True)
    sweep = [
        ("0.0",      0.0),
        ("0.5",      0.5),
        ("1.0",      1.0),
        ("1.5",      1.5),
        ("2.0",      2.0),
        ("3.0",      3.0),
        ("target/4", "target/4"),
        ("target/2", "target/2"),  # current deployed
    ]
    results = {}
    for label, spec in sweep:
        t1 = time.time()
        res = run_buffer(rows, caches, spec, latency_ms=100.0)
        results[label] = res
        print(
            f"  buffer={label:<10} "
            f"fills={res['n_filled']:>5}/{res['n_filled']+res['n_dropped']} "
            f"({res['fill_rate']*100:>4.1f}%) "
            f"$/day=${res['pnl_per_day']:+7.2f} "
            f"$/tr=${res['pnl_per_trade']:+5.2f} "
            f"MaxDD=${res['max_dd']:>4.0f}  ({time.time()-t1:.1f}s)",
            flush=True,
        )

    # Sensitivity check at the current deployed buffer with different latencies
    print("\nLatency sensitivity at buffer=target/2 (current deployed):",
          flush=True)
    for lat in [50, 100, 200, 500]:
        res = run_buffer(rows, caches, "target/2", latency_ms=float(lat))
        print(
            f"  latency={lat:>3}ms  "
            f"fills={res['n_filled']:>5}  "
            f"$/day=${res['pnl_per_day']:+7.2f}  "
            f"$/tr=${res['pnl_per_trade']:+5.2f}",
            flush=True,
        )

    # Comparison summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    base = results["target/2"]
    print(f"Current deployed (target/2):      "
          f"${base['pnl_per_day']:+.2f}/day  fill={base['fill_rate']*100:.1f}%")
    print(f"Old no-slippage backtest baseline: "
          f"${sum(r['pnl_usd'] for r in rows)/days:+.2f}/day (overstated by "
          f"${(sum(r['pnl_usd'] for r in rows)/days) - base['pnl_per_day']:+.2f}"
          f"/day vs slippage-modeled)")
    best_label = max(
        results.keys(), key=lambda k: results[k]["pnl_per_day"]
    )
    best = results[best_label]
    print(f"\nBest in sweep: buffer={best_label} "
          f"=> ${best['pnl_per_day']:+.2f}/day, "
          f"fill={best['fill_rate']*100:.1f}%, MaxDD=${best['max_dd']:.0f}")
    print(f"Δ vs current: "
          f"${best['pnl_per_day'] - base['pnl_per_day']:+.2f}/day")


if __name__ == "__main__":
    main()
