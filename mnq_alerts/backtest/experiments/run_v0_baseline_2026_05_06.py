"""run_v0_baseline_2026_05_06.py — fresh V0_baseline run for filter audit.

Runs the current deployed C+IBH=0.75 config through simulate_day_v2 with
slippage modeling, exact same infrastructure as the 16-variant audit ran.
Produces per-day, per-quarter, per-level results so we can compute clean
deltas for V6 (no Monday caps) and other variants.

Wall clock: ~12 min.

Output: /tmp/filter_audit/V0_baseline.json (matches schema of audit JSONs)
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

OUTPUT_DIR = "/tmp/filter_audit"
NAME = "V0_baseline"
LOG = os.path.join(OUTPUT_DIR, f"{NAME}.log")
JSON = os.path.join(OUTPUT_DIR, f"{NAME}.json")

DEFAULT_TS = {
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
DEFAULT_CAPS = {
    "FIB_0.236": 18,
    "FIB_0.618": 3,
    "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6,
    "FIB_EXT_LO_1.272": 6,
    "IBH": 7,
}


def _ts() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"[{_ts()}] {msg}\n")
    print(f"[{_ts()}] {msg}", flush=True)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # If LOG exists, append fresh start
    log(f"START V0_baseline (current deployed C+IBH=0.75 config)")

    from mnq_alerts.backtest.data import load_all_days, precompute_arrays
    from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

    t0 = time.time()
    log("Loading caches...")
    dates, caches = load_all_days()
    log(f"Caches loaded: {len(dates)} days in {time.time()-t0:.0f}s")

    sink = StringIO()
    total_pnl = 0.0
    total_trades = 0
    by_level = defaultdict(lambda: [0, 0, 0.0])
    by_outcome: dict = defaultdict(int)
    day_pnl: dict = {}

    n = len(dates)
    t1 = time.time()
    log(f"Sim starting over {n} days...")

    for i, date in enumerate(dates):
        dc = caches[date]
        arr = precompute_arrays(dc)

        # Apply Monday-double externally (matches filter_audit handling)
        day_caps = dict(DEFAULT_CAPS)
        if date.weekday() == 0:
            day_caps = {k: v * 2 for k, v in day_caps.items()}

        with redirect_stdout(sink):
            trades = simulate_day_v2(
                dc, arr,
                per_level_ts=DEFAULT_TS,
                per_level_caps=day_caps,
                exclude_levels={"FIB_0.5", "FIB_EXT_LO_1.272"},
                include_ibl=False,
                include_vwap=False,
                direction_filter={"IBH": "down"},
                daily_loss=200.0,
                timeout_secs=900,
                momentum_max=0.0,
                simulate_slippage=True,
                latency_ms=100.0,
                entry_limit_buffer_pts_override=None,  # use config dict {"IBH": 0.75}
            )
        sink.truncate(0); sink.seek(0)

        d_pnl = sum(t.pnl_usd for t in trades)
        total_pnl += d_pnl
        total_trades += len(trades)
        day_pnl[date] = d_pnl
        for t in trades:
            s = by_level[t.level]
            s[0] += 1
            if t.pnl_usd >= 0:
                s[1] += 1
            s[2] += t.pnl_usd
            by_outcome[t.outcome] += 1

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t1
            eta = (n - i - 1) * (elapsed / (i + 1))
            log(f"  day {i+1}/{n} | {total_trades} trades | "
                f"${total_pnl/(i+1):+.2f}/day so far | ETA {eta:.0f}s")

    # MaxDD
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    sorted_dates = sorted(day_pnl.keys())
    for d in sorted_dates:
        cum += day_pnl[d]
        if cum > peak:
            peak = cum
        if peak - cum > max_dd:
            max_dd = peak - cum

    # Per-quarter
    n_days = len(sorted_dates)
    q_size = n_days // 4
    per_q = {}
    per_q_dd = {}
    for q_label, qdates in [
        ("Q1", sorted_dates[:q_size]),
        ("Q2", sorted_dates[q_size:2 * q_size]),
        ("Q3", sorted_dates[2 * q_size:3 * q_size]),
        ("Q4", sorted_dates[3 * q_size:]),
    ]:
        q_pnl = sum(day_pnl[d] for d in qdates)
        per_q[q_label] = q_pnl / len(qdates)
        qcum = 0.0
        qpeak = 0.0
        qdd = 0.0
        for d in qdates:
            qcum += day_pnl[d]
            if qcum > qpeak:
                qpeak = qcum
            if qpeak - qcum > qdd:
                qdd = qpeak - qcum
        per_q_dd[q_label] = qdd

    result = {
        "name": NAME,
        "label": "Baseline (current deployed C+IBH=0.75)",
        "trades": total_trades,
        "days": n_days,
        "pnl_total": total_pnl,
        "pnl_per_day": total_pnl / n_days if n_days else 0.0,
        "max_dd": max_dd,
        "per_quarter_pnl_per_day": per_q,
        "per_quarter_max_dd": per_q_dd,
        "by_level": {k: list(v) for k, v in by_level.items()},
        "by_outcome": dict(by_outcome),
        "elapsed_secs": time.time() - t0,
    }
    with open(JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log(f"DONE  trades={total_trades} "
        f"$/day=${total_pnl/n_days:+.2f} "
        f"MaxDD=${max_dd:.0f} "
        f"elapsed={time.time()-t0:.0f}s")
    log(f"Per-Q $/day: Q1={per_q['Q1']:+.2f} Q2={per_q['Q2']:+.2f} "
        f"Q3={per_q['Q3']:+.2f} Q4={per_q['Q4']:+.2f}")


if __name__ == "__main__":
    main()
