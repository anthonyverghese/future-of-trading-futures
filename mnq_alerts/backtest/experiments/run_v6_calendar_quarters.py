"""run_v6_calendar_quarters.py — V6 baseline aggregated by calendar (year, quarter).

V6 = V0 deployed config with Monday-double caps DISABLED (deployed 2026-05-06).
Existing /tmp/filter_audit/V6_no_monday_caps.json aggregates Q1-Q4 by sorting all
days into quartiles, which doesn't match calendar quarters. The level-model's
walk-forward folds are calendar-quarterly, so we need per-(year, quarter) here.

Wall clock: ~12 min.

Outputs:
- /tmp/filter_audit/V6_calendar_quarters.json — full result + day_pnl + per_year_quarter
- Prints V6_PER_QUARTER dict ready to paste into mnq_alerts/scripts/select_architecture.py
- Prints V6_FINAL_MEAN_DAILY ready to paste into mnq_alerts/scripts/run_final_test.py
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
NAME = "V6_calendar_quarters"
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

FINAL_TEST_TRADING_DAYS = 30


def _ts() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[{_ts()}] {msg}\n")
    print(f"[{_ts()}] {msg}", flush=True)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("START V6 (V0 with monday_double=False), calendar-quarter aggregation")

    from mnq_alerts.backtest.data import load_all_days, precompute_arrays
    from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

    t0 = time.time()
    log("Loading caches...")
    dates, caches = load_all_days()
    log(f"Caches loaded: {len(dates)} days in {time.time()-t0:.0f}s")

    sink = StringIO()
    total_pnl = 0.0
    total_trades = 0
    day_pnl: dict = {}

    n = len(dates)
    t1 = time.time()
    log(f"Sim starting over {n} days...")

    for i, date in enumerate(dates):
        dc = caches[date]
        arr = precompute_arrays(dc)

        # V6: monday_double DISABLED — caps are NEVER doubled regardless of weekday.
        day_caps = dict(DEFAULT_CAPS)

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
                entry_limit_buffer_pts_override=None,
            )
        sink.truncate(0); sink.seek(0)

        d_pnl = sum(t.pnl_usd for t in trades)
        total_pnl += d_pnl
        total_trades += len(trades)
        day_pnl[date] = d_pnl

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t1
            eta = (n - i - 1) * (elapsed / (i + 1))
            log(f"  day {i+1}/{n} | {total_trades} trades | "
                f"${total_pnl/(i+1):+.2f}/day so far | ETA {eta:.0f}s")

    sorted_dates = sorted(day_pnl.keys())

    # Per-(year, calendar quarter) aggregation.
    per_yq_pnl = defaultdict(list)
    for d in sorted_dates:
        q = (d.year, (d.month - 1) // 3 + 1)
        per_yq_pnl[q].append(day_pnl[d])
    per_yq_mean = {f"{y}-Q{q}": sum(v) / len(v) for (y, q), v in per_yq_pnl.items()}

    # Final-test window: last FINAL_TEST_TRADING_DAYS days.
    final_window = sorted_dates[-FINAL_TEST_TRADING_DAYS:]
    final_mean = sum(day_pnl[d] for d in final_window) / len(final_window) if final_window else 0.0

    # Dev-set per-(y, q): exclude final test window's dates, then aggregate.
    final_set = set(final_window)
    dev_yq_pnl = defaultdict(list)
    for d in sorted_dates:
        if d in final_set:
            continue
        q = (d.year, (d.month - 1) // 3 + 1)
        dev_yq_pnl[q].append(day_pnl[d])
    dev_yq_mean = {f"{y}-Q{q}": sum(v) / len(v) for (y, q), v in dev_yq_pnl.items()}

    result = {
        "name": NAME,
        "label": "V6 with calendar-quarter aggregation",
        "trades": total_trades,
        "days": len(sorted_dates),
        "pnl_total": total_pnl,
        "pnl_per_day": total_pnl / len(sorted_dates) if sorted_dates else 0.0,
        "per_year_quarter_pnl_per_day_full": per_yq_mean,
        "per_year_quarter_pnl_per_day_dev": dev_yq_mean,
        "final_test_mean_daily_pnl": final_mean,
        "final_test_window_start": str(final_window[0]) if final_window else None,
        "final_test_window_end": str(final_window[-1]) if final_window else None,
        "day_pnl": {str(d): v for d, v in sorted(day_pnl.items())},
        "elapsed_secs": time.time() - t0,
    }
    with open(JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log(f"DONE  trades={total_trades} $/day=${total_pnl/len(sorted_dates):+.2f} "
        f"elapsed={time.time()-t0:.0f}s")
    log("Per-(year, quarter) $/day on FULL sample:")
    for k, v in sorted(per_yq_mean.items()):
        log(f"  {k}: ${v:+.2f}")
    log("Per-(year, quarter) $/day excluding final-test window (use these for V6_PER_QUARTER):")
    for k, v in sorted(dev_yq_mean.items()):
        log(f"  {k}: ${v:+.2f}")
    log(f"Final-test mean daily P&L (use for V6_FINAL_MEAN_DAILY): ${final_mean:+.2f}")
    log(f"  window: {final_window[0]} -> {final_window[-1]}")

    # Convenience: print drop-in dict for select_architecture.py.
    print()
    print("Paste into mnq_alerts/scripts/select_architecture.py:")
    print("V6_PER_QUARTER = {")
    for k, v in sorted(dev_yq_mean.items()):
        y, q = k.split("-Q")
        print(f"    ({y}, {q}): {v:+.4f},")
    print("}")
    print()
    print(f"Paste into mnq_alerts/scripts/run_final_test.py:")
    print(f"V6_FINAL_MEAN_DAILY = {final_mean:.4f}")


if __name__ == "__main__":
    main()
