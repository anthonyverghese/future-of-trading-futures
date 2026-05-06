"""filter_audit_2026_05_06.py — comprehensive filter audit (16 variants).

Re-validates every currently-deployed bot filter under the new slippage-
modeled real-sim framework. Each variant turns ONE filter off (or sweeps
one parameter) and compares to the deployed C+IBH=0.75 baseline.

Architecture:
  - 2 worker processes (mp.Pool) — each loads day caches once (~3 min)
    then processes variants from a shared queue (each variant ~9 min).
  - Per-variant progress logs at /tmp/filter_audit/<name>.log
  - Per-variant results JSON at /tmp/filter_audit/<name>.json
  - Master heartbeat to /tmp/filter_audit/MASTER.log every 30s, plus
    final aggregated comparison table.

Wall clock estimate: ~75 min.

Usage: python -m mnq_alerts.backtest.experiments.filter_audit_2026_05_06
"""

from __future__ import annotations

import datetime as _dt
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

OUTPUT_DIR = "/tmp/filter_audit"
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "results", "filter_audit_2026_05_06.json"
)

# Module-level globals populated by init_worker (one copy per worker process)
_CACHES = None
_DATES = None

# Default deployed config (matches production after 2026-05-05 deploy)
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
DEFAULT_EXCLUDE = {"FIB_EXT_LO_1.272"}  # FIB_0.5 + IBL added internally by simulate_v2
DEFAULT_DIR_FILTER = {"IBH": "down"}
DEFAULT_DAILY_LOSS = 200.0
DEFAULT_TIMEOUT_SECS = 900
DEFAULT_VOL_FILTER = 0.0015
DEFAULT_MONDAY_DOUBLE = True

# Variant definitions — each overrides one or more defaults. Anything
# not in the dict uses the default value above. The audit reads:
#   "what changes when this filter is removed/changed?"
VARIANTS = [
    # Group 1 — currently-excluded levels (re-include each one)
    {"name": "V1_include_FIB_EXT_LO", "label": "Re-include FIB_EXT_LO_1.272",
     "exclude_levels": set()},
    {"name": "V2_include_FIB_0.5",   "label": "Re-include FIB_0.5",
     "include_fib_0_5": True},
    {"name": "V3_include_IBL",       "label": "Re-include IBL",
     "include_ibl": True},
    {"name": "V4_include_VWAP",      "label": "Re-include VWAP",
     "include_vwap": True},

    # Group 2 — trade-gating mechanics
    {"name": "V5_allow_IBH_BUY",     "label": "Allow IBH BUY (drop direction filter)",
     "direction_filter": {}},
    {"name": "V6_no_monday_caps",    "label": "Disable Monday double caps",
     "monday_double": False},
    {"name": "V7_no_position_timeout", "label": "No 15-min position timeout",
     "timeout_secs": 99999},

    # Group 3 — daily loss limit sweep
    {"name": "L1_loss_150",          "label": "Loss limit $150",
     "daily_loss": 150.0},
    {"name": "L2_loss_300",          "label": "Loss limit $300",
     "daily_loss": 300.0},
    {"name": "L3_loss_off",          "label": "No daily loss limit",
     "daily_loss": 999999.0},

    # Group 4 — vol filter sweep
    {"name": "W1_vol_010",           "label": "Vol filter 0.10%",
     "vol_filter": 0.0010},
    {"name": "W2_vol_020",           "label": "Vol filter 0.20%",
     "vol_filter": 0.0020},
    {"name": "W3_vol_off",           "label": "Vol filter off",
     "vol_filter": 0.0},

    # Group 5 — per-level cap sweeps
    {"name": "C1_caps_doubled",      "label": "All caps doubled",
     "caps": {"FIB_0.236": 36, "FIB_0.618": 6, "FIB_0.764": 10,
              "FIB_EXT_HI_1.272": 12, "FIB_EXT_LO_1.272": 12, "IBH": 14}},
    {"name": "C2_caps_halved",       "label": "All caps halved",
     "caps": {"FIB_0.236": 9, "FIB_0.618": 2, "FIB_0.764": 3,
              "FIB_EXT_HI_1.272": 3, "FIB_EXT_LO_1.272": 3, "IBH": 4}},
    {"name": "C3_caps_off",          "label": "Caps off (effectively unlimited)",
     "caps": {"FIB_0.236": 99, "FIB_0.618": 99, "FIB_0.764": 99,
              "FIB_EXT_HI_1.272": 99, "FIB_EXT_LO_1.272": 99, "IBH": 99}},
]


def _ts() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def _log(path: str, msg: str) -> None:
    with open(path, "a") as f:
        f.write(f"[{_ts()}] {msg}\n")


def init_worker() -> None:
    """Worker process initializer — loads caches once and keeps them."""
    global _CACHES, _DATES
    pid = os.getpid()
    log_path = os.path.join(OUTPUT_DIR, f"worker_{pid}.log")
    _log(log_path, f"Worker {pid} starting — loading day caches...")
    try:
        from mnq_alerts.backtest.data import load_all_days
        t0 = time.time()
        _DATES, _CACHES = load_all_days()
        _log(log_path, f"Worker {pid} ready — {len(_DATES)} days loaded "
             f"in {time.time()-t0:.0f}s")
    except Exception as e:
        _log(log_path, f"Worker {pid} INIT FAILED: {type(e).__name__}: {e}")
        _log(log_path, traceback.format_exc())
        raise


def run_variant(variant: dict) -> tuple[str, str]:
    """Run one variant in the worker process.

    Returns (variant_name, status) where status is "ok" or "error:<msg>".
    All progress and final result are written to disk.
    """
    name = variant["name"]
    label = variant["label"]
    log_path = os.path.join(OUTPUT_DIR, f"{name}.log")
    json_path = os.path.join(OUTPUT_DIR, f"{name}.json")

    # Build effective config from variant overrides
    exclude_levels = set(variant.get("exclude_levels", DEFAULT_EXCLUDE))
    include_ibl = variant.get("include_ibl", False)
    include_vwap = variant.get("include_vwap", False)
    include_fib_0_5 = variant.get("include_fib_0_5", False)
    direction_filter = variant.get("direction_filter", DEFAULT_DIR_FILTER)
    daily_loss = variant.get("daily_loss", DEFAULT_DAILY_LOSS)
    timeout_secs = variant.get("timeout_secs", DEFAULT_TIMEOUT_SECS)
    vol_filter = variant.get("vol_filter", DEFAULT_VOL_FILTER)
    monday_double = variant.get("monday_double", DEFAULT_MONDAY_DOUBLE)
    caps = dict(variant.get("caps", DEFAULT_CAPS))

    _log(log_path, f"START {label}")
    _log(log_path, f"  exclude={exclude_levels} ibl={include_ibl} "
         f"vwap={include_vwap} fib_0_5={include_fib_0_5}")
    _log(log_path, f"  dir_filter={direction_filter} daily_loss=${daily_loss:.0f} "
         f"timeout={timeout_secs}s vol={vol_filter:.4f} mon_dbl={monday_double}")
    _log(log_path, f"  caps={caps}")

    # Patch config + bot_trader bindings for vol filter (simulate_v2 doesn't
    # expose this as a parameter). Restore in finally.
    try:
        import config as cfg
        import bot_trader as bt_mod
    except ImportError:
        # Worker runs from project root; mnq_alerts is on sys.path implicitly
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
        import config as cfg
        import bot_trader as bt_mod

    orig_vol_cfg = cfg.BOT_VOL_FILTER_MIN_RANGE_PCT
    orig_vol_bt = bt_mod.BOT_VOL_FILTER_MIN_RANGE_PCT
    cfg.BOT_VOL_FILTER_MIN_RANGE_PCT = vol_filter
    bt_mod.BOT_VOL_FILTER_MIN_RANGE_PCT = vol_filter
    # Force monday-double off in bot_trader so we can apply it explicitly
    # at the script level (mirrors run_slippage_aware_v1 pattern, but with
    # the per-variant flag).
    orig_mon_cfg = cfg.BOT_MONDAY_DOUBLE_CAPS
    orig_mon_bt = bt_mod.BOT_MONDAY_DOUBLE_CAPS
    cfg.BOT_MONDAY_DOUBLE_CAPS = False
    bt_mod.BOT_MONDAY_DOUBLE_CAPS = False

    # FIB_0.5 is excluded by default inside simulate_v2's call site (the
    # exclude_levels arg is unioned with {"FIB_0.5", "IBL"} when relevant).
    # If we want to RE-INCLUDE FIB_0.5, simulate_v2 needs to receive an
    # exclude that doesn't add it back. simulate_v2.py itself doesn't add
    # FIB_0.5 internally — that's done by the caller (run_slippage_aware_v1
    # adds {"FIB_0.5", "IBL"} on the call site). Here we control the full
    # exclude_levels set, so FIB_0.5 is excluded only if we put it there.
    if not include_fib_0_5:
        exclude_levels = set(exclude_levels) | {"FIB_0.5"}

    try:
        from mnq_alerts.backtest.simulate_v2 import simulate_day_v2
        from mnq_alerts.backtest.data import precompute_arrays

        sink = StringIO()
        total_pnl = 0.0
        total_trades = 0
        by_level = defaultdict(lambda: [0, 0, 0.0])  # [n, wins, pnl]
        by_outcome: dict[str, int] = defaultdict(int)
        day_pnl: dict = {}

        n = len(_DATES)
        t0 = time.time()
        _log(log_path, f"Sim starting over {n} days...")

        for i, date in enumerate(_DATES):
            dc = _CACHES[date]
            arr = precompute_arrays(dc)

            # Apply Monday-double externally (we forced bot_trader's flag off)
            day_caps = dict(caps)
            if monday_double and date.weekday() == 0:
                day_caps = {k: v * 2 for k, v in day_caps.items()}

            with redirect_stdout(sink):
                trades = simulate_day_v2(
                    dc, arr,
                    per_level_ts=DEFAULT_TS,
                    per_level_caps=day_caps,
                    exclude_levels=exclude_levels,
                    include_ibl=include_ibl,
                    include_vwap=include_vwap,
                    direction_filter=direction_filter,
                    daily_loss=daily_loss,
                    timeout_secs=timeout_secs,
                    momentum_max=0.0,
                    simulate_slippage=True,
                    latency_ms=100.0,
                    entry_limit_buffer_pts_override=None,  # use config dict
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
                elapsed = time.time() - t0
                eta = (n - i - 1) * (elapsed / (i + 1))
                _log(log_path,
                     f"  day {i+1}/{n} | {total_trades} trades | "
                     f"${total_pnl/(i+1):+.2f}/day so far | ETA {eta:.0f}s")

        # Compute MaxDD
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        sorted_dates = sorted(day_pnl.keys())
        for d in sorted_dates:
            cum += day_pnl[d]
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        # Per-quarter $/day for walk-forward
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
            # Per-quarter DD
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
            "name": name,
            "label": label,
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
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        _log(log_path,
             f"DONE  trades={total_trades} "
             f"$/day=${total_pnl/n_days:+.2f} "
             f"MaxDD=${max_dd:.0f} "
             f"elapsed={time.time()-t0:.0f}s")
        return name, "ok"

    except Exception as e:
        tb = traceback.format_exc()
        _log(log_path, f"ERROR {type(e).__name__}: {e}")
        _log(log_path, tb)
        with open(json_path, "w") as f:
            json.dump({"name": name, "error": str(e), "traceback": tb}, f, indent=2)
        return name, f"error:{type(e).__name__}: {e}"

    finally:
        cfg.BOT_VOL_FILTER_MIN_RANGE_PCT = orig_vol_cfg
        bt_mod.BOT_VOL_FILTER_MIN_RANGE_PCT = orig_vol_bt
        cfg.BOT_MONDAY_DOUBLE_CAPS = orig_mon_cfg
        bt_mod.BOT_MONDAY_DOUBLE_CAPS = orig_mon_bt


def _print_progress(master_log: str, completed: list[str],
                    total: int, t_start: float) -> None:
    """Tail per-variant logs and write an aggregated heartbeat."""
    lines = []
    lines.append(f"=== HEARTBEAT @ {_ts()} (elapsed {time.time()-t_start:.0f}s) ===")
    lines.append(f"  {len(completed)}/{total} variants done")

    # Tail every per-variant log to extract last status line
    for v in VARIANTS:
        nm = v["name"]
        log_path = os.path.join(OUTPUT_DIR, f"{nm}.log")
        if not os.path.exists(log_path):
            continue
        if nm in completed:
            continue
        try:
            with open(log_path) as f:
                tail = f.readlines()[-1].strip() if os.path.getsize(log_path) > 0 else ""
            lines.append(f"  {nm}: {tail}")
        except Exception:
            pass

    if completed:
        lines.append("  Recently completed:")
        for nm in completed[-5:]:
            jp = os.path.join(OUTPUT_DIR, f"{nm}.json")
            if os.path.exists(jp):
                try:
                    with open(jp) as f:
                        d = json.load(f)
                    if "error" in d:
                        lines.append(f"    {nm}: ERROR {d['error']}")
                    else:
                        lines.append(
                            f"    {nm}: ${d['pnl_per_day']:+.2f}/day, "
                            f"MaxDD ${d['max_dd']:.0f}, "
                            f"trades={d['trades']}, "
                            f"{d['elapsed_secs']:.0f}s"
                        )
                except Exception:
                    pass

    msg = "\n".join(lines) + "\n"
    with open(master_log, "a") as f:
        f.write(msg)
    print(msg, flush=True)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Wipe prior logs
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    master_log = os.path.join(OUTPUT_DIR, "MASTER.log")
    t_start = time.time()

    _log(master_log, f"=== FILTER AUDIT 2026-05-06 starting ===")
    _log(master_log, f"  {len(VARIANTS)} variants, 2 workers")
    _log(master_log, f"  output dir: {OUTPUT_DIR}")
    _log(master_log, f"  expected wall clock: ~75 min")
    print(f"=== filter audit starting — {len(VARIANTS)} variants, 2 workers ===",
          flush=True)
    print(f"  logs: {OUTPUT_DIR}/", flush=True)
    print(f"  master heartbeat: {master_log}", flush=True)

    completed: list[str] = []
    errored: list[tuple[str, str]] = []

    # Use fork — preserves sys.path and parent process state. Pure compute
    # task, no Cocoa/UI involvement, so fork is safe on Mac.
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=2, initializer=init_worker)

    try:
        # Submit all variants up-front; pool runs 2 at a time, queues rest.
        async_results = {
            v["name"]: pool.apply_async(run_variant, (v,)) for v in VARIANTS
        }

        _print_progress(master_log, completed, len(VARIANTS), t_start)

        # Poll for completion every 30s; print heartbeat each tick
        seen: set[str] = set()
        while True:
            for nm, ar in async_results.items():
                if nm in seen:
                    continue
                if ar.ready():
                    seen.add(nm)
                    try:
                        result_nm, status = ar.get(timeout=1)
                        if status == "ok":
                            completed.append(result_nm)
                            _log(master_log, f"  COMPLETED {result_nm}")
                            print(f"[{_ts()}] COMPLETED {result_nm}", flush=True)
                        else:
                            errored.append((result_nm, status))
                            _log(master_log, f"  ERRORED   {result_nm}: {status}")
                            print(f"[{_ts()}] ERRORED   {result_nm}: {status}",
                                  flush=True)
                    except Exception as e:
                        errored.append((nm, f"{type(e).__name__}: {e}"))
                        _log(master_log, f"  ERRORED   {nm}: {type(e).__name__}: {e}")
                        print(f"[{_ts()}] ERRORED   {nm}: {type(e).__name__}: {e}",
                              flush=True)

            if len(seen) >= len(VARIANTS):
                break

            time.sleep(30)
            _print_progress(master_log, completed, len(VARIANTS), t_start)

        _print_progress(master_log, completed, len(VARIANTS), t_start)

    finally:
        pool.close()
        pool.join()

    # Final summary
    _log(master_log, f"=== ALL VARIANTS DONE ({time.time()-t_start:.0f}s) ===")
    print(f"\n=== all variants done in {time.time()-t_start:.0f}s ===", flush=True)
    print(f"  completed: {len(completed)}/{len(VARIANTS)}", flush=True)
    if errored:
        print(f"  ERRORED: {len(errored)}", flush=True)
        for nm, msg in errored:
            print(f"    {nm}: {msg}", flush=True)

    # Aggregate all variant JSONs into a single results file
    aggregated = {"generated_at": _dt.datetime.now().isoformat(),
                  "elapsed_secs": time.time() - t_start,
                  "variants": {}}
    for v in VARIANTS:
        nm = v["name"]
        jp = os.path.join(OUTPUT_DIR, f"{nm}.json")
        if os.path.exists(jp):
            with open(jp) as f:
                aggregated["variants"][nm] = json.load(f)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"\n  aggregated results: {RESULTS_FILE}", flush=True)

    # Print final comparison table
    print_final_table(aggregated)


def print_final_table(agg: dict) -> None:
    BASELINE_PNL = 16.23  # known C+IBH=0.75 baseline (slippage-modeled, full sample)
    BASELINE_DD = 964
    print(f"\n{'='*120}")
    print(f"FILTER AUDIT — comparison vs C+IBH=0.75 baseline (${BASELINE_PNL:+.2f}/day, MaxDD ${BASELINE_DD})")
    print(f"{'='*120}")
    hdr = (f"{'Variant':<28} {'Label':<40} "
           f"{'$/day':>8} {'Δ$/day':>8} {'MaxDD':>7} {'ΔDD':>6} "
           f"{'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6} {'#trades':>8}")
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        nm = v["name"]
        if nm not in agg["variants"]:
            print(f"{nm:<28} (no result)")
            continue
        d = agg["variants"][nm]
        if "error" in d:
            print(f"{nm:<28} ERROR: {d['error'][:80]}")
            continue
        pday = d["pnl_per_day"]
        dd = d["max_dd"]
        per_q = d.get("per_quarter_pnl_per_day", {})
        print(f"{nm:<28} {d['label'][:38]:<40} "
              f"{pday:>+8.2f} {pday-BASELINE_PNL:>+8.2f} "
              f"{dd:>7.0f} {dd-BASELINE_DD:>+6.0f} "
              f"{per_q.get('Q1', 0):>+6.2f} {per_q.get('Q2', 0):>+6.2f} "
              f"{per_q.get('Q3', 0):>+6.2f} {per_q.get('Q4', 0):>+6.2f} "
              f"{d['trades']:>8d}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
