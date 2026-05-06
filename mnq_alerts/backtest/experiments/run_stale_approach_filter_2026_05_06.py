"""run_stale_approach_filter_2026_05_06.py — real-sim of the stale-approach filter.

Tests the BOT_MAX_SECS_SINCE_LAST_TRADE_THIS_DAY filter (Phase C candidate)
under slippage-modeled real-sim with cap-budget effects. Three thresholds:
  S20: 1200s (20 min)
  S30: 1800s (30 min)  ← Phase C signal threshold
  S45: 2700s (45 min)  ← wider, less aggressive

Compares each to the freshly-computed V0_baseline (+$16.23/day, MaxDD $964)
with per-quarter walk-forward.

Architecture mirrors filter_audit_2026_05_06: 2 parallel workers (mp.Pool
with fork context), each loads caches once, processes assigned variants,
writes JSON + log per variant. Master prints heartbeat every 30s.

Wall clock estimate: ~3 min cache load + ~9 min/sim × ceil(3/2) batches
= ~25-30 min total.
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

OUTPUT_DIR = "/tmp/stale_filter_audit"
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "stale_approach_filter_2026_05_06.json",
)

_CACHES = None
_DATES = None

DEFAULT_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25), "IBH": (6, 20),
}
DEFAULT_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}

VARIANTS = [
    {"name": "S20_max_gap_1200s", "label": "Max gap 20min (1200s)", "max_gap": 1200},
    {"name": "S30_max_gap_1800s", "label": "Max gap 30min (1800s) — Phase C signal", "max_gap": 1800},
    {"name": "S45_max_gap_2700s", "label": "Max gap 45min (2700s)", "max_gap": 2700},
]


def _ts() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def _log(path: str, msg: str) -> None:
    with open(path, "a") as f:
        f.write(f"[{_ts()}] {msg}\n")


def init_worker() -> None:
    global _CACHES, _DATES
    pid = os.getpid()
    log_path = os.path.join(OUTPUT_DIR, f"worker_{pid}.log")
    _log(log_path, f"Worker {pid} starting — loading day caches...")
    try:
        from mnq_alerts.backtest.data import load_all_days
        t0 = time.time()
        _DATES, _CACHES = load_all_days()
        _log(log_path, f"Worker {pid} ready — {len(_DATES)} days "
             f"in {time.time()-t0:.0f}s")
    except Exception as e:
        _log(log_path, f"Worker {pid} INIT FAILED: {type(e).__name__}: {e}")
        _log(log_path, traceback.format_exc())
        raise


def run_variant(variant: dict) -> tuple[str, str]:
    name = variant["name"]
    label = variant["label"]
    max_gap = variant["max_gap"]
    log_path = os.path.join(OUTPUT_DIR, f"{name}.log")
    json_path = os.path.join(OUTPUT_DIR, f"{name}.json")

    _log(log_path, f"START {label} (max_gap={max_gap}s)")

    try:
        from mnq_alerts.backtest.simulate_v2 import simulate_day_v2
        from mnq_alerts.backtest.data import precompute_arrays

        sink = StringIO()
        total_pnl = 0.0
        total_trades = 0
        by_level = defaultdict(lambda: [0, 0, 0.0])
        by_outcome: dict = defaultdict(int)
        day_pnl: dict = {}

        n = len(_DATES)
        t0 = time.time()
        _log(log_path, f"Sim starting over {n} days...")

        for i, date in enumerate(_DATES):
            dc = _CACHES[date]
            arr = precompute_arrays(dc)

            # Match audit baseline config: deployed C+IBH=0.75 + V6 (no Mon caps)
            # We're testing on TOP of the V6 deploy candidate.
            day_caps = dict(DEFAULT_CAPS)
            # Note: NO Monday-doubling — we're testing on top of V6.

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
                    max_secs_since_last_trade=max_gap,
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
                     f"${total_pnl/(i+1):+.2f}/day | ETA {eta:.0f}s")

        # MaxDD
        cum = 0.0; peak = 0.0; max_dd = 0.0
        sorted_dates = sorted(day_pnl.keys())
        for d in sorted_dates:
            cum += day_pnl[d]
            if cum > peak:
                peak = cum
            if peak - cum > max_dd:
                max_dd = peak - cum

        # Per-Q
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
            qcum = 0.0; qpeak = 0.0; qdd = 0.0
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
            "max_secs_since_last_trade": max_gap,
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


def _print_progress(master_log: str, completed: list[str],
                    total: int, t_start: float) -> None:
    lines = [f"=== HEARTBEAT @ {_ts()} (elapsed {time.time()-t_start:.0f}s) ===",
             f"  {len(completed)}/{total} variants done"]
    for v in VARIANTS:
        nm = v["name"]
        lp = os.path.join(OUTPUT_DIR, f"{nm}.log")
        if not os.path.exists(lp) or nm in completed:
            continue
        try:
            with open(lp) as f:
                tail = f.readlines()[-1].strip() if os.path.getsize(lp) > 0 else ""
            lines.append(f"  {nm}: {tail}")
        except Exception:
            pass
    if completed:
        lines.append("  Completed:")
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
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    master_log = os.path.join(OUTPUT_DIR, "MASTER.log")
    t_start = time.time()
    _log(master_log, f"=== STALE-APPROACH FILTER REAL-SIM ===")
    _log(master_log, f"  {len(VARIANTS)} variants, 2 workers")
    _log(master_log, f"  output: {OUTPUT_DIR}")

    completed: list[str] = []
    errored: list[tuple[str, str]] = []

    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=2, initializer=init_worker)

    try:
        async_results = {
            v["name"]: pool.apply_async(run_variant, (v,)) for v in VARIANTS
        }
        _print_progress(master_log, completed, len(VARIANTS), t_start)
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
                        else:
                            errored.append((result_nm, status))
                            _log(master_log, f"  ERRORED   {result_nm}: {status}")
                    except Exception as e:
                        errored.append((nm, f"{type(e).__name__}: {e}"))
                        _log(master_log, f"  ERRORED   {nm}: {type(e).__name__}: {e}")
            if len(seen) >= len(VARIANTS):
                break
            time.sleep(30)
            _print_progress(master_log, completed, len(VARIANTS), t_start)
        _print_progress(master_log, completed, len(VARIANTS), t_start)
    finally:
        pool.close()
        pool.join()

    print(f"\n=== all variants done in {time.time()-t_start:.0f}s ===", flush=True)

    # Aggregate
    aggregated = {"generated_at": _dt.datetime.now().isoformat(),
                  "elapsed_secs": time.time() - t_start,
                  "v0_baseline_pnl_per_day": 16.23,
                  "v0_baseline_max_dd": 964,
                  "v0_baseline_per_quarter": {
                      "Q1": 29.74, "Q2": 18.93, "Q3": 6.15, "Q4": 10.09
                  },
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
    print(f"\n  results: {RESULTS_FILE}", flush=True)

    # Final table
    BASE_PNL = 16.23
    BASE_DD = 964
    BASE_Q = {"Q1": 29.74, "Q2": 18.93, "Q3": 6.15, "Q4": 10.09}
    print(f"\n{'='*120}")
    print(f"STALE-APPROACH FILTER — vs V0_baseline (${BASE_PNL:+.2f}/day, MaxDD ${BASE_DD})")
    print(f"{'='*120}")
    print(f"{'Variant':<28} {'$/day':>8} {'Δ':>8} {'MaxDD':>7} {'ΔDD':>6} "
          f"{'Q1':>7} {'ΔQ1':>7} {'Q2':>7} {'ΔQ2':>7} {'Q3':>7} {'ΔQ3':>7} "
          f"{'Q4':>7} {'ΔQ4':>7} {'#tr':>6}")
    for v in VARIANTS:
        nm = v["name"]
        if nm not in aggregated["variants"]:
            print(f"{nm:<28} (no result)")
            continue
        d = aggregated["variants"][nm]
        if "error" in d:
            print(f"{nm:<28} ERROR: {d['error'][:80]}")
            continue
        pq = d["per_quarter_pnl_per_day"]
        print(f"{nm:<28} {d['pnl_per_day']:>+8.2f} "
              f"{d['pnl_per_day']-BASE_PNL:>+8.2f} "
              f"{d['max_dd']:>7.0f} {d['max_dd']-BASE_DD:>+6.0f} "
              f"{pq['Q1']:>+7.2f} {pq['Q1']-BASE_Q['Q1']:>+7.2f} "
              f"{pq['Q2']:>+7.2f} {pq['Q2']-BASE_Q['Q2']:>+7.2f} "
              f"{pq['Q3']:>+7.2f} {pq['Q3']-BASE_Q['Q3']:>+7.2f} "
              f"{pq['Q4']:>+7.2f} {pq['Q4']-BASE_Q['Q4']:>+7.2f} "
              f"{d['trades']:>6d}")
    print(f"{'='*120}")
    # Walk-forward verdict per variant
    print()
    for v in VARIANTS:
        nm = v["name"]
        if nm not in aggregated["variants"] or "error" in aggregated["variants"][nm]:
            continue
        d = aggregated["variants"][nm]
        pq = d["per_quarter_pnl_per_day"]
        deltas = {q: pq[q] - BASE_Q[q] for q in ["Q1", "Q2", "Q3", "Q4"]}
        positives = sum(1 for d in deltas.values() if d > 0.5)
        negatives = sum(1 for d in deltas.values() if d < -0.5)
        if positives >= 3 and negatives == 0:
            verdict = "ROBUST — deploy candidate (3-4/4 positive)"
        elif positives >= 3:
            verdict = "MIXED — partial robust (3+/4 pos but with negatives)"
        elif positives == 2:
            verdict = "WEAK — 2/4 positive, regime-coupled risk"
        else:
            verdict = "REJECTED — fewer than 2/4 quarters positive"
        print(f"  {nm}: {verdict}")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            print(f"    {q}: ${pq[q]:+6.2f} (Δ{deltas[q]:+5.2f})")


if __name__ == "__main__":
    main()
