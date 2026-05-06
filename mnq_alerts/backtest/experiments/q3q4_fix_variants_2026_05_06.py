"""q3q4_fix_variants_2026_05_06.py — real-sim variants targeting Q3+Q4 lift.

Diagnostic showed (q3q4_diagnostic_2026_05_06.json):
  - FIB_EXT_HI SELL: -$1.92/day in Q3+Q4 (n=461)
  - FIB_0.236 BUY:   -$1.23/day in Q3+Q4 (n=984)
  - FIB_0.764 SELL:  -$0.84/day in Q3+Q4 (n=390)
  - Recent 60d: FIB_0.236 -$2.27/day, IBH -$1.03/day, FIB_0.618 +$13.12/day

Tests 6 fix variants under V6 (Monday caps off) baseline:
  V7a: Drop FIB_0.236 BUY only (surgical)
  V7b: Drop the 3 failing direction combos (aggressive)
  V11: Reduce FIB_0.236 cap 18→10, IBH cap 7→4
  V13: Drop FIB_0.236 entirely
  V14: Drop IBH entirely
  V15: V7b + cap reduction (combined)

Walk-forward acceptance: Q3 ≥ 8.0, Q4 ≥ 10.0, full sample ≥ V6's $17.87.

2 parallel workers. Same architecture as filter_audit_2026_05_06.py.
Wall clock: ~45 min.
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

OUTPUT_DIR = "/tmp/q3q4_fix_audit"
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "q3q4_fix_variants_2026_05_06.json",
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
# V6 baseline: Monday caps OFF (deployed)
DEFAULT_EXCLUDE = {"FIB_EXT_LO_1.272", "FIB_0.5"}
DEFAULT_DIR_FILTER = {"IBH": "down"}

VARIANTS = [
    {
        "name": "V6_baseline_rerun",
        "label": "V6 baseline (rerun for clean comparison)",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": DEFAULT_DIR_FILTER,
        "caps": DEFAULT_CAPS,
        "include_ibh": True,
    },
    {
        "name": "V7a_drop_FIB236_BUY",
        "label": "Drop FIB_0.236 BUY (surgical)",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": {"IBH": "down", "FIB_0.236": "down"},
        "caps": DEFAULT_CAPS,
        "include_ibh": True,
    },
    {
        "name": "V7b_drop_3_combos",
        "label": "Drop FIB_EXT_HI SELL + FIB_0.236 BUY + FIB_0.764 SELL",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": {
            "IBH": "down",
            "FIB_EXT_HI_1.272": "up",  # SELL only ban means UP only (BUY allowed)
            "FIB_0.236": "down",        # SELL allowed, BUY blocked
            "FIB_0.764": "up",          # BUY allowed, SELL blocked
        },
        "caps": DEFAULT_CAPS,
        "include_ibh": True,
    },
    {
        "name": "V11_reduce_FIB236_IBH_caps",
        "label": "Reduce FIB_0.236 cap 18→10, IBH cap 7→4",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": DEFAULT_DIR_FILTER,
        "caps": {**DEFAULT_CAPS, "FIB_0.236": 10, "IBH": 4},
        "include_ibh": True,
    },
    {
        "name": "V13_drop_FIB236",
        "label": "Drop FIB_0.236 entirely (worst recent-60d level)",
        "exclude": DEFAULT_EXCLUDE | {"FIB_0.236"},
        "direction_filter": DEFAULT_DIR_FILTER,
        "caps": DEFAULT_CAPS,
        "include_ibh": True,
    },
    {
        "name": "V14_drop_IBH",
        "label": "Drop IBH entirely (Q4 -$3.13/day, recent 60d -$1.03/day)",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": {},  # no direction filter needed if IBH excluded
        "caps": DEFAULT_CAPS,
        "include_ibh": False,
    },
    {
        "name": "V15_combined",
        "label": "V7b + cap reduction (combined fix)",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": {
            "IBH": "down",
            "FIB_EXT_HI_1.272": "up",
            "FIB_0.236": "down",
            "FIB_0.764": "up",
        },
        "caps": {**DEFAULT_CAPS, "FIB_0.236": 10, "IBH": 4},
        "include_ibh": True,
    },
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
    _log(log_path, f"Worker {pid} starting — loading caches...")
    try:
        from mnq_alerts.backtest.data import load_all_days
        t0 = time.time()
        _DATES, _CACHES = load_all_days()
        _log(log_path, f"Worker {pid} ready — {len(_DATES)} days in {time.time()-t0:.0f}s")
    except Exception as e:
        _log(log_path, f"Worker {pid} INIT FAILED: {type(e).__name__}: {e}")
        raise


def run_variant(variant: dict) -> tuple[str, str]:
    name = variant["name"]
    label = variant["label"]
    log_path = os.path.join(OUTPUT_DIR, f"{name}.log")
    json_path = os.path.join(OUTPUT_DIR, f"{name}.json")

    _log(log_path, f"START {label}")

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

            with redirect_stdout(sink):
                trades = simulate_day_v2(
                    dc, arr,
                    per_level_ts=DEFAULT_TS,
                    per_level_caps=variant["caps"],
                    exclude_levels=variant["exclude"],
                    include_ibl=False,
                    include_vwap=False,
                    direction_filter=variant["direction_filter"],
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

        # Recent 60-day stats
        recent_dates = sorted_dates[-60:]
        recent_pnl = sum(day_pnl[d] for d in recent_dates)
        recent_per_day = recent_pnl / 60

        result = {
            "name": name, "label": label,
            "trades": total_trades, "days": n_days,
            "pnl_total": total_pnl,
            "pnl_per_day": total_pnl / n_days if n_days else 0.0,
            "max_dd": max_dd,
            "per_quarter_pnl_per_day": per_q,
            "per_quarter_max_dd": per_q_dd,
            "recent_60d_pnl_per_day": recent_per_day,
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
             f"recent60d=${recent_per_day:+.2f}/day "
             f"per-Q: Q1=${per_q['Q1']:+.2f} Q2=${per_q['Q2']:+.2f} "
             f"Q3=${per_q['Q3']:+.2f} Q4=${per_q['Q4']:+.2f} "
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
                            f"recent60d=${d['recent_60d_pnl_per_day']:+.2f}/d, "
                            f"Q1=${d['per_quarter_pnl_per_day']['Q1']:+.2f} "
                            f"Q2=${d['per_quarter_pnl_per_day']['Q2']:+.2f} "
                            f"Q3=${d['per_quarter_pnl_per_day']['Q3']:+.2f} "
                            f"Q4=${d['per_quarter_pnl_per_day']['Q4']:+.2f}"
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
    _log(master_log, f"=== Q3+Q4 FIX VARIANT AUDIT ===")
    _log(master_log, f"  {len(VARIANTS)} variants, 2 workers")

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
                        _log(master_log, f"  ERRORED   {nm}: {e}")
            if len(seen) >= len(VARIANTS):
                break
            time.sleep(30)
            _print_progress(master_log, completed, len(VARIANTS), t_start)
        _print_progress(master_log, completed, len(VARIANTS), t_start)
    finally:
        pool.close()
        pool.join()

    print(f"\n=== all variants done in {time.time()-t_start:.0f}s ===", flush=True)

    aggregated = {
        "generated_at": _dt.datetime.now().isoformat(),
        "elapsed_secs": time.time() - t_start,
        "v6_baseline_pnl_per_day": 17.87,
        "v6_baseline_max_dd": 857,
        "v6_baseline_per_quarter": {
            "Q1": 33.36, "Q2": 18.39, "Q3": 8.81, "Q4": 10.91,
        },
        "variants": {},
    }
    for v in VARIANTS:
        nm = v["name"]
        jp = os.path.join(OUTPUT_DIR, f"{nm}.json")
        if os.path.exists(jp):
            with open(jp) as f:
                aggregated["variants"][nm] = json.load(f)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_FILE}", flush=True)

    # Final comparison
    BASE_PNL = 17.87
    BASE_DD = 857
    BASE_Q = {"Q1": 33.36, "Q2": 18.39, "Q3": 8.81, "Q4": 10.91}
    print(f"\n{'='*120}")
    print(f"Q3+Q4 FIX VARIANTS — vs V6 baseline (${BASE_PNL:+.2f}/day, MaxDD ${BASE_DD})")
    print(f"{'='*120}")
    print(f"{'Variant':<28} {'$/day':>8} {'Δ':>7} {'MaxDD':>6} {'rec60':>6} "
          f"{'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'#tr':>5}")
    for v in VARIANTS:
        nm = v["name"]
        if nm not in aggregated["variants"]:
            continue
        d = aggregated["variants"][nm]
        if "error" in d:
            print(f"{nm:<28} ERROR")
            continue
        pq = d["per_quarter_pnl_per_day"]
        q3_status = "✓" if pq["Q3"] >= BASE_Q["Q3"] - 0.5 else "❌"
        q4_status = "✓" if pq["Q4"] >= BASE_Q["Q4"] - 0.5 else "❌"
        full_status = "✓" if d["pnl_per_day"] >= BASE_PNL - 0.5 else "❌"
        flags = f"{full_status}{q3_status}{q4_status}"
        print(f"{nm:<28} ${d['pnl_per_day']:>+7.2f} "
              f"${d['pnl_per_day']-BASE_PNL:>+6.2f} "
              f"${d['max_dd']:>5.0f} ${d['recent_60d_pnl_per_day']:>+5.2f} "
              f"${pq['Q1']:>+6.2f} ${pq['Q2']:>+6.2f} "
              f"${pq['Q3']:>+6.2f} ${pq['Q4']:>+6.2f} "
              f"{d['trades']:>5d}  {flags}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
