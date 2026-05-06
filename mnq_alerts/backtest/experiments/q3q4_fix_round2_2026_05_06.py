"""q3q4_fix_round2_2026_05_06.py — round 2 of Q3+Q4 fix variants.

Round 1 (q3q4_fix_variants_2026_05_06) found V13 (drop FIB_0.236) is the
closest to passing acceptance — Q4 +$1.27 better, full -$2.24 worse.

Round 2 hypothesis: FIB_0.618 is the only level GETTING STRONGER (Q1
+$2.01/tr → Q4 +$6.52/tr, recent 60d $13.12/day at cap=3). EXPANDING
its cap should compound the lift on Q3+Q4 while offsetting Q1+Q2 loss.

Variants tested (all on V6 baseline = Monday caps off):
  V19: just FIB_0.618 cap 3→6 (isolated test of cap expansion)
  V20: just FIB_0.618 cap 3→10 (more aggressive)
  V16: V13 (drop FIB_0.236) + FIB_0.618 cap 3→6
  V17: V13 + FIB_0.618 cap 3→10
  V18: V13 + V14 (drop IBH too) + FIB_0.618 cap 3→6

Acceptance: full ≥ $17.87, Q3 ≥ $8.81, Q4 ≥ $10.91 (V6 baseline).

2 parallel workers, ~30 min wall clock.
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

OUTPUT_DIR = "/tmp/q3q4_fix_round2"
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "results",
    "q3q4_fix_round2_2026_05_06.json",
)

_CACHES = None
_DATES = None

DEFAULT_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25), "IBH": (6, 20),
}
BASE_CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}
DEFAULT_EXCLUDE = {"FIB_EXT_LO_1.272", "FIB_0.5"}
DEFAULT_DIR_FILTER = {"IBH": "down"}

VARIANTS = [
    {
        "name": "V19_FIB618_cap6",
        "label": "FIB_0.618 cap 3→6 only (isolated cap expansion)",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": DEFAULT_DIR_FILTER,
        "caps": {**BASE_CAPS, "FIB_0.618": 6},
        "include_ibh": True,
    },
    {
        "name": "V20_FIB618_cap10",
        "label": "FIB_0.618 cap 3→10 (aggressive isolated expansion)",
        "exclude": DEFAULT_EXCLUDE,
        "direction_filter": DEFAULT_DIR_FILTER,
        "caps": {**BASE_CAPS, "FIB_0.618": 10},
        "include_ibh": True,
    },
    {
        "name": "V16_drop236_cap618_6",
        "label": "Drop FIB_0.236 + FIB_0.618 cap 3→6",
        "exclude": DEFAULT_EXCLUDE | {"FIB_0.236"},
        "direction_filter": DEFAULT_DIR_FILTER,
        "caps": {**BASE_CAPS, "FIB_0.618": 6},
        "include_ibh": True,
    },
    {
        "name": "V17_drop236_cap618_10",
        "label": "Drop FIB_0.236 + FIB_0.618 cap 3→10",
        "exclude": DEFAULT_EXCLUDE | {"FIB_0.236"},
        "direction_filter": DEFAULT_DIR_FILTER,
        "caps": {**BASE_CAPS, "FIB_0.618": 10},
        "include_ibh": True,
    },
    {
        "name": "V18_drop236_dropIBH_cap618_6",
        "label": "Drop FIB_0.236 + drop IBH + FIB_0.618 cap 3→6",
        "exclude": DEFAULT_EXCLUDE | {"FIB_0.236"},
        "direction_filter": {},  # no direction filter; IBH excluded
        "caps": {**BASE_CAPS, "FIB_0.618": 6},
        "include_ibh": False,
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
    _log(log_path, f"Worker {pid} starting...")
    try:
        from mnq_alerts.backtest.data import load_all_days
        t0 = time.time()
        _DATES, _CACHES = load_all_days()
        _log(log_path, f"Worker {pid} ready in {time.time()-t0:.0f}s")
    except Exception as e:
        _log(log_path, f"Worker {pid} INIT FAILED: {e}")
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
                if t.pnl_usd >= 0: s[1] += 1
                s[2] += t.pnl_usd
                by_outcome[t.outcome] += 1
            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.time() - t0
                eta = (n - i - 1) * (elapsed / (i + 1))
                _log(log_path, f"  day {i+1}/{n} | {total_trades} trades | "
                     f"${total_pnl/(i+1):+.2f}/day | ETA {eta:.0f}s")

        cum = peak = max_dd = 0.0
        sorted_dates = sorted(day_pnl.keys())
        for d in sorted_dates:
            cum += day_pnl[d]
            if cum > peak: peak = cum
            if peak - cum > max_dd: max_dd = peak - cum

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
            qcum = qpeak = qdd = 0.0
            for d in qdates:
                qcum += day_pnl[d]
                if qcum > qpeak: qpeak = qcum
                if qpeak - qcum > qdd: qdd = qpeak - qcum
            per_q_dd[q_label] = qdd

        recent_dates = sorted_dates[-60:]
        recent_per_day = sum(day_pnl[d] for d in recent_dates) / 60

        result = {
            "name": name, "label": label,
            "trades": total_trades, "days": n_days,
            "pnl_total": total_pnl,
            "pnl_per_day": total_pnl / n_days,
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
             f"DONE  trades={total_trades} $/day=${total_pnl/n_days:+.2f} "
             f"MaxDD=${max_dd:.0f} recent60d=${recent_per_day:+.2f} "
             f"per-Q: Q1=${per_q['Q1']:+.2f} Q2=${per_q['Q2']:+.2f} "
             f"Q3=${per_q['Q3']:+.2f} Q4=${per_q['Q4']:+.2f}")
        return name, "ok"
    except Exception as e:
        tb = traceback.format_exc()
        _log(log_path, f"ERROR {e}")
        _log(log_path, tb)
        with open(json_path, "w") as f:
            json.dump({"name": name, "error": str(e), "traceback": tb}, f, indent=2)
        return name, f"error:{e}"


def _print_progress(master_log, completed, total, t_start):
    lines = [f"=== HEARTBEAT @ {_ts()} (elapsed {time.time()-t_start:.0f}s) ===",
             f"  {len(completed)}/{total} done"]
    for v in VARIANTS:
        nm = v["name"]
        lp = os.path.join(OUTPUT_DIR, f"{nm}.log")
        if not os.path.exists(lp) or nm in completed: continue
        try:
            with open(lp) as f:
                tail = f.readlines()[-1].strip() if os.path.getsize(lp) > 0 else ""
            lines.append(f"  {nm}: {tail}")
        except: pass
    if completed:
        lines.append("  Completed:")
        for nm in completed[-5:]:
            jp = os.path.join(OUTPUT_DIR, f"{nm}.json")
            if os.path.exists(jp):
                try:
                    with open(jp) as f: d = json.load(f)
                    if "error" in d:
                        lines.append(f"    {nm}: ERROR")
                    else:
                        lines.append(f"    {nm}: ${d['pnl_per_day']:+.2f}/day, "
                                     f"recent60d=${d['recent_60d_pnl_per_day']:+.2f}, "
                                     f"Q1=${d['per_quarter_pnl_per_day']['Q1']:+.2f} "
                                     f"Q2=${d['per_quarter_pnl_per_day']['Q2']:+.2f} "
                                     f"Q3=${d['per_quarter_pnl_per_day']['Q3']:+.2f} "
                                     f"Q4=${d['per_quarter_pnl_per_day']['Q4']:+.2f}")
                except: pass
    msg = "\n".join(lines) + "\n"
    with open(master_log, "a") as f:
        f.write(msg)
    print(msg, flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))
    master_log = os.path.join(OUTPUT_DIR, "MASTER.log")
    t_start = time.time()
    _log(master_log, f"=== ROUND 2: FIB_0.618 cap expansion variants ===")
    completed = []
    errored = []
    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=2, initializer=init_worker)
    try:
        async_results = {v["name"]: pool.apply_async(run_variant, (v,)) for v in VARIANTS}
        _print_progress(master_log, completed, len(VARIANTS), t_start)
        seen = set()
        while True:
            for nm, ar in async_results.items():
                if nm in seen: continue
                if ar.ready():
                    seen.add(nm)
                    try:
                        rn, status = ar.get(timeout=1)
                        if status == "ok":
                            completed.append(rn)
                            _log(master_log, f"  COMPLETED {rn}")
                        else:
                            errored.append((rn, status))
                    except Exception as e:
                        errored.append((nm, str(e)))
            if len(seen) >= len(VARIANTS): break
            time.sleep(30)
            _print_progress(master_log, completed, len(VARIANTS), t_start)
        _print_progress(master_log, completed, len(VARIANTS), t_start)
    finally:
        pool.close()
        pool.join()

    aggregated = {
        "generated_at": _dt.datetime.now().isoformat(),
        "elapsed_secs": time.time() - t_start,
        "v6_baseline": {
            "pnl_per_day": 17.87, "max_dd": 857, "recent_60d": 13.76,
            "per_quarter": {"Q1": 33.36, "Q2": 18.39, "Q3": 8.81, "Q4": 10.91},
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

    BASE_PNL = 17.87
    BASE_DD = 857
    BASE_Q = {"Q1": 33.36, "Q2": 18.39, "Q3": 8.81, "Q4": 10.91}
    print(f"\n{'='*120}")
    print(f"ROUND 2 — vs V6 baseline (${BASE_PNL:+.2f}/day)")
    print(f"{'='*120}")
    print(f"{'Variant':<32} {'$/day':>8} {'Δ':>7} {'MaxDD':>6} {'rec60':>6} "
          f"{'Q1':>7} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'#tr':>5}  {'Pass':>5}")
    for v in VARIANTS:
        nm = v["name"]
        if nm not in aggregated["variants"]: continue
        d = aggregated["variants"][nm]
        if "error" in d: continue
        pq = d["per_quarter_pnl_per_day"]
        full_status = "✓" if d["pnl_per_day"] >= BASE_PNL - 0.5 else "❌"
        q3_status = "✓" if pq["Q3"] >= BASE_Q["Q3"] - 0.5 else "❌"
        q4_status = "✓" if pq["Q4"] >= BASE_Q["Q4"] - 0.5 else "❌"
        print(f"{nm:<32} ${d['pnl_per_day']:>+7.2f} ${d['pnl_per_day']-BASE_PNL:>+6.2f} "
              f"${d['max_dd']:>5.0f} ${d['recent_60d_pnl_per_day']:>+5.2f} "
              f"${pq['Q1']:>+6.2f} ${pq['Q2']:>+6.2f} ${pq['Q3']:>+6.2f} ${pq['Q4']:>+6.2f} "
              f"{d['trades']:>5d}  {full_status}{q3_status}{q4_status}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
