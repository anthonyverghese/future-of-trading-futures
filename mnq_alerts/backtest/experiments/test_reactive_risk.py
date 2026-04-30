"""Test reactive risk management — responds to what's happening TODAY.

Based on analysis of 50 bad days:
- Bad days trade 2x faster (14.3/hr vs 6.4/hr)
- After 2 consec losses: 0% recovery on bad, 100% on good
- 72% of bad-day losses in first hour
- FIB_0.236 is 54% WR on bad days vs 87% on good days

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_reactive_risk.py
"""
import os, sys, time
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.results import compute_stats

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH", "IBL"}

_DATES = None
_CACHES = None
_ARRAYS = None

VARIANTS = [
    ("Baseline", "baseline"),
    # 1. Stop after 2 consec losses within 15 min
    ("2 consec L within 15min", "consec_15min"),
    # 2. Stop after 2 consec losses if P&L negative
    ("2 consec L if P&L neg", "consec_pnl_neg"),
    # 3. Dynamic pace: pause 10 min after 4+ trades in 30 min
    ("Pause 10m after 4/30min", "dynamic_pace"),
    # 4. FIB_0.236 cap drops to 3 after its first loss
    ("0.236 cap=3 after loss", "fib236_reactive"),
    # 5. Trail $30 after $10+
    ("Trail $30 after $10+", "trail_30_10"),
    # 6. $75 loss limit
    ("$75 loss limit", "limit_75"),
    # 7. Combo: #2 + #4
    ("Consec L neg + 0.236 cap", "combo_consec_fib"),
    # 8. Combo: #4 + #5
    ("0.236 cap + trail", "combo_fib_trail"),
]


def _apply_filters(trades, variant, arrays_day):
    """Apply reactive filters to a day's trades.

    Returns filtered list of trades (post-filter approach).
    """
    if variant == "baseline":
        return trades

    filtered = []
    cum_pnl = 0.0
    peak_pnl = 0.0
    consec_losses = 0
    prev_loss_et = None
    trades_in_window = []  # (et_mins, timestamp) for pace tracking
    fib236_had_loss = False
    fib236_post_loss_count = 0
    stopped = False

    for t in trades:
        if stopped:
            break

        et = t.get('et_mins', 0)
        entry_ns = t.get('entry_ns', 0)
        pnl = t['pnl_usd']
        level = t['level']

        # --- Pre-trade checks ---

        # Dynamic pace: if 4+ trades in last 30 min, pause 10 min
        if variant in ("dynamic_pace",):
            cutoff_30 = entry_ns - 30 * 60 * 1_000_000_000
            recent = sum(1 for _, tns in trades_in_window if tns >= cutoff_30)
            if recent >= 4:
                # Check if 10 min have passed since the 4th trade
                cutoff_10 = entry_ns - 10 * 60 * 1_000_000_000
                fourth_recent = sum(1 for _, tns in trades_in_window if tns >= cutoff_10)
                if fourth_recent >= 4:
                    continue  # still in pause

        # FIB_0.236 reactive cap
        if variant in ("fib236_reactive", "combo_consec_fib", "combo_fib_trail"):
            if level == "FIB_0.236" and fib236_had_loss:
                if fib236_post_loss_count >= 3:
                    continue

        # Trail stop
        if variant in ("trail_30_10", "combo_fib_trail"):
            if peak_pnl >= 10.0 and cum_pnl < peak_pnl - 30.0:
                stopped = True
                break

        # $75 loss limit
        if variant == "limit_75":
            if cum_pnl <= -75:
                stopped = True
                break

        # --- Accept trade ---
        filtered.append(t)
        cum_pnl += pnl
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        trades_in_window.append((et, entry_ns))

        # --- Post-trade state updates ---

        if pnl < 0:
            # Track consecutive losses
            if consec_losses > 0 and prev_loss_et is not None:
                gap_min = et - prev_loss_et
                # 2 consec losses within 15 min → stop
                if variant in ("consec_15min",):
                    if gap_min <= 15:
                        stopped = True
                # 2 consec losses and P&L negative → stop
                if variant in ("consec_pnl_neg", "combo_consec_fib"):
                    if cum_pnl < 0:
                        stopped = True

            consec_losses += 1
            prev_loss_et = et

            # FIB_0.236 loss tracking
            if level == "FIB_0.236":
                fib236_had_loss = True
                fib236_post_loss_count = 0
        else:
            consec_losses = 0
            prev_loss_et = None

        # Track FIB_0.236 trades after its first loss
        if level == "FIB_0.236" and fib236_had_loss:
            fib236_post_loss_count += 1

    return filtered


def _run_one(args):
    name, variant = args
    all_trades = []
    streak = (0, 0)

    for date in _DATES:
        dc = _CACHES[date]
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        trades, streak = simulate_day(
            dc, _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=30,
        )

        # Convert TradeRecords to dicts for filtering
        trade_dicts = []
        for t in trades:
            td = {
                'level': t.level,
                'direction': t.direction,
                'pnl_usd': t.pnl_usd,
                'outcome': t.outcome,
                'et_mins': t.factors.et_mins if t.factors else 0,
                'entry_ns': t.entry_ns,
                'date': str(t.date),
            }
            trade_dicts.append(td)

        filtered_dicts = _apply_filters(trade_dicts, variant, _ARRAYS[date])

        # Map back to TradeRecords for compute_stats
        filtered_set = set()
        for i, td in enumerate(trade_dicts):
            for fd in filtered_dicts:
                if td is fd:
                    filtered_set.add(i)
                    break

        filtered_trades = [trades[i] for i in range(len(trades)) if i in filtered_set]
        all_trades.extend(filtered_trades)

    stats = compute_stats(all_trades, len(_DATES), list(_DATES))
    stats["name"] = name
    return stats


def main():
    global _DATES, _CACHES, _ARRAYS
    t0 = time.time()

    print("Loading data...", flush=True)
    _DATES, _CACHES = load_all_days()
    print(f"Loaded {len(_DATES)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    _ARRAYS = {d: precompute_arrays(_CACHES[d]) for d in _DATES}

    n_variants = len(VARIANTS)
    print(f"Running {n_variants} variants across 3 workers...", flush=True)

    with Pool(3) as pool:
        results = pool.map(_run_one, VARIANTS)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    baseline = results[0]
    b_pnl = baseline["pnl_per_day"]

    print("=" * 125)
    print(f"{'Variant':<28} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 125)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<28} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    print()
    print("Quarterly:")
    for r in results:
        q = r.get("quarterly_pnl_per_day", {})
        print(f"  {r['name']:<28} Q1={q.get('Q1_oldest',0):>+6.1f} Q2={q.get('Q2',0):>+6.1f} Q3={q.get('Q3',0):>+6.1f} Q4={q.get('Q4_newest',0):>+6.1f}")

    winners = [r for r in results[1:] if r["pnl_per_day"] > b_pnl]
    if winners:
        print(f"\n  Variants that beat baseline (${b_pnl:.2f}/day):")
        for r in sorted(winners, key=lambda x: x["pnl_per_day"], reverse=True):
            diff = r["pnl_per_day"] - b_pnl
            print(f"    {r['name']:<28} ${r['pnl_per_day']:>+.2f}/day ({diff:>+.2f})")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"reactive_risk_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
