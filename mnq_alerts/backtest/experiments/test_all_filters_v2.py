"""Test each defensive factor individually using v2 accurate simulation.

For each factor, compare baseline (all factors on) vs that factor removed.
Shows whether each factor improves P&L and reduces MaxDD.

Usage:
    cd /Users/anthonyverghese/future-of-trading-futures
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_all_filters_v2.py
"""
import datetime, sys, os
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

dates, caches = load_all_days()
print("Loaded %d days" % len(dates), flush=True)

TS = {"FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
      "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
      "IBH": (6, 20), "IBL": (6, 20), "VWAP": (8, 25), "FIB_0.5": (10, 25)}
BASE_CAPS = {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
             "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7}
HIGH_CAPS = {k: 99 for k in BASE_CAPS}

# Each variant: (name, kwargs_overrides)
# Baseline uses: caps=BASE_CAPS, exclude={FIB_0.5,IBL}, dir_filter={IBH:down},
#                daily_loss=200, timeout=900, momentum=0, include_ibl=False, include_vwap=False
VARIANTS = [
    ("Baseline (deployed)", {}),
    # 1. Remove per-level caps
    ("No per-level caps", {"per_level_caps": HIGH_CAPS}),
    # 2. Remove Monday double caps (test by running all days with non-Monday caps)
    # Can't directly test this through simulate_day_v2 params since Monday doubling
    # is inside BotTrader. Instead test by comparing Monday vs non-Monday performance.
    # 3. Remove IBH SELL only (allow both directions)
    ("IBH both directions", {"direction_filter": {}}),
    # 4. Remove daily loss limit
    ("No daily loss limit", {"daily_loss": 9999.0}),
    # 5. Different position timeouts
    ("Timeout 5 min", {"timeout_secs": 300}),
    ("Timeout 10 min", {"timeout_secs": 600}),
    ("Timeout 20 min", {"timeout_secs": 1200}),
    ("Timeout 30 min", {"timeout_secs": 1800}),
    # 6. Remove volatility filter — need to check if v2 supports this
    # Vol filter is in BotTrader via BOT_VOL_FILTER_MIN_RANGE_PCT config
    # Can't override via simulate_day_v2 params. Skip for now.
    # 7. Bring back excluded levels
    ("+ IBL", {"exclude_levels": {"FIB_0.5"}, "include_ibl": True}),
    ("+ VWAP", {"exclude_levels": {"FIB_0.5", "IBL"}, "include_vwap": True}),
    ("+ FIB_0.5", {"exclude_levels": {"IBL"}}),
    ("+ All excluded levels", {"exclude_levels": set(), "include_ibl": True, "include_vwap": True}),
]

def run_variant(name, overrides):
    kwargs = {
        "per_level_ts": TS,
        "per_level_caps": dict(BASE_CAPS),
        "exclude_levels": {"FIB_0.5", "IBL"},
        "direction_filter": {"IBH": "down"},
        "daily_loss": 200.0,
        "timeout_secs": 900,
        "momentum_max": 0.0,
        "include_ibl": False,
        "include_vwap": False,
    }
    kwargs.update(overrides)

    all_trades = []
    for date in dates:
        dc = caches[date]
        arr = precompute_arrays(dc)
        caps = dict(kwargs["per_level_caps"])
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        day_kwargs = dict(kwargs)
        day_kwargs["per_level_caps"] = caps
        trades = simulate_day_v2(dc, arr, **day_kwargs)
        all_trades.extend([(date, t) for t in trades])
    return all_trades

def compute_stats(all_trades):
    n = len(dates)
    pnl = sum(t.pnl_usd for _, t in all_trades)
    wins = sum(1 for _, t in all_trades if t.pnl_usd >= 0)
    cum = 0.0; peak = 0.0; max_dd = 0.0
    for _, t in all_trades:
        cum += t.pnl_usd
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd
    day_pnl = defaultdict(float)
    for d, t in all_trades:
        day_pnl[d] += t.pnl_usd
    bad = sum(1 for p in day_pnl.values() if p <= -200)
    return {
        "trades": len(all_trades), "wins": wins,
        "wr": wins / len(all_trades) * 100 if all_trades else 0,
        "pnl_day": pnl / n, "max_dd": max_dd, "bad_days": bad,
    }

_DATES = None
_CACHES = None

def _run_one(args):
    import time as _time
    t0 = _time.time()
    name, overrides = args
    trades = run_variant(name, overrides)
    stats = compute_stats(trades)
    stats["name"] = name
    elapsed = _time.time() - t0
    # Write progress to a file (bypasses stdout buffering)
    with open("/tmp/backtest_progress_all_filters.txt", "a") as f:
        f.write("Done: %s — %d trades, $%.2f/day, %.1f min\n" % (
            name, stats["trades"], stats["pnl_day"], elapsed / 60))
        f.flush()
    return stats

_DATES = dates
_CACHES = caches

# Clear progress file
with open("/tmp/backtest_progress_all_filters.txt", "w") as f:
    f.write("Starting %d variants across 3 workers...\n" % len(VARIANTS))

print("\nRunning %d variants across 3 workers..." % len(VARIANTS), flush=True)
with Pool(3) as pool:
    results = pool.map(_run_one, VARIANTS)

baseline = results[0]
print("\n" + "=" * 100)
print("%-30s %6s %5s %8s %6s %6s %8s" % (
    "Variant", "Trades", "WR%", "$/day", "MaxDD", "Bad", "vs base"))
print("-" * 100)
for r in results:
    diff = r["pnl_day"] - baseline["pnl_day"]
    print("%-30s %6d %5.1f %+8.2f %6.0f %6d %+8.2f" % (
        r["name"], r["trades"], r["wr"], r["pnl_day"],
        r["max_dd"], r["bad_days"], diff))
