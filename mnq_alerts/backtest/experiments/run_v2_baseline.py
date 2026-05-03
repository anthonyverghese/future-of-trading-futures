"""Run deployed config through simulate_day_v2 for accurate baseline."""
import datetime, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

dates, caches = load_all_days()
print(f"Loaded {len(dates)} days", flush=True)

all_trades = []
for di, date in enumerate(dates):
    if di % 50 == 0:
        print(f"  Day {di}/{len(dates)}...", flush=True)
    dc = caches[date]
    arr = precompute_arrays(dc)
    caps = {
        "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
    }
    if date.weekday() == 0:
        caps = {k: v * 2 for k, v in caps.items()}
    trades = simulate_day_v2(
        dc, arr,
        per_level_ts={
            "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
            "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
            "IBH": (6, 20),
        },
        per_level_caps=caps,
        exclude_levels={"FIB_0.5", "IBL"},
        direction_filter={"IBH": "down"},
        daily_loss=200.0,
        momentum_max=5.0,
    )
    all_trades.extend([(date, t) for t in trades])

n_days = len(dates)
total_pnl = sum(t.pnl_usd for _, t in all_trades)
wins = sum(1 for _, t in all_trades if t.pnl_usd >= 0)
losses = len(all_trades) - wins

# Max drawdown
cum = 0.0
peak = 0.0
max_dd = 0.0
for _, t in all_trades:
    cum += t.pnl_usd
    if cum > peak:
        peak = cum
    dd = peak - cum
    if dd > max_dd:
        max_dd = dd

# Day-level stats
day_pnl = defaultdict(float)
for d, t in all_trades:
    day_pnl[d] += t.pnl_usd
bad_days = sum(1 for p in day_pnl.values() if p <= -200)
winning_days = sum(1 for p in day_pnl.values() if p > 0)

# Quarterly
sorted_dates = sorted(day_pnl.keys())
q_size = len(sorted_dates) // 4
for i, label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
    start = i * q_size
    end = (i + 1) * q_size if i < 3 else len(sorted_dates)
    q_dates = sorted_dates[start:end]
    q_pnl = sum(day_pnl[d] for d in q_dates)
    print(f"  {label}: ${q_pnl / len(q_dates):+.1f}/day")

# Recent
r60_dates = set(sorted_dates[-60:])
r30_dates = set(sorted_dates[-30:])
r60_pnl = sum(t.pnl_usd for d, t in all_trades if d in r60_dates)
r30_pnl = sum(t.pnl_usd for d, t in all_trades if d in r30_dates)

print()
print("=== DEPLOYED CONFIG (v2 accurate baseline) ===")
print(f"Days: {n_days}")
print(f"Trades: {len(all_trades)} ({wins}W/{losses}L)")
print(f"WR: {wins / len(all_trades) * 100:.1f}%")
print(f"P&L/day: ${total_pnl / n_days:+.2f}")
print(f"MaxDD: ${max_dd:.0f}")
print(f"Bad days (<=\u0024-200): {bad_days}")
print(f"Winning days: {winning_days}/{n_days} ({winning_days / n_days * 100:.1f}%)")
print(f"R60d: ${r60_pnl / 60:+.2f}")
print(f"R30d: ${r30_pnl / 30:+.2f}")
