"""Regenerate golden results + run full baseline with/without momentum."""
import json, datetime, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

dates, caches = load_all_days()
print("Loaded %d days" % len(dates), flush=True)

TS = {"FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
      "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
      "IBH": (6, 20)}
BASE_CAPS = {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
             "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7}

def run_config(momentum_max):
    all_trades = []
    for di, date in enumerate(dates):
        if di % 50 == 0:
            print("  Day %d/%d..." % (di, len(dates)), flush=True)
        dc = caches[date]
        arr = precompute_arrays(dc)
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        trades = simulate_day_v2(
            dc, arr, per_level_ts=TS, per_level_caps=caps,
            exclude_levels={"FIB_0.5", "IBL"},
            direction_filter={"IBH": "down"},
            daily_loss=200.0, momentum_max=momentum_max)
        all_trades.extend([(date, t) for t in trades])
    return all_trades

def print_stats(label, all_trades):
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
    win_days = sum(1 for p in day_pnl.values() if p > 0)
    sd = sorted(day_pnl.keys())
    r60 = sum(t.pnl_usd for d, t in all_trades if d in set(sd[-60:])) / 60
    r30 = sum(t.pnl_usd for d, t in all_trades if d in set(sd[-30:])) / 30
    qs = len(sd) // 4
    print("\n=== %s ===" % label)
    print("Trades: %d (%dW/%dL)" % (len(all_trades), wins, len(all_trades)-wins))
    print("WR: %.1f%%" % (wins/len(all_trades)*100))
    print("P&L/day: $%+.2f" % (pnl/n))
    print("MaxDD: $%.0f" % max_dd)
    print("Bad days: %d" % bad)
    print("Win days: %d/%d (%.1f%%)" % (win_days, n, win_days/n*100))
    print("R60d: $%+.2f  R30d: $%+.2f" % (r60, r30))
    for i, ql in enumerate(["Q1","Q2","Q3","Q4"]):
        s = i*qs; e = (i+1)*qs if i < 3 else len(sd)
        qp = sum(day_pnl[d] for d in sd[s:e]) / len(sd[s:e])
        print("  %s: $%+.1f/day" % (ql, qp))

# Step 1: Regenerate golden results
print("\n=== REGENERATING GOLDEN RESULTS ===", flush=True)
golden_days = [datetime.date(2026, 1, 6), datetime.date(2026, 2, 12),
               datetime.date(2026, 4, 28), datetime.date(2026, 5, 1)]
golden = {}
for day in golden_days:
    if day not in caches:
        continue
    dc = caches[day]
    arr = precompute_arrays(dc)
    caps = dict(BASE_CAPS)
    if day.weekday() == 0:
        caps = {k: v * 2 for k, v in caps.items()}
    trades = simulate_day_v2(dc, arr, per_level_ts=TS, per_level_caps=caps,
        exclude_levels={"FIB_0.5", "IBL"}, direction_filter={"IBH": "down"},
        daily_loss=200.0, momentum_max=5.0)
    pnl = round(sum(t.pnl_usd for t in trades), 2)
    wins = sum(1 for t in trades if t.pnl_usd >= 0)
    golden[str(day)] = {
        "trades": len(trades), "wins": wins, "losses": len(trades)-wins,
        "pnl": pnl,
        "trade_details": [{"level": t.level, "direction": t.direction,
            "entry_count": t.entry_count, "outcome": t.outcome,
            "pnl_usd": round(t.pnl_usd, 2), "entry_idx": t.entry_idx,
            "exit_idx": t.exit_idx} for t in trades],
    }
    print("  %s: %dt (%dW/%dL) $%.2f" % (day, len(trades), wins, len(trades)-wins, pnl))

with open("mnq_alerts/tests/golden_sim_results.json", "w") as f:
    json.dump(golden, f, indent=2)
print("Golden results saved.", flush=True)

# Step 2: Full baseline WITH momentum
print("\n=== FULL BASELINE WITH MOMENTUM (5.0) ===", flush=True)
trades_mom = run_config(5.0)
print_stats("WITH MOMENTUM (5.0)", trades_mom)

# Step 3: Full baseline WITHOUT momentum
print("\n=== FULL BASELINE WITHOUT MOMENTUM (0.0) ===", flush=True)
trades_no = run_config(0.0)
print_stats("WITHOUT MOMENTUM (0.0)", trades_no)

pnl_mom = sum(t.pnl_usd for _, t in trades_mom) / len(dates)
pnl_no = sum(t.pnl_usd for _, t in trades_no) / len(dates)
print("\n=== MOMENTUM IMPACT ===")
print("With:    %d trades, $%+.2f/day" % (len(trades_mom), pnl_mom))
print("Without: %d trades, $%+.2f/day" % (len(trades_no), pnl_no))
print("Diff:    $%+.2f/day (%s)" % (pnl_mom - pnl_no,
    "momentum HELPS" if pnl_mom > pnl_no else "momentum HURTS"))
