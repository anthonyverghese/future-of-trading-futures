"""Show trade-by-trade comparison with and without momentum on Jan 6."""
import datetime, sys, os, pytz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

ET = pytz.timezone("America/New_York")
dates, caches = load_all_days()
day = datetime.date(2026, 1, 6)
dc = caches[day]
arr = precompute_arrays(dc)
caps = {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
        "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7}

TS = {"FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
      "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
      "IBH": (6, 20)}

t_mom = simulate_day_v2(dc, arr, per_level_ts=TS, per_level_caps=caps,
    exclude_levels={"FIB_0.5", "IBL"}, direction_filter={"IBH": "down"},
    daily_loss=200.0, momentum_max=5.0)

t_no = simulate_day_v2(dc, arr, per_level_ts=TS, per_level_caps=caps,
    exclude_levels={"FIB_0.5", "IBL"}, direction_filter={"IBH": "down"},
    daily_loss=200.0, momentum_max=0.0)

def fmt(t):
    dt = datetime.datetime.fromtimestamp(t.entry_ns / 1e9, tz=pytz.utc).astimezone(ET)
    return "%s %s %s #%d => %s $%+.2f idx=%d->%d" % (
        dt.strftime("%H:%M:%S"), t.level, t.direction, t.entry_count,
        t.outcome, t.pnl_usd, t.entry_idx, t.exit_idx)

print("=== WITH MOMENTUM (5.0): %d trades, $%.2f ===" % (
    len(t_mom), sum(t.pnl_usd for t in t_mom)))
for i, t in enumerate(t_mom):
    print("  %d: %s" % (i, fmt(t)))

print()
print("=== WITHOUT MOMENTUM (0.0): %d trades, $%.2f ===" % (
    len(t_no), sum(t.pnl_usd for t in t_no)))
for i, t in enumerate(t_no):
    print("  %d: %s" % (i, fmt(t)))

print()
print("=== DIFFERENCES ===")
max_len = max(len(t_mom), len(t_no))
for i in range(max_len):
    m = t_mom[i] if i < len(t_mom) else None
    n = t_no[i] if i < len(t_no) else None
    if m and n:
        if m.level != n.level or m.direction != n.direction or m.entry_idx != n.entry_idx:
            print("Trade %d differs:" % i)
            print("  MOM:   %s" % fmt(m))
            print("  NOMOM: %s" % fmt(n))
            # Check the momentum value at the no-momentum trade's entry
            fp = dc.full_prices
            entry_idx_n = n.entry_idx
            lookback = 1000
            prev_idx = max(0, entry_idx_n - lookback)
            price_now = float(fp[entry_idx_n])
            price_5m = float(fp[prev_idx])
            raw_mom = price_now - price_5m
            if n.direction == "down":
                raw_mom = -raw_mom
            print("  NOMOM trade momentum: price=%.2f, 5m_ago=%.2f, mom=%.2f (%s)" % (
                price_now, price_5m, raw_mom,
                "WOULD BE BLOCKED" if raw_mom > 5.0 else "would pass"))
            print()
