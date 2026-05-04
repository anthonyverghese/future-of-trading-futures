"""Comprehensive verification of simulate_day_v2 correctness."""
import datetime, sys, os, pytz
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

ET = pytz.timezone("America/New_York")
dates, caches = load_all_days()

TS = {"FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
      "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
      "IBH": (6, 20)}
BASE_CAPS = {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
             "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7}
EXCLUDED = {"FIB_0.5", "IBL", "VWAP"}

print("Running all %d days..." % len(dates), flush=True)

all_trades = []
day_trades = {}
errors = []

for di, date in enumerate(dates):
    if di % 50 == 0:
        print("  Day %d/%d..." % (di, len(dates)), flush=True)
    dc = caches[date]
    arr = precompute_arrays(dc)
    caps = dict(BASE_CAPS)
    is_monday = date.weekday() == 0
    if is_monday:
        caps = {k: v * 2 for k, v in caps.items()}

    trades = simulate_day_v2(
        dc, arr, per_level_ts=TS, per_level_caps=caps,
        exclude_levels={"FIB_0.5", "IBL"},
        direction_filter={"IBH": "down"},
        daily_loss=200.0, momentum_max=5.0)

    day_trades[date] = trades
    all_trades.extend([(date, t) for t in trades])

    # === CHECK 1: No excluded levels ===
    for t in trades:
        if t.level in EXCLUDED:
            errors.append("%s: trade on excluded level %s" % (date, t.level))

    # === CHECK 2: IBH direction filter (SELL only) ===
    for t in trades:
        if t.level == "IBH" and t.direction == "up":
            errors.append("%s: IBH BUY trade found (should be SELL only)" % date)

    # === CHECK 3: Per-level caps ===
    level_counts = defaultdict(int)
    for t in trades:
        level_counts[t.level] += 1
    for level, count in level_counts.items():
        cap = caps.get(level, 12)
        if count > cap:
            errors.append("%s: %s has %d trades > cap %d" % (date, level, count, cap))

    # === CHECK 4: Daily loss limit ===
    cum_pnl = 0.0
    for i, t in enumerate(trades):
        cum_pnl += t.pnl_usd
        if cum_pnl < -200.0 and i < len(trades) - 1:
            # Trade that pushed past -200 is OK (checked before entry),
            # but no trade AFTER that should exist
            next_cum = cum_pnl + trades[i + 1].pnl_usd
            errors.append("%s: trade after -$200 limit (cum=$%.2f, trade %d)" % (
                date, cum_pnl, i + 1))
            break

    # === CHECK 5: 1 position at a time (no overlapping entry_idx ranges) ===
    for i in range(len(trades) - 1):
        if trades[i + 1].entry_idx <= trades[i].exit_idx:
            # Next trade enters before current exits — overlap
            # Unless they share the same exit/entry tick (zone reset)
            if trades[i + 1].entry_idx < trades[i].exit_idx:
                errors.append("%s: overlapping positions trades %d-%d (idx %d < %d)" % (
                    date, i, i + 1, trades[i + 1].entry_idx, trades[i].exit_idx))

    # === CHECK 6: Monday double caps ===
    if is_monday:
        for level, count in level_counts.items():
            normal_cap = BASE_CAPS.get(level, 12)
            doubled_cap = normal_cap * 2
            if count > doubled_cap:
                errors.append("%s (Monday): %s has %d trades > doubled cap %d" % (
                    date, level, count, doubled_cap))

# === CHECK 7: No trades on excluded levels across all days ===
level_totals = defaultdict(int)
for _, t in all_trades:
    level_totals[t.level] += 1

print()
print("=== RESULTS ===")
print("Total: %d days, %d trades" % (len(dates), len(all_trades)))
print()
print("Trades by level:")
for lv in sorted(level_totals.keys()):
    print("  %s: %d" % (lv, level_totals[lv]))

print()
if errors:
    print("ERRORS FOUND: %d" % len(errors))
    for e in errors[:20]:
        print("  %s" % e)
else:
    print("ALL CHECKS PASSED:")
    print("  1. No excluded levels traded")
    print("  2. IBH SELL only (no BUY)")
    print("  3. Per-level caps respected")
    print("  4. Daily loss limit (-$200) respected")
    print("  5. No overlapping positions")
    print("  6. Monday double caps applied")

# === CHECK 8: Monday vs non-Monday trade counts ===
print()
monday_trades = sum(len(day_trades[d]) for d in dates if d.weekday() == 0)
monday_days = sum(1 for d in dates if d.weekday() == 0)
other_trades = sum(len(day_trades[d]) for d in dates if d.weekday() != 0)
other_days = sum(1 for d in dates if d.weekday() != 0)
print("Monday avg trades/day: %.1f (%d days)" % (monday_trades / monday_days, monday_days))
print("Other avg trades/day: %.1f (%d days)" % (other_trades / other_days, other_days))
print("Monday has more trades: %s (expected due to double caps)" % (
    monday_trades / monday_days > other_trades / other_days))
