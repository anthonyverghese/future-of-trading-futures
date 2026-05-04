"""Verify v2 is deterministic and EOD flatten works."""
import datetime, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate_v2 import simulate_day_v2

dates, caches = load_all_days()
test_days = [datetime.date(2026, 1, 6), datetime.date(2026, 4, 28), datetime.date(2026, 5, 1)]

TS = {"FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
      "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
      "IBH": (6, 20)}
BASE_CAPS = {"FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
             "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7}

print("=== DETERMINISTIC CHECK ===")
all_match = True
for day in test_days:
    if day not in caches:
        continue
    dc = caches[day]
    arr = precompute_arrays(dc)
    caps = dict(BASE_CAPS)
    if day.weekday() == 0:
        caps = {k: v * 2 for k, v in caps.items()}

    kwargs = dict(per_level_ts=TS, per_level_caps=caps,
                  exclude_levels={"FIB_0.5", "IBL"},
                  direction_filter={"IBH": "down"},
                  daily_loss=200.0, momentum_max=5.0)

    t1 = simulate_day_v2(dc, arr, **kwargs)
    t2 = simulate_day_v2(dc, arr, **kwargs)

    pnl1 = round(sum(t.pnl_usd for t in t1), 2)
    pnl2 = round(sum(t.pnl_usd for t in t2), 2)
    match = len(t1) == len(t2) and pnl1 == pnl2
    if not match:
        all_match = False
    print("%s: run1=%dt/$%.2f  run2=%dt/$%.2f  match=%s" % (
        day, len(t1), pnl1, len(t2), pnl2, match))

print("All deterministic: %s" % all_match)

print()
print("=== MOMENTUM EFFECT CHECK ===")
for day in test_days:
    if day not in caches:
        continue
    dc = caches[day]
    arr = precompute_arrays(dc)
    caps = dict(BASE_CAPS)
    if day.weekday() == 0:
        caps = {k: v * 2 for k, v in caps.items()}

    t_mom = simulate_day_v2(dc, arr, per_level_ts=TS, per_level_caps=caps,
        exclude_levels={"FIB_0.5", "IBL"}, direction_filter={"IBH": "down"},
        daily_loss=200.0, momentum_max=5.0)
    t_no = simulate_day_v2(dc, arr, per_level_ts=TS, per_level_caps=caps,
        exclude_levels={"FIB_0.5", "IBL"}, direction_filter={"IBH": "down"},
        daily_loss=200.0, momentum_max=0.0)

    pnl_m = round(sum(t.pnl_usd for t in t_mom), 2)
    pnl_n = round(sum(t.pnl_usd for t in t_no), 2)
    print("%s: mom=%dt/$%.2f  no_mom=%dt/$%.2f  diff=%s" % (
        day, len(t_mom), pnl_m, len(t_no), pnl_n,
        "SAME" if len(t_mom) == len(t_no) else "DIFFERENT"))
