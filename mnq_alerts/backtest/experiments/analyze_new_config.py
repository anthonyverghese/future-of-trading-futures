"""Analyze bad vs good days on the new deployed config (IBH SELL + 0.236 cap=18)."""
import os, sys
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
    "IBH": (6, 20),
}
CAPS = {
    "FIB_0.236": 18, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6, "IBH": 7,
}

print("Loading and simulating new config...")
dates, caches = load_all_days()
arrays = {d: precompute_arrays(caches[d]) for d in dates}

by_date = defaultdict(list)
streak = (0, 0)
for date in dates:
    caps = dict(CAPS)
    if date.weekday() == 0:
        caps = {k: v * 2 for k, v in caps.items()}
    trades, streak = simulate_day(
        caches[date], arrays[date],
        zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
        target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
        stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
        max_per_level_map=caps,
        exclude_levels={"FIB_0.5", "IBL"},
        include_ibl=False, include_vwap=False,
        global_cooldown_after_loss_secs=30,
        direction_filter={"IBH": "down"},
    )
    by_date[date] = trades

all_days = [(d, by_date[d], sum(t.pnl_usd for t in by_date[d])) for d in dates]
bad_days = [(d, dt, pnl) for d, dt, pnl in all_days if pnl <= -100]
good_days = [(d, dt, pnl) for d, dt, pnl in all_days if pnl >= 50]

print(f"New config: {sum(len(dt) for _,dt,_ in all_days)} trades, {len(bad_days)} bad days, {len(good_days)} good days")

# 1. Level breakdown on bad days
print(f"\n=== LEVEL BREAKDOWN ON BAD DAYS ({len(bad_days)}) ===")
level_bad = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "days": set()})
for date, dt, pnl in bad_days:
    for t in dt:
        level_bad[t.level]["trades"] += 1
        level_bad[t.level]["pnl"] += t.pnl_usd
        level_bad[t.level]["days"].add(date)

for lv in sorted(level_bad.keys()):
    s = level_bad[lv]
    avg = s["pnl"] / s["trades"] if s["trades"] > 0 else 0
    print(f"  {lv:<20} {len(s['days'])} days, {s['trades']} trades, ${s['pnl']:+.0f} total, ${avg:+.2f}/trade")

# Dominant loser
dominant = Counter()
for date, dt, pnl in bad_days:
    lp = defaultdict(float)
    for t in dt:
        lp[t.level] += t.pnl_usd
    worst = min(lp.items(), key=lambda x: x[1])
    dominant[worst[0]] += 1
print(f"\nDominant loser per bad day:")
for lv, c in dominant.most_common():
    print(f"  {lv:<20} {c} days ({c/len(bad_days)*100:.0f}%)")

# 2. Time of losses
print(f"\n=== TIME OF LOSSES ON BAD DAYS ===")
loss_times = [t.factors.et_mins for _, dt, _ in bad_days for t in dt if t.pnl_usd < 0]
for start, end, label in [(631,690,"10:31-11:30"),(690,750,"11:30-12:30"),(750,810,"12:30-13:30"),(840,900,"14:00-15:00"),(900,960,"15:00-16:00")]:
    n = sum(1 for et in loss_times if start <= et < end)
    print(f"  {label}: {n} losses ({n/len(loss_times)*100:.0f}%)")

# 3. P&L trajectory
print(f"\n=== P&L TRAJECTORY ===")
peaked = 0
peaks = []
for date, dt, pnl in bad_days:
    cum = 0
    peak = 0
    for t in dt:
        cum += t.pnl_usd
        if cum > peak: peak = cum
    peaks.append(peak)
    if peak > 10: peaked += 1
print(f"  Peaked >$10 then crashed: {peaked}/{len(bad_days)} ({peaked/len(bad_days)*100:.0f}%)")
print(f"  Avg peak before crash: ${np.mean(peaks):.1f}")

# 4. First trade outcome
first_w_bad = sum(1 for _,dt,_ in bad_days if dt[0].pnl_usd >= 0)
first_w_good = sum(1 for _,dt,_ in good_days if dt[0].pnl_usd >= 0)
print(f"\n=== FIRST TRADE OUTCOME ===")
print(f"  Bad days: {first_w_bad}/{len(bad_days)} win ({first_w_bad/len(bad_days)*100:.0f}%)")
print(f"  Good days: {first_w_good}/{len(good_days)} win ({first_w_good/len(good_days)*100:.0f}%)")

# 5. Trade speed
print(f"\n=== TRADE SPEED ===")
for label, days in [("Bad", bad_days), ("Good", good_days)]:
    speeds = []
    for _,dt,_ in days:
        if len(dt) < 2: continue
        span = (dt[-1].factors.et_mins - dt[0].factors.et_mins) / 60
        if span > 0: speeds.append(len(dt)/span)
    print(f"  {label}: {np.mean(speeds):.1f} trades/hr (median {np.median(speeds):.1f})")

# 6. After first loss recovery
print(f"\n=== AFTER FIRST LOSS ===")
for label, days in [("Bad", bad_days), ("Good", good_days)]:
    recovered = 0
    total = 0
    pnls = []
    for _,dt,_ in days:
        idx = None
        for i,t in enumerate(dt):
            if t.pnl_usd < 0:
                idx = i
                break
        if idx is not None:
            total += 1
            rem_pnl = sum(t.pnl_usd for t in dt[idx+1:])
            pnls.append(rem_pnl)
            if rem_pnl > 0: recovered += 1
    print(f"  {label}: {recovered}/{total} recover ({recovered/max(total,1)*100:.0f}%), avg ${np.mean(pnls):.1f} after")

# 7. IBH contribution
print(f"\n=== IBH SELL CONTRIBUTION ===")
for label, days in [("Bad", bad_days), ("Good", good_days)]:
    ibh = [t for _,dt,_ in days for t in dt if t.level == "IBH"]
    pnl = sum(t.pnl_usd for t in ibh)
    wr = sum(1 for t in ibh if t.pnl_usd >= 0) / max(len(ibh),1) * 100
    print(f"  {label}: {len(ibh)} trades, {wr:.0f}% WR, ${pnl:+.0f}")

# 8. Consecutive losses
print(f"\n=== MAX CONSECUTIVE LOSSES PER DAY ===")
for label, days in [("Bad", bad_days), ("Good", good_days)]:
    max_consec = []
    for _,dt,_ in days:
        c = 0; mc = 0
        for t in dt:
            if t.pnl_usd < 0: c += 1; mc = max(mc, c)
            else: c = 0
        max_consec.append(mc)
    print(f"  {label}: avg {np.mean(max_consec):.1f}, median {np.median(max_consec):.0f}")

# 9. IB range
print(f"\n=== IB RANGE ===")
for label, days in [("Bad", bad_days), ("Good", good_days)]:
    ranges = [caches[d].ibh - caches[d].ibl for d,_,_ in days]
    print(f"  {label}: avg {np.mean(ranges):.0f}pts, median {np.median(ranges):.0f}pts")

# 10. What makes good days good?
print(f"\n=== GOOD DAY CHARACTERISTICS ===")
good_trades = [len(dt) for _,dt,_ in good_days]
good_levels = Counter(t.level for _,dt,_ in good_days for t in dt)
print(f"  Avg trades: {np.mean(good_trades):.1f}")
print(f"  Level distribution:")
for lv, c in good_levels.most_common():
    print(f"    {lv:<20} {c} trades ({c/sum(good_levels.values())*100:.0f}%)")

# 11. Direction analysis on bad days
print(f"\n=== DIRECTION ON BAD DAYS ===")
for label, days in [("Bad", bad_days), ("Good", good_days)]:
    buys = [t for _,dt,_ in days for t in dt if t.direction == "up"]
    sells = [t for _,dt,_ in days for t in dt if t.direction == "down"]
    buy_wr = sum(1 for t in buys if t.pnl_usd >= 0) / max(len(buys),1) * 100
    sell_wr = sum(1 for t in sells if t.pnl_usd >= 0) / max(len(sells),1) * 100
    print(f"  {label}: BUY {len(buys)} trades {buy_wr:.0f}% WR, SELL {len(sells)} trades {sell_wr:.0f}% WR")
