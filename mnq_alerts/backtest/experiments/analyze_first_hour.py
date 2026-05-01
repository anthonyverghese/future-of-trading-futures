"""Deep analysis of first hour (10:30-11:30 AM ET) patterns on bad vs good days."""
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

FIRST_HOUR_START = 630   # 10:30 AM ET
FIRST_HOUR_END = 690     # 11:30 AM ET

print("Loading data and simulating...")
dates, caches = load_all_days()
arrays_map = {d: precompute_arrays(caches[d]) for d in dates}

by_date = defaultdict(list)
streak = (0, 0)
for date in dates:
    caps = dict(CAPS)
    if date.weekday() == 0:
        caps = {k: v * 2 for k, v in caps.items()}
    trades, streak = simulate_day(
        caches[date], arrays_map[date],
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
neutral_days = [(d, dt, pnl) for d, dt, pnl in all_days if -100 < pnl < 50]

print(f"\nTotal: {len(dates)} days | Bad (<=-$100): {len(bad_days)} | Good (>=$50): {len(good_days)} | Neutral: {len(neutral_days)}")
print(f"Total trades: {sum(len(dt) for _,dt,_ in all_days)}")

def first_hour_trades(trades):
    return [t for t in trades if FIRST_HOUR_START <= t.factors.et_mins < FIRST_HOUR_END]

def first_15_trades(trades):
    return [t for t in trades if FIRST_HOUR_START <= t.factors.et_mins < FIRST_HOUR_START + 15]

def first_30_trades(trades):
    return [t for t in trades if FIRST_HOUR_START <= t.factors.et_mins < FIRST_HOUR_START + 30]


# =========================================================================
# 1. FIRST-HOUR TRADE OUTCOMES BY LEVEL
# =========================================================================
print("\n" + "="*80)
print("1. FIRST-HOUR TRADE OUTCOMES BY LEVEL (et_mins 630-690)")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    print(f"\n  --- {label} ({len(days)} days) ---")
    level_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
    for _, dt, _ in days:
        for t in first_hour_trades(dt):
            s = level_stats[t.level]
            s["count"] += 1
            s["pnl"] += t.pnl_usd
            if t.pnl_usd >= 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

    total_fh = sum(s["count"] for s in level_stats.values())
    print(f"  Total first-hour trades: {total_fh}")
    print(f"  {'Level':<22} {'Trades':>6} {'WR':>6} {'AvgPnL':>8} {'TotalPnL':>10}")
    for lv in sorted(level_stats.keys()):
        s = level_stats[lv]
        wr = s["wins"] / s["count"] * 100 if s["count"] else 0
        avg = s["pnl"] / s["count"] if s["count"] else 0
        print(f"  {lv:<22} {s['count']:>6} {wr:>5.0f}% {avg:>+7.1f} {s['pnl']:>+10.1f}")


# =========================================================================
# 2. FIRST-HOUR PRICE ACTION RELATIVE TO IB RANGE
# =========================================================================
print("\n" + "="*80)
print("2. FIRST-HOUR PRICE ACTION RELATIVE TO IB RANGE")
print("="*80)

import pytz
_ET = pytz.timezone("America/New_York")

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    print(f"\n  --- {label} ({len(days)} days) ---")

    max_dist_from_ib_15m = []
    max_dist_from_ib_60m = []
    ib_breakout_60m = 0
    ib_range_traversed_pct = []
    price_at_ib_set = []

    for date, dt, pnl in days:
        dc = caches[date]
        ib_range = dc.ibh - dc.ibl
        if ib_range <= 0:
            continue

        fp = dc.full_prices
        ft = dc.full_ts_ns

        # Get ET minutes for all ticks
        arr = arrays_map[date]
        em = arr.et_mins

        # First 15 min after IB (630-645)
        mask_15 = (em >= 630) & (em < 645)
        if mask_15.any():
            prices_15 = fp[mask_15]
            dist_ibh_15 = np.max(np.abs(prices_15 - dc.ibh))
            dist_ibl_15 = np.max(np.abs(prices_15 - dc.ibl))
            max_dist_from_ib_15m.append(min(dist_ibh_15, dist_ibl_15))

        # First 60 min after IB (630-690)
        mask_60 = (em >= 630) & (em < 690)
        if mask_60.any():
            prices_60 = fp[mask_60]
            hi_60 = np.max(prices_60)
            lo_60 = np.min(prices_60)

            # Distance from nearest IB level
            dist_ibh = np.max(np.abs(prices_60 - dc.ibh))
            dist_ibl = np.max(np.abs(prices_60 - dc.ibl))
            max_dist_from_ib_60m.append(min(dist_ibh, dist_ibl))

            # IB breakout?
            if hi_60 > dc.ibh or lo_60 < dc.ibl:
                ib_breakout_60m += 1

            # % of IB range traversed
            range_60 = hi_60 - lo_60
            ib_range_traversed_pct.append(range_60 / ib_range * 100)

        # Price position when IB sets (at minute 630)
        mask_ib_set = em == 630
        if mask_ib_set.any():
            p_at_set = fp[mask_ib_set][0]
            pos_in_range = (p_at_set - dc.ibl) / ib_range * 100
            price_at_ib_set.append(pos_in_range)

    n = len(days)
    print(f"  Max dist from nearest IB level (first 15m): avg {np.mean(max_dist_from_ib_15m):.1f} pts" if max_dist_from_ib_15m else "  No data")
    print(f"  Max dist from nearest IB level (first 60m): avg {np.mean(max_dist_from_ib_60m):.1f} pts" if max_dist_from_ib_60m else "  No data")
    print(f"  IB breakout in first hour: {ib_breakout_60m}/{n} ({ib_breakout_60m/n*100:.0f}%)")
    if ib_range_traversed_pct:
        print(f"  IB range traversed in first hour: avg {np.mean(ib_range_traversed_pct):.0f}%, median {np.median(ib_range_traversed_pct):.0f}%")
    if price_at_ib_set:
        print(f"  Price position in IB range at 10:30: avg {np.mean(price_at_ib_set):.0f}% (0=IBL, 100=IBH)")
        near_top = sum(1 for p in price_at_ib_set if p > 75)
        near_bot = sum(1 for p in price_at_ib_set if p < 25)
        mid = sum(1 for p in price_at_ib_set if 25 <= p <= 75)
        print(f"    Near IBH (>75%): {near_top}/{len(price_at_ib_set)} ({near_top/len(price_at_ib_set)*100:.0f}%)")
        print(f"    Middle (25-75%): {mid}/{len(price_at_ib_set)} ({mid/len(price_at_ib_set)*100:.0f}%)")
        print(f"    Near IBL (<25%): {near_bot}/{len(price_at_ib_set)} ({near_bot/len(price_at_ib_set)*100:.0f}%)")


# =========================================================================
# 3. FIRST-HOUR TRADE SPEED
# =========================================================================
print("\n" + "="*80)
print("3. FIRST-HOUR TRADE SPEED")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    trades_15 = [len(first_15_trades(dt)) for _, dt, _ in days]
    trades_30 = [len(first_30_trades(dt)) for _, dt, _ in days]
    trades_60 = [len(first_hour_trades(dt)) for _, dt, _ in days]

    print(f"\n  --- {label} ({len(days)} days) ---")
    print(f"  First 15 min (630-645): avg {np.mean(trades_15):.1f}, median {np.median(trades_15):.0f}, max {np.max(trades_15)}")
    print(f"  First 30 min (630-660): avg {np.mean(trades_30):.1f}, median {np.median(trades_30):.0f}, max {np.max(trades_30)}")
    print(f"  First 60 min (630-690): avg {np.mean(trades_60):.1f}, median {np.median(trades_60):.0f}, max {np.max(trades_60)}")

    # Distribution of first-60-min trade counts
    print(f"  Distribution of first-hour trade counts:")
    for bucket_lo, bucket_hi in [(0,0),(1,2),(3,4),(5,7),(8,99)]:
        n = sum(1 for t in trades_60 if bucket_lo <= t <= bucket_hi)
        print(f"    {bucket_lo}-{bucket_hi}: {n} days ({n/len(days)*100:.0f}%)")


# =========================================================================
# 4. FIRST LOSS TIMING
# =========================================================================
print("\n" + "="*80)
print("4. FIRST LOSS TIMING (minutes after IB set)")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    first_loss_mins = []
    no_loss_in_fh = 0
    for _, dt, _ in days:
        found = False
        for t in dt:
            if t.pnl_usd < 0:
                first_loss_mins.append(t.factors.et_mins - 630)
                found = True
                break
        if not found:
            no_loss_in_fh += 1

    print(f"\n  --- {label} ({len(days)} days) ---")
    if first_loss_mins:
        print(f"  Avg first loss at +{np.mean(first_loss_mins):.0f} min after IB")
        print(f"  Median first loss at +{np.median(first_loss_mins):.0f} min after IB")
        early = sum(1 for m in first_loss_mins if m < 15)
        mid = sum(1 for m in first_loss_mins if 15 <= m < 30)
        late = sum(1 for m in first_loss_mins if 30 <= m < 60)
        after = sum(1 for m in first_loss_mins if m >= 60)
        total = len(first_loss_mins)
        print(f"  First loss in 0-15 min: {early}/{total} ({early/total*100:.0f}%)")
        print(f"  First loss in 15-30 min: {mid}/{total} ({mid/total*100:.0f}%)")
        print(f"  First loss in 30-60 min: {late}/{total} ({late/total*100:.0f}%)")
        print(f"  First loss after 60 min: {after}/{total} ({after/total*100:.0f}%)")
    print(f"  No losses at all: {no_loss_in_fh} days")


# =========================================================================
# 5. CUMULATIVE P&L TRAJECTORY IN FIRST HOUR
# =========================================================================
print("\n" + "="*80)
print("5. CUMULATIVE P&L TRAJECTORY IN FIRST HOUR (bad days)")
print("="*80)

# Categorize bad days by first-hour trajectory
went_positive_first = 0
started_losing = 0
peaked_amounts = []
trough_after_peak = []

for _, dt, daily_pnl in bad_days:
    fh = first_hour_trades(dt)
    if not fh:
        continue
    cum = 0
    peak = 0
    trough = 0
    ever_positive = False
    trajectory = []
    for t in fh:
        cum += t.pnl_usd
        trajectory.append(cum)
        if cum > peak:
            peak = cum
        if cum < trough:
            trough = cum
        if cum > 0:
            ever_positive = True

    if ever_positive:
        went_positive_first += 1
    else:
        started_losing += 1
    peaked_amounts.append(peak)
    trough_after_peak.append(trough)

n_with_fh = went_positive_first + started_losing
print(f"  Bad days with first-hour trades: {n_with_fh}")
print(f"  Went positive first then crashed: {went_positive_first} ({went_positive_first/max(n_with_fh,1)*100:.0f}%)")
print(f"  Started losing immediately: {started_losing} ({started_losing/max(n_with_fh,1)*100:.0f}%)")
if peaked_amounts:
    print(f"  Avg peak in first hour: ${np.mean(peaked_amounts):.1f}")
    print(f"  Avg trough in first hour: ${np.mean(trough_after_peak):.1f}")

# Also show first-hour P&L on bad days
fh_pnls_bad = []
for _, dt, _ in bad_days:
    fh = first_hour_trades(dt)
    fh_pnls_bad.append(sum(t.pnl_usd for t in fh))
print(f"\n  First-hour P&L on bad days: avg ${np.mean(fh_pnls_bad):.1f}, median ${np.median(fh_pnls_bad):.1f}")
fh_negative = sum(1 for p in fh_pnls_bad if p < 0)
print(f"  First-hour P&L negative: {fh_negative}/{len(bad_days)} ({fh_negative/len(bad_days)*100:.0f}%)")

# Contrast with good days
fh_pnls_good = []
for _, dt, _ in good_days:
    fh = first_hour_trades(dt)
    fh_pnls_good.append(sum(t.pnl_usd for t in fh))
print(f"  First-hour P&L on good days: avg ${np.mean(fh_pnls_good):.1f}, median ${np.median(fh_pnls_good):.1f}")

# Show the contribution of first hour to total daily P&L on bad days
print(f"\n  First-hour % of total daily loss (bad days):")
for _, dt, daily_pnl in bad_days[:5]:  # sample
    fh = first_hour_trades(dt)
    fh_pnl = sum(t.pnl_usd for t in fh)
    pct = fh_pnl / daily_pnl * 100 if daily_pnl != 0 else 0
    print(f"    Daily: ${daily_pnl:.0f}, First hour: ${fh_pnl:.0f} ({pct:.0f}%)")
avg_fh_contrib = np.mean([sum(t.pnl_usd for t in first_hour_trades(dt)) / pnl * 100
                          for _, dt, pnl in bad_days if pnl != 0])
print(f"  Avg first-hour contribution to daily loss: {avg_fh_contrib:.0f}%")


# =========================================================================
# 6. LEVEL-SPECIFIC PATTERNS: WHICH LEVEL FIRES FIRST
# =========================================================================
print("\n" + "="*80)
print("6. WHICH LEVEL FIRES FIRST")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    first_level = Counter()
    first_level_outcome = defaultdict(lambda: {"wins": 0, "losses": 0})
    for _, dt, _ in days:
        if dt:
            lv = dt[0].level
            first_level[lv] += 1
            if dt[0].pnl_usd >= 0:
                first_level_outcome[lv]["wins"] += 1
            else:
                first_level_outcome[lv]["losses"] += 1

    print(f"\n  --- {label} ({len(days)} days) ---")
    print(f"  {'Level':<22} {'Count':>6} {'%':>6} {'WR':>6}")
    for lv, c in first_level.most_common():
        o = first_level_outcome[lv]
        wr = o["wins"] / (o["wins"] + o["losses"]) * 100 if (o["wins"] + o["losses"]) else 0
        print(f"  {lv:<22} {c:>6} {c/len(days)*100:>5.0f}% {wr:>5.0f}%")


# =========================================================================
# 7. PRICE POSITION RELATIVE TO VWAP AT IB SET
# =========================================================================
print("\n" + "="*80)
print("7. PRICE POSITION RELATIVE TO VWAP AT IB SET")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    vwap_diffs = []
    above_vwap = 0

    for date, dt, pnl in days:
        dc = caches[date]
        arr = arrays_map[date]
        em = arr.et_mins

        # Find price and VWAP right at IB set (minute 630)
        mask = em == 630
        if mask.any():
            idx = np.where(mask)[0][0]
            price_at_ib = dc.full_prices[idx]
            vwap_at_ib = dc.post_ib_vwaps[0] if len(dc.post_ib_vwaps) > 0 else None
            if vwap_at_ib is not None:
                diff = price_at_ib - vwap_at_ib
                vwap_diffs.append(diff)
                if diff > 0:
                    above_vwap += 1

    n = len(vwap_diffs)
    print(f"\n  --- {label} ({n} days with data) ---")
    if vwap_diffs:
        print(f"  Avg distance from VWAP at IB set: {np.mean(vwap_diffs):+.1f} pts")
        print(f"  Median distance: {np.median(vwap_diffs):+.1f} pts")
        print(f"  Above VWAP: {above_vwap}/{n} ({above_vwap/n*100:.0f}%)")
        print(f"  Below VWAP: {n-above_vwap}/{n} ({(n-above_vwap)/n*100:.0f}%)")
        print(f"  Abs distance > 20 pts: {sum(1 for d in vwap_diffs if abs(d) > 20)}/{n} ({sum(1 for d in vwap_diffs if abs(d) > 20)/n*100:.0f}%)")
        print(f"  Abs distance > 50 pts: {sum(1 for d in vwap_diffs if abs(d) > 50)}/{n} ({sum(1 for d in vwap_diffs if abs(d) > 50)/n*100:.0f}%)")


# =========================================================================
# 8. IB RANGE CHARACTERISTICS
# =========================================================================
print("\n" + "="*80)
print("8. IB RANGE CHARACTERISTICS")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    ib_ranges = []
    price_positions = []  # 0-100% where price is in IB range at lock

    for date, dt, pnl in days:
        dc = caches[date]
        ib_range = dc.ibh - dc.ibl
        ib_ranges.append(ib_range)

        arr = arrays_map[date]
        em = arr.et_mins
        mask = em == 630
        if mask.any() and ib_range > 0:
            p = dc.full_prices[np.where(mask)[0][0]]
            pos = (p - dc.ibl) / ib_range * 100
            price_positions.append(pos)

    print(f"\n  --- {label} ({len(days)} days) ---")
    print(f"  IB Range: avg {np.mean(ib_ranges):.0f} pts, median {np.median(ib_ranges):.0f} pts")

    # Bucket IB ranges
    for lo, hi in [(0,50),(50,100),(100,150),(150,200),(200,999)]:
        n = sum(1 for r in ib_ranges if lo <= r < hi)
        pct = n / len(ib_ranges) * 100
        print(f"    {lo}-{hi} pts: {n} days ({pct:.0f}%)")

    if price_positions:
        print(f"  Price position at IB lock: avg {np.mean(price_positions):.0f}%")
        # Near top = likely to hit IBH first, near bottom = IBL first
        top_third = sum(1 for p in price_positions if p > 67)
        mid_third = sum(1 for p in price_positions if 33 <= p <= 67)
        bot_third = sum(1 for p in price_positions if p < 33)
        n = len(price_positions)
        print(f"    Top third (>67%): {top_third}/{n} ({top_third/n*100:.0f}%)")
        print(f"    Middle third (33-67%): {mid_third}/{n} ({mid_third/n*100:.0f}%)")
        print(f"    Bottom third (<33%): {bot_third}/{n} ({bot_third/n*100:.0f}%)")


# =========================================================================
# 9. FIRST-HOUR WIN STREAKS
# =========================================================================
print("\n" + "="*80)
print("9. FIRST-HOUR WIN STREAKS BEFORE FIRST LOSS")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    initial_wins = []
    for _, dt, _ in days:
        fh = first_hour_trades(dt)
        wins_before_loss = 0
        for t in fh:
            if t.pnl_usd >= 0:
                wins_before_loss += 1
            else:
                break
        initial_wins.append(wins_before_loss)

    print(f"\n  --- {label} ({len(days)} days) ---")
    print(f"  Avg consecutive wins before first loss in first hour: {np.mean(initial_wins):.1f}")
    print(f"  Median: {np.median(initial_wins):.0f}")
    for n_wins in range(6):
        c = sum(1 for w in initial_wins if w == n_wins)
        print(f"    Exactly {n_wins} wins first: {c}/{len(days)} ({c/len(days)*100:.0f}%)")
    c = sum(1 for w in initial_wins if w >= 6)
    print(f"    6+ wins first: {c}/{len(days)} ({c/len(days)*100:.0f}%)")


# =========================================================================
# 10. DIRECTION OF FIRST-HOUR TRADES
# =========================================================================
print("\n" + "="*80)
print("10. DIRECTION OF FIRST-HOUR TRADES")
print("="*80)

for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days)]:
    buy_wins = 0; buy_losses = 0; sell_wins = 0; sell_losses = 0
    buy_pnl = 0.0; sell_pnl = 0.0

    for _, dt, _ in days:
        for t in first_hour_trades(dt):
            if t.direction == "up":
                if t.pnl_usd >= 0: buy_wins += 1
                else: buy_losses += 1
                buy_pnl += t.pnl_usd
            else:
                if t.pnl_usd >= 0: sell_wins += 1
                else: sell_losses += 1
                sell_pnl += t.pnl_usd

    total_buys = buy_wins + buy_losses
    total_sells = sell_wins + sell_losses
    print(f"\n  --- {label} ({len(days)} days) ---")
    if total_buys:
        print(f"  BUY:  {total_buys} trades, WR {buy_wins/total_buys*100:.0f}%, total ${buy_pnl:+.0f}, avg ${buy_pnl/total_buys:+.1f}")
    else:
        print(f"  BUY:  0 trades")
    if total_sells:
        print(f"  SELL: {total_sells} trades, WR {sell_wins/total_sells*100:.0f}%, total ${sell_pnl:+.0f}, avg ${sell_pnl/total_sells:+.1f}")
    else:
        print(f"  SELL: 0 trades")

    # Ratio
    total = total_buys + total_sells
    if total:
        print(f"  Buy/Sell ratio: {total_buys/total*100:.0f}% / {total_sells/total*100:.0f}%")

    # Per-day direction dominance
    days_buy_dominant = 0
    days_sell_dominant = 0
    days_balanced = 0
    for _, dt, _ in days:
        fh = first_hour_trades(dt)
        b = sum(1 for t in fh if t.direction == "up")
        s = sum(1 for t in fh if t.direction == "down")
        if b > s: days_buy_dominant += 1
        elif s > b: days_sell_dominant += 1
        else: days_balanced += 1
    print(f"  Days buy-dominant: {days_buy_dominant}, sell-dominant: {days_sell_dominant}, balanced: {days_balanced}")


# =========================================================================
# BONUS: ACTIONABLE SUMMARY
# =========================================================================
print("\n" + "="*80)
print("BONUS: FIRST-HOUR KILL SWITCH ANALYSIS")
print("="*80)

# What if we stopped trading after first loss in first hour?
print("\n  --- Impact of stopping after first loss in first hour ---")
for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days), ("ALL DAYS", all_days)]:
    saved_pnl = 0
    lost_pnl = 0
    affected_days = 0

    for _, dt, _ in days:
        first_loss_in_fh = None
        for i, t in enumerate(dt):
            if t.factors.et_mins >= FIRST_HOUR_END:
                break
            if t.pnl_usd < 0:
                first_loss_in_fh = i
                break

        if first_loss_in_fh is not None:
            # All trades after the first loss
            remaining = dt[first_loss_in_fh + 1:]
            remaining_pnl = sum(t.pnl_usd for t in remaining)
            if remaining_pnl < 0:
                saved_pnl += abs(remaining_pnl)
            else:
                lost_pnl += remaining_pnl
            affected_days += 1

    print(f"  {label}: {affected_days} days affected, saved ${saved_pnl:.0f} in losses, lost ${lost_pnl:.0f} in wins")

# What if we delayed trading until 11:00 AM (minute 660)?
print("\n  --- Impact of delaying start to 11:00 AM (et_mins 660) ---")
for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days), ("ALL DAYS", all_days)]:
    removed_pnl = 0
    removed_trades = 0
    for _, dt, _ in days:
        early = [t for t in dt if t.factors.et_mins < 660]
        removed_pnl += sum(t.pnl_usd for t in early)
        removed_trades += len(early)
    print(f"  {label}: would remove {removed_trades} trades worth ${removed_pnl:+.0f}")

# What if we delayed to 11:30 AM (minute 690)?
print("\n  --- Impact of skipping entire first hour (start at 11:30 AM / et_mins 690) ---")
for label, days in [("BAD DAYS", bad_days), ("GOOD DAYS", good_days), ("ALL DAYS", all_days)]:
    removed_pnl = 0
    removed_trades = 0
    for _, dt, _ in days:
        early = first_hour_trades(dt)
        removed_pnl += sum(t.pnl_usd for t in early)
        removed_trades += len(early)
    remaining_pnl = sum(pnl for _,_,pnl in days) - removed_pnl
    print(f"  {label}: would remove {removed_trades} trades worth ${removed_pnl:+.0f}, remaining ${remaining_pnl:+.0f}")

print("\nDone.")
