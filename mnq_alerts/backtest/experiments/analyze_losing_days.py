"""Analyze patterns on losing vs winning days.

Phase 1: Run backtest once, collect all trades + daily context.
Phase 2: Compare factor distributions between winning and losing days.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/analyze_losing_days.py
"""
import os, sys, datetime
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.results import save_trades_data

PER_LEVEL_TS = {
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
}
PER_LEVEL_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}


def load_vix():
    """Load VIX daily data from Yahoo Finance."""
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start="2024-12-01", end="2026-05-01", progress=False)
        return {d.date(): float(row["Close"].iloc[0]) for d, row in vix.iterrows()}
    except Exception as e:
        print(f"  WARNING: Could not load VIX: {e}")
        return {}


def compute_gaps(dates, caches):
    """Compute opening gap from previous day's close for each day."""
    cache_dir = os.path.join(os.path.dirname(__file__), "../../data_cache")
    gaps = {}
    prev_close = None
    for date in dates:
        dc = caches[date]
        today_open = float(dc.full_prices[0])
        if prev_close is not None and prev_close > 0:
            gap_pts = today_open - prev_close
            gap_pct = gap_pts / prev_close * 100
            gaps[date] = {"gap_pts": gap_pts, "gap_pct": gap_pct}
        today_close = float(dc.full_prices[-1])
        prev_close = today_close
    return gaps


def get_econ_events():
    """Major economic event dates (FOMC, CPI, NFP) for 2025-2026.

    Sources: Federal Reserve, BLS release schedules.
    """
    # FOMC meeting dates (announcement days) 2025
    fomc = [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        # 2026
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    ]
    # CPI release dates 2025 (typically 2nd or 3rd week)
    cpi = [
        "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
        "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
        "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
        "2026-01-14", "2026-02-12", "2026-03-11", "2026-04-14",
    ]
    # NFP (first Friday of each month) 2025
    nfp = [
        "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
        "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
        "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
        "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    ]
    events = {}
    for d in fomc:
        events[datetime.date.fromisoformat(d)] = "FOMC"
    for d in cpi:
        events[datetime.date.fromisoformat(d)] = "CPI"
    for d in nfp:
        events[datetime.date.fromisoformat(d)] = "NFP"
    return events


def analyze_factor(name, winning_vals, losing_vals):
    """Compare a factor between winning and losing days."""
    if not winning_vals or not losing_vals:
        return None
    w = np.array(winning_vals)
    l = np.array(losing_vals)
    return {
        "winning_mean": round(float(np.mean(w)), 2),
        "winning_median": round(float(np.median(w)), 2),
        "losing_mean": round(float(np.mean(l)), 2),
        "losing_median": round(float(np.median(l)), 2),
        "diff_mean": round(float(np.mean(l) - np.mean(w)), 2),
    }


def analyze_bucket(name, values, pnls, bucket_fn, bucket_labels):
    """Bucket a factor and show WR + P&L per bucket."""
    buckets = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0.0, "n_days": 0})
    for val, pnl in zip(values, pnls):
        b = bucket_fn(val)
        if b is None:
            continue
        buckets[b]["pnl"] += pnl
        buckets[b]["n_days"] += 1
        if pnl >= 0:
            buckets[b]["w"] += 1
        else:
            buckets[b]["l"] += 1

    print(f"\n  {name}:")
    print(f"  {'Bucket':<25} {'Days':>5} {'W':>4} {'L':>4} {'WR%':>6} {'$/day':>8}")
    print(f"  {'-'*55}")
    for label in bucket_labels:
        if label not in buckets:
            continue
        b = buckets[label]
        total = b["w"] + b["l"]
        wr = b["w"] / total * 100 if total > 0 else 0
        avg = b["pnl"] / b["n_days"] if b["n_days"] > 0 else 0
        print(f"  {label:<25} {b['n_days']:>5} {b['w']:>4} {b['l']:>4} {wr:>5.1f}% ${avg:>+7.2f}")


def main():
    print("=== Phase 1: Run backtest and collect data ===")
    print("Loading data...")
    dates, caches = load_all_days()
    print(f"Loaded {len(dates)} days")
    print("Precomputing arrays...")
    arrays_cache = {d: precompute_arrays(caches[d]) for d in dates}

    print("Loading VIX...")
    vix_data = load_vix()
    print(f"VIX data: {len(vix_data)} days")

    print("Computing gaps...")
    gaps = compute_gaps(dates, caches)
    print(f"Gap data: {len(gaps)} days")

    econ_events = get_econ_events()
    print(f"Economic events: {len(econ_events)} dates")

    print("Running backtest...")
    all_trades = []
    daily_data = {}
    streak = (0, 0)
    ib_ranges = []

    for date in dates:
        dc = caches[date]
        arrays = arrays_cache[date]
        ib_range = dc.ibh - dc.ibl
        ib_ranges.append(ib_range)

        trades, streak = simulate_day(
            dc, arrays,
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: PER_LEVEL_TS.get(lv, (8, 25))[1],
            max_per_level_map=PER_LEVEL_CAPS,
            exclude_levels={"FIB_0.5", "IBH"},
            include_ibl=False, include_vwap=False,
        )

        day_pnl = sum(t.pnl_usd for t in trades)
        # Rolling 20-day average IB range
        idx = dates.index(date)
        if idx >= 20:
            avg_ib = np.mean(ib_ranges[idx-20:idx])
        else:
            avg_ib = np.mean(ib_ranges[:idx+1])
        ib_ratio = ib_range / avg_ib if avg_ib > 0 else 1.0

        daily_data[date] = {
            "pnl": day_pnl,
            "trades": len(trades),
            "wins": sum(1 for t in trades if t.pnl_usd >= 0),
            "losses": sum(1 for t in trades if t.pnl_usd < 0),
            "ib_range": ib_range,
            "ib_range_avg20": avg_ib,
            "ib_ratio": ib_ratio,
            "vix": vix_data.get(date),
            "gap_pct": gaps.get(date, {}).get("gap_pct"),
            "gap_pts": gaps.get(date, {}).get("gap_pts"),
            "econ_event": econ_events.get(date),
            "day_of_week": date.weekday(),  # 0=Mon, 4=Fri
            "first_trade_win": trades[0].pnl_usd >= 0 if trades else None,
            "all_same_dir": len(set(t.direction for t in trades if t.pnl_usd < 0)) <= 1 if trades else None,
        }
        all_trades.extend(trades)

    # Save full trade data for future analysis
    print("Saving trade data...")
    daily_context = {str(d): v for d, v in daily_data.items()}
    path = save_trades_data("deployed_5level_analysis", all_trades, daily_context)
    print(f"Saved to {path}")

    print(f"\n=== Phase 2: Analyze losing days vs winning days ===")
    print(f"Total: {len(dates)} days, {len(all_trades)} trades")

    winning_days = [(d, daily_data[d]) for d in dates if daily_data[d]["pnl"] >= 0]
    losing_days = [(d, daily_data[d]) for d in dates if daily_data[d]["pnl"] < 0]
    bad_days = [(d, daily_data[d]) for d in dates if daily_data[d]["pnl"] <= -100]
    print(f"Winning days: {len(winning_days)}, Losing days: {len(losing_days)}, -$100 days: {len(bad_days)}")

    # === Factor analysis ===
    day_pnls = [daily_data[d]["pnl"] for d in dates]

    # 1. IB Range
    ib_ranges_list = [daily_data[d]["ib_range"] for d in dates]
    analyze_bucket("IB Range (absolute)", ib_ranges_list, day_pnls,
        lambda v: "<100" if v < 100 else "<150" if v < 150 else "<200" if v < 200 else "<250" if v < 250 else "250+",
        ["<100", "<150", "<200", "<250", "250+"])

    # 2. IB Range vs 20-day average
    ib_ratios = [daily_data[d]["ib_ratio"] for d in dates]
    analyze_bucket("IB Range vs 20d avg", ib_ratios, day_pnls,
        lambda v: "<0.6" if v < 0.6 else "<0.8" if v < 0.8 else "<1.0" if v < 1.0 else "<1.2" if v < 1.2 else "<1.5" if v < 1.5 else "1.5+",
        ["<0.6", "<0.8", "<1.0", "<1.2", "<1.5", "1.5+"])

    # 3. VIX
    vix_vals = [(daily_data[d].get("vix"), daily_data[d]["pnl"]) for d in dates if daily_data[d].get("vix") is not None]
    if vix_vals:
        vix_v, vix_p = zip(*vix_vals)
        analyze_bucket("VIX level", list(vix_v), list(vix_p),
            lambda v: "<15" if v < 15 else "<20" if v < 20 else "<25" if v < 25 else "<30" if v < 30 else "30+",
            ["<15", "<20", "<25", "<30", "30+"])

    # 4. Opening gap
    gap_vals = [(daily_data[d].get("gap_pct"), daily_data[d]["pnl"]) for d in dates if daily_data[d].get("gap_pct") is not None]
    if gap_vals:
        gap_v, gap_p = zip(*gap_vals)
        analyze_bucket("Opening gap (%)", [abs(v) for v in gap_v], list(gap_p),
            lambda v: "<0.1%" if v < 0.1 else "<0.3%" if v < 0.3 else "<0.5%" if v < 0.5 else "<1.0%" if v < 1.0 else "1.0%+",
            ["<0.1%", "<0.3%", "<0.5%", "<1.0%", "1.0%+"])

    # 5. Economic events
    econ_pnls = defaultdict(list)
    for d in dates:
        event = daily_data[d].get("econ_event")
        econ_pnls[event or "None"].append(daily_data[d]["pnl"])
    print(f"\n  Economic events:")
    print(f"  {'Event':<10} {'Days':>5} {'Avg $/day':>9} {'WR%':>6}")
    print(f"  {'-'*35}")
    for event in ["None", "FOMC", "CPI", "NFP"]:
        if event not in econ_pnls:
            continue
        pnls = econ_pnls[event]
        w = sum(1 for p in pnls if p >= 0)
        avg = np.mean(pnls)
        wr = w / len(pnls) * 100
        print(f"  {event:<10} {len(pnls):>5} ${avg:>+8.2f} {wr:>5.1f}%")

    # 6. Day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    dow_vals = [daily_data[d]["day_of_week"] for d in dates]
    analyze_bucket("Day of week", dow_vals, day_pnls,
        lambda v: dow_names[v] if v < 5 else None,
        dow_names)

    # 7. First trade outcome
    first_win = [(1 if daily_data[d]["first_trade_win"] else 0, daily_data[d]["pnl"])
                 for d in dates if daily_data[d]["first_trade_win"] is not None]
    if first_win:
        fw_v, fw_p = zip(*first_win)
        analyze_bucket("First trade outcome", list(fw_v), list(fw_p),
            lambda v: "Win" if v == 1 else "Loss",
            ["Win", "Loss"])

    # 8. Number of trades per day
    trade_counts = [daily_data[d]["trades"] for d in dates]
    analyze_bucket("Trades per day", trade_counts, day_pnls,
        lambda v: "0" if v == 0 else "1-5" if v <= 5 else "6-10" if v <= 10 else "11-20" if v <= 20 else "21-30" if v <= 30 else "31+",
        ["0", "1-5", "6-10", "11-20", "21-30", "31+"])

    # 9. ADR consumed (session move / IB range)
    # Use max session move across all trades on that day
    for d in dates:
        dc = caches[d]
        post_ib = dc.post_ib_prices
        if len(post_ib) > 0:
            session_high = float(np.max(post_ib))
            session_low = float(np.min(post_ib))
            session_range = session_high - session_low
            ib_range = dc.ibh - dc.ibl
            daily_data[d]["adr_consumed"] = session_range / ib_range if ib_range > 0 else 0

    adr_vals = [daily_data[d].get("adr_consumed", 0) for d in dates]
    analyze_bucket("ADR consumed (session range / IB range)", adr_vals, day_pnls,
        lambda v: "<0.5x" if v < 0.5 else "<1.0x" if v < 1.0 else "<1.5x" if v < 1.5 else "<2.0x" if v < 2.0 else "2.0x+",
        ["<0.5x", "<1.0x", "<1.5x", "<2.0x", "2.0x+"])

    # 10. Losing direction consistency (all losses same direction = trending day)
    dir_vals = []
    for d in dates:
        dd = daily_data[d]
        if dd["losses"] > 0:
            dir_vals.append((1 if dd["all_same_dir"] else 0, dd["pnl"]))
    if dir_vals:
        dv, dp = zip(*dir_vals)
        analyze_bucket("All losses same direction (trend signal)", list(dv), list(dp),
            lambda v: "Same dir (trending)" if v == 1 else "Mixed dir",
            ["Same dir (trending)", "Mixed dir"])

    # Summary: factors with biggest winning vs losing day difference
    print(f"\n{'='*60}")
    print("FACTOR SUMMARY: Winning vs Losing day means")
    print(f"{'='*60}")
    factors_to_compare = {
        "IB Range": ("ib_range", None),
        "IB Ratio (vs 20d avg)": ("ib_ratio", None),
        "VIX": ("vix", None),
        "Gap % (abs)": ("gap_pct", abs),
        "Trades/day": ("trades", None),
    }
    for name, (key, transform) in factors_to_compare.items():
        w_vals = [daily_data[d][key] for d in dates if daily_data[d]["pnl"] >= 0 and daily_data[d].get(key) is not None]
        l_vals = [daily_data[d][key] for d in dates if daily_data[d]["pnl"] < 0 and daily_data[d].get(key) is not None]
        if transform:
            w_vals = [transform(v) for v in w_vals]
            l_vals = [transform(v) for v in l_vals]
        if w_vals and l_vals:
            result = analyze_factor(name, w_vals, l_vals)
            print(f"  {name:<25} W={result['winning_mean']:>8.2f}  L={result['losing_mean']:>8.2f}  diff={result['diff_mean']:>+8.2f}")


if __name__ == "__main__":
    main()
