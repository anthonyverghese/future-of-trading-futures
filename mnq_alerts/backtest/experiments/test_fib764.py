"""Quick backtest: compare FIB_0.764 (corrected) vs FIB_0.786 (old).

Runs the full simulation with interior fibs to verify performance
of the corrected level and check if T/S needs updating.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from config import BOT_PER_LEVEL_TS, BOT_MAX_ENTRIES_PER_LEVEL

# Per-level target/stop from deployed config.
PER_LEVEL_TS = {
    "IBH": (6, 20),
    "FIB_EXT_HI_1.272": (6, 20),
    "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25),
    "FIB_0.5": (10, 25),
    "FIB_0.618": (12, 20),
    "FIB_0.764": (10, 25),
}


def target_fn(level):
    return PER_LEVEL_TS.get(level, (8, 25))[0]

def stop_fn(level):
    return PER_LEVEL_TS.get(level, (8, 25))[1]

def main():
    print("Loading data...")
    dates, caches = load_all_days()
    print(f"Loaded {len(dates)} days")

    # Run simulation.
    total_pnl = 0.0
    total_trades = 0
    wins = 0
    losses = 0
    level_stats = {}
    streak = (0, 0)

    for date in dates:
        dc = caches[date]
        arrays = precompute_arrays(dc)

        trades, streak = simulate_day(
            dc, arrays,
            zone_factory=lambda name, price, drifts: BotZoneTradeReset(price, drifts),
            target_fn=target_fn,
            max_per_level=BOT_MAX_ENTRIES_PER_LEVEL,
            streak_state=streak,
            stop_fn=stop_fn,
            include_ibl=False,
            include_vwap=False,
        )

        for t in trades:
            total_trades += 1
            total_pnl += t.pnl_usd
            if t.pnl_usd >= 0:
                wins += 1
            else:
                losses += 1
            if t.level not in level_stats:
                level_stats[t.level] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0}
            ls = level_stats[t.level]
            ls["trades"] += 1
            ls["pnl"] += t.pnl_usd
            if t.pnl_usd >= 0:
                ls["wins"] += 1
            else:
                ls["losses"] += 1

    print(f"\n{'='*60}")
    print(f"RESULTS ({len(dates)} days)")
    print(f"{'='*60}")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins}, Losses: {losses}")
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    print(f"Win rate: {wr:.1f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Avg P&L/day: ${total_pnl / len(dates):.2f}")
    print(f"\nPer-level breakdown:")
    print(f"{'Level':<20} {'Trades':>6} {'W':>4} {'L':>4} {'WR%':>6} {'P&L':>10} {'$/day':>8}")
    print(f"{'-'*60}")
    for level in sorted(level_stats.keys()):
        ls = level_stats[level]
        lwr = ls["wins"] / ls["trades"] * 100 if ls["trades"] > 0 else 0
        lpd = ls["pnl"] / len(dates)
        print(f"{level:<20} {ls['trades']:>6} {ls['wins']:>4} {ls['losses']:>4} {lwr:>5.1f}% ${ls['pnl']:>9.2f} ${lpd:>7.2f}")


if __name__ == "__main__":
    main()
