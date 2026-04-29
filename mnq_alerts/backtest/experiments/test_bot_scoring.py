"""Test bot_entry_score with different min_score thresholds.

Uses the existing bot_entry_score function (trained on old config)
to see if any scoring threshold improves P&L on the current 5-level
config. The old weights may not be optimal but this tests whether
scoring has signal at all.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_bot_scoring.py
"""
import os, sys, time, datetime
import multiprocessing
multiprocessing.set_start_method("fork", force=True)
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.results import compute_stats
from mnq_alerts.bot_trader import bot_entry_score

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH", "IBL"}

_DATES = None
_CACHES = None
_ARRAYS = None


def _make_score_fn(et_tz):
    """Create a score function that wraps bot_entry_score.

    bot_entry_score takes different args than EntryFactors, so we
    need to adapt.
    """
    def score_fn(fac):
        # Convert et_mins back to a time object for bot_entry_score.
        h = fac.et_mins // 60
        m = fac.et_mins % 60
        now_et = datetime.time(h, m) if 0 <= h < 24 else None

        # bot_entry_score uses session_move_pct (percentage),
        # but fac.session_move is in points. Approximate the pct.
        # At MNQ ~27000, 1pt = 0.0037%. Use fac.range_30m for vol.
        price_approx = 27000.0  # rough MNQ price for pct conversion
        session_move_pct = fac.session_move / price_approx * 100 if price_approx > 0 else 0
        range_30m_pct = fac.range_30m / price_approx * 100 if price_approx > 0 else None

        return bot_entry_score(
            level=fac.level,
            direction=fac.direction,
            entry_count=fac.entry_count,
            trend_60m=0.0,  # not available in backtest factors
            tick_rate=fac.tick_rate,
            session_move_pct=session_move_pct,
            range_30m_pct=range_30m_pct,
            now_et=now_et,
        )
    return score_fn


VARIANTS = [
    ("Baseline (no scoring)", -99),
    ("Score >= -5", -5),
    ("Score >= -4", -4),
    ("Score >= -3", -3),
    ("Score >= -2", -2),
    ("Score >= -1", -1),
    ("Score >= 0", 0),
    ("Score >= 1", 1),
    ("Score >= 2", 2),
    ("Score >= 3", 3),
]


def _run_one(args):
    name, min_score = args
    all_trades = []
    streak = (0, 0)
    sfn = _make_score_fn(None) if min_score > -99 else None

    for date in _DATES:
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}

        trades, streak = simulate_day(
            _CACHES[date], _ARRAYS[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=30,
            min_score=min_score,
            score_fn=sfn,
        )
        all_trades.extend(trades)

    stats = compute_stats(all_trades, len(_DATES), list(_DATES))
    stats["name"] = name
    return stats


def main():
    global _DATES, _CACHES, _ARRAYS
    t0 = time.time()

    print("Loading data...", flush=True)
    _DATES, _CACHES = load_all_days()
    print(f"Loaded {len(_DATES)} days in {time.time()-t0:.0f}s", flush=True)

    print("Precomputing arrays...", flush=True)
    _ARRAYS = {d: precompute_arrays(_CACHES[d]) for d in _DATES}

    n_variants = len(VARIANTS)
    print(f"Running {n_variants} variants across 3 workers...", flush=True)

    with Pool(3) as pool:
        results = pool.map(_run_one, VARIANTS)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    baseline = results[0]
    b_pnl = baseline["pnl_per_day"]

    print("=" * 120)
    print(f"{'Variant':<25} {'Trades':>6} {'WR%':>5} {'$/day':>7} {'MaxDD':>6} {'-$100d':>6} {'R60d':>7} {'R30d':>7} {'W%days':>6} {'vs base':>7}")
    print("-" * 120)
    for r in results:
        diff = r["pnl_per_day"] - b_pnl
        r60 = r.get("recent_60d_pnl_per_day", 0)
        r30 = r.get("recent_30d_pnl_per_day", 0)
        l100 = r.get("days_below_neg100", 0)
        wd = r.get("winning_days_pct", 0)
        print(
            f"{r['name']:<25} {r['trades']:>6} "
            f"{r['wr']:>5.1f} {r['pnl_per_day']:>+7.2f} "
            f"{r['max_dd']:>6.0f} {l100:>6} "
            f"{r60:>+7.2f} {r30:>+7.2f} {wd:>5.1f}% {diff:>+7.2f}"
        )

    winners = [r for r in results[1:] if r["pnl_per_day"] > b_pnl]
    if winners:
        print(f"\n  Variants that beat baseline (${b_pnl:.2f}/day):")
        for r in sorted(winners, key=lambda x: x["pnl_per_day"], reverse=True):
            diff = r["pnl_per_day"] - b_pnl
            print(f"    {r['name']:<25} ${r['pnl_per_day']:>+.2f}/day ({diff:>+.2f})")
    else:
        print(f"\n  No variants beat baseline (${b_pnl:.2f}/day)")

    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"bot_scoring_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
