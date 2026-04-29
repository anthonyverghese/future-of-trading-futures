"""Walk-forward scoring: train weights on current 5-level config.

Instead of using stale weights from old config, trains fresh weights
on the first 200 days and validates on the last 132 days. Uses the
same train_weights methodology from backtest/scoring.py but on
trades from the deployed config.

Phase 1: Run full backtest unscored, collect all trades with factors.
Phase 2: Split into train (first 200 days) / test (last 132 days).
Phase 3: Train weights from train set using suggest_weight.
Phase 4: Test multiple min_score thresholds on held-out test set.

Usage:
    PYTHONPATH=. python -u mnq_alerts/backtest/experiments/test_walkforward_scoring.py
"""
import os, sys, time, datetime
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from mnq_alerts.backtest.data import load_all_days, precompute_arrays
from mnq_alerts.backtest.simulate import simulate_day, TradeRecord
from mnq_alerts.backtest.zones import BotZoneTradeReset
from mnq_alerts.backtest.scoring import EntryFactors, score_entry
from mnq_alerts.backtest.results import compute_stats

# Try to import suggest_weight
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from mnq_alerts.score_optimizer import suggest_weight
except ImportError:
    from score_optimizer import suggest_weight

BASE_TS = {
    "FIB_EXT_HI_1.272": (6, 20), "FIB_EXT_LO_1.272": (6, 20),
    "FIB_0.236": (8, 25), "FIB_0.618": (12, 20), "FIB_0.764": (10, 25),
}
BASE_CAPS = {
    "FIB_0.236": 12, "FIB_0.618": 3, "FIB_0.764": 5,
    "FIB_EXT_HI_1.272": 6, "FIB_EXT_LO_1.272": 6,
}
BASE_EXCLUDE = {"FIB_0.5", "IBH", "IBL"}

TRAIN_DAYS = 200
# Levels in current config
LEVELS = ["FIB_0.236", "FIB_0.618", "FIB_0.764", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]


def train_fresh_weights(trades):
    """Train weights from trade outcomes using suggest_weight.

    Creates weights for all scoring factors based on actual WR
    per factor bucket on the current 5-level config.
    """
    total = len(trades)
    if total == 0:
        return {}
    wins = sum(1 for t in trades if t.outcome == "win")
    baseline_wr = wins / total * 100
    sw = suggest_weight

    def wr(fn):
        sub = [t for t in trades if fn(t)]
        if len(sub) < 30:
            return baseline_wr
        return sum(1 for t in sub if t.outcome == "win") / len(sub) * 100

    w = {}

    # Level quality — now includes interior fibs
    for lv in LEVELS:
        key = f"lv_{lv.replace('.', '_').lower()}"
        w[key] = sw(wr(lambda t, l=lv: t.level == l), baseline_wr)

    # Direction combos — all 10 combinations
    for lv in LEVELS:
        for d in ["up", "down"]:
            key = f"co_{lv.replace('.', '_').lower()}_{d}"
            w[key] = sw(wr(lambda t, l=lv, dr=d: t.level == l and t.direction == dr), baseline_wr)

    # Time of day
    w["t_postib"] = sw(wr(lambda t: 631 <= t.factors.et_mins < 690), baseline_wr)
    w["t_midday"] = sw(wr(lambda t: 720 <= t.factors.et_mins < 840), baseline_wr)
    w["t_power"] = sw(wr(lambda t: t.factors.et_mins >= 900), baseline_wr)

    # Tick rate buckets
    w["tr_low"] = sw(wr(lambda t: t.factors.tick_rate < 500), baseline_wr)
    w["tr_med"] = sw(wr(lambda t: 1000 <= t.factors.tick_rate < 2000), baseline_wr)
    w["tr_high"] = sw(wr(lambda t: t.factors.tick_rate >= 3000), baseline_wr)

    # Entry count
    w["ec_1"] = sw(wr(lambda t: t.entry_count == 1), baseline_wr)
    w["ec_2"] = sw(wr(lambda t: t.entry_count == 2), baseline_wr)
    w["ec_3"] = sw(wr(lambda t: t.entry_count == 3), baseline_wr)
    w["ec_4plus"] = sw(wr(lambda t: t.entry_count >= 4), baseline_wr)

    # Session move
    w["sm_big_down"] = sw(wr(lambda t: t.factors.session_move < -50), baseline_wr)
    w["sm_small_down"] = sw(wr(lambda t: -50 <= t.factors.session_move < -10), baseline_wr)
    w["sm_neutral"] = sw(wr(lambda t: -10 <= t.factors.session_move <= 10), baseline_wr)
    w["sm_small_up"] = sw(wr(lambda t: 10 < t.factors.session_move <= 50), baseline_wr)
    w["sm_big_up"] = sw(wr(lambda t: t.factors.session_move > 50), baseline_wr)

    # 30m range
    w["r30_low"] = sw(wr(lambda t: t.factors.range_30m < 50), baseline_wr)
    w["r30_med"] = sw(wr(lambda t: 50 <= t.factors.range_30m < 100), baseline_wr)
    w["r30_high"] = sw(wr(lambda t: t.factors.range_30m >= 150), baseline_wr)

    # Approach speed
    w["asp_slow"] = sw(wr(lambda t: t.factors.approach_speed < 0.5), baseline_wr)
    w["asp_fast"] = sw(wr(lambda t: t.factors.approach_speed >= 3.0), baseline_wr)

    return w, baseline_wr


def score_trade(t, weights):
    """Score a trade using trained weights."""
    s = 0
    lv_key = f"lv_{t.level.replace('.', '_').lower()}"
    s += weights.get(lv_key, 0)

    co_key = f"co_{t.level.replace('.', '_').lower()}_{t.direction}"
    s += weights.get(co_key, 0)

    et = t.factors.et_mins
    if 631 <= et < 690: s += weights.get("t_postib", 0)
    elif 720 <= et < 840: s += weights.get("t_midday", 0)
    elif et >= 900: s += weights.get("t_power", 0)

    tr = t.factors.tick_rate
    if tr < 500: s += weights.get("tr_low", 0)
    elif 1000 <= tr < 2000: s += weights.get("tr_med", 0)
    elif tr >= 3000: s += weights.get("tr_high", 0)

    ec = t.entry_count
    if ec == 1: s += weights.get("ec_1", 0)
    elif ec == 2: s += weights.get("ec_2", 0)
    elif ec == 3: s += weights.get("ec_3", 0)
    elif ec >= 4: s += weights.get("ec_4plus", 0)

    sm = t.factors.session_move
    if sm < -50: s += weights.get("sm_big_down", 0)
    elif sm < -10: s += weights.get("sm_small_down", 0)
    elif sm <= 10: s += weights.get("sm_neutral", 0)
    elif sm <= 50: s += weights.get("sm_small_up", 0)
    else: s += weights.get("sm_big_up", 0)

    r30 = t.factors.range_30m
    if r30 < 50: s += weights.get("r30_low", 0)
    elif r30 < 100: s += weights.get("r30_med", 0)
    elif r30 >= 150: s += weights.get("r30_high", 0)

    asp = t.factors.approach_speed
    if asp < 0.5: s += weights.get("asp_slow", 0)
    elif asp >= 3.0: s += weights.get("asp_fast", 0)

    return s


def main():
    t0 = time.time()
    print("=== Phase 1: Collect all trades ===", flush=True)
    dates, caches = load_all_days()
    arrays = {d: precompute_arrays(caches[d]) for d in dates}
    print(f"Loaded {len(dates)} days in {time.time()-t0:.0f}s")

    # Run unscored to get all trades
    all_trades = []
    daily_trades = defaultdict(list)
    streak = (0, 0)
    for date in dates:
        caps = dict(BASE_CAPS)
        if date.weekday() == 0:
            caps = {k: v * 2 for k, v in caps.items()}
        trades, streak = simulate_day(
            caches[date], arrays[date],
            zone_factory=lambda n, p, dr: BotZoneTradeReset(p, dr),
            target_fn=lambda lv: BASE_TS.get(lv, (8, 25))[0],
            stop_fn=lambda lv: BASE_TS.get(lv, (8, 25))[1],
            max_per_level_map=caps,
            exclude_levels=BASE_EXCLUDE,
            include_ibl=False, include_vwap=False,
            global_cooldown_after_loss_secs=30,
        )
        all_trades.extend(trades)
        daily_trades[date] = trades

    print(f"Total trades: {len(all_trades)}")

    print(f"\n=== Phase 2: Train/test split ===")
    train_dates = dates[:TRAIN_DAYS]
    test_dates = dates[TRAIN_DAYS:]
    train_trades = [t for t in all_trades if t.date in set(train_dates)]
    test_trades = [t for t in all_trades if t.date in set(test_dates)]
    print(f"Train: {len(train_dates)} days, {len(train_trades)} trades")
    print(f"Test:  {len(test_dates)} days, {len(test_trades)} trades")

    print(f"\n=== Phase 3: Train weights ===")
    weights, baseline_wr = train_fresh_weights(train_trades)
    print(f"Baseline WR (train): {baseline_wr:.1f}%")
    print(f"Trained weights:")
    for k, v in sorted(weights.items()):
        if v != 0:
            print(f"  {k:<30} {v:+d}")

    print(f"\n=== Phase 4: Score distribution on test set ===")
    test_scores = [(score_trade(t, weights), t) for t in test_trades]
    score_dist = defaultdict(lambda: {"w": 0, "l": 0})
    for sc, t in test_scores:
        if t.outcome == "win":
            score_dist[sc]["w"] += 1
        else:
            score_dist[sc]["l"] += 1

    print(f"{'Score':>6} {'Trades':>7} {'WR%':>6} {'Cumul Trades':>12} {'Cumul WR%':>10}")
    print("-" * 50)
    cumul_w = sum(s["w"] for s in score_dist.values())
    cumul_l = sum(s["l"] for s in score_dist.values())
    for sc in sorted(score_dist.keys()):
        s = score_dist[sc]
        total = s["w"] + s["l"]
        wr = s["w"] / total * 100 if total > 0 else 0
        cumul_wr = cumul_w / (cumul_w + cumul_l) * 100 if (cumul_w + cumul_l) > 0 else 0
        print(f"{sc:>6} {total:>7} {wr:>5.1f}% {cumul_w + cumul_l:>12} {cumul_wr:>9.1f}%")
        cumul_w -= s["w"]
        cumul_l -= s["l"]

    print(f"\n=== Phase 5: Test thresholds on held-out data ===")
    thresholds = [-5, -4, -3, -2, -1, 0, 1, 2, 3]

    # Baseline: unscored test set
    test_pnl = sum(t.pnl_usd for t in test_trades)
    test_wr = sum(1 for t in test_trades if t.outcome == "win") / len(test_trades) * 100
    n_test = len(test_dates)
    print(f"\nBaseline (unscored): {len(test_trades)} trades, {test_wr:.1f}% WR, ${test_pnl/n_test:.2f}/day")

    print(f"\n{'Threshold':>10} {'Trades':>7} {'Removed':>8} {'WR%':>6} {'$/day':>8} {'vs base':>8}")
    print("-" * 55)
    for threshold in thresholds:
        kept = [(sc, t) for sc, t in test_scores if sc >= threshold]
        if not kept:
            continue
        kept_trades = [t for _, t in kept]
        removed = len(test_trades) - len(kept_trades)
        pnl = sum(t.pnl_usd for t in kept_trades)
        w = sum(1 for t in kept_trades if t.outcome == "win")
        wr = w / len(kept_trades) * 100 if kept_trades else 0
        pnl_day = pnl / n_test
        diff = pnl_day - test_pnl / n_test
        print(f"{'>=' + str(threshold):>10} {len(kept_trades):>7} {removed:>8} {wr:>5.1f}% ${pnl_day:>7.2f} ${diff:>+7.2f}")

    # Save
    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"walkforward_scoring_{ts}.json"
    result = {
        "train_days": TRAIN_DAYS,
        "test_days": len(test_dates),
        "train_trades": len(train_trades),
        "test_trades": len(test_trades),
        "baseline_wr": baseline_wr,
        "weights": weights,
        "score_distribution": {str(k): v for k, v in score_dist.items()},
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to: {path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
