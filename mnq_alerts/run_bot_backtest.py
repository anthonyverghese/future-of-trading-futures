"""Run bot backtest — unbiased parameter sweep.

Defines a parameter grid and tests ALL combinations. Results ranked by
a single consistent metric (OOS P&L/day). No config is labeled "current"
or "best" — the data speaks for itself.

Usage:
    python -u run_bot_backtest.py
"""

import sys, os, time, datetime
sys.path.insert(0, os.path.dirname(__file__))

from backtest.data import load_all_days, precompute_arrays
from backtest.zones import BotZoneTradeReset
from backtest.scoring import score_entry, train_weights, EntryFactors
from backtest.simulate import simulate_day, TradeRecord
from backtest.report import fmt, per_level_breakdown
from backtest.results import (
    BacktestParams, BacktestResult, compute_stats, save_result, display_results,
)
from backtest.evaluate import PER_LEVEL_TARGETS

INITIAL_TRAIN_DAYS = 60
STEP_DAYS = 20


# ═══════════════════════════════════════════════════════════════
# PARAMETER GRID — define all values to sweep
# No labels, no favorites. Every combination is tested equally.
# ═══════════════════════════════════════════════════════════════

TARGET_CONFIGS = {
    "T8": lambda lv: 8.0,
    "T6": lambda lv: 6.0,
    "T10": lambda lv: 10.0,
    "adj-level": lambda lv: float({"IBH": 8, "IBL": 8, "FIB_EXT_HI_1.272": 5, "FIB_EXT_LO_1.272": 8, "VWAP": 6}.get(lv, 8)),
}

STOP_VALUES = [20.0]
MAX_PER_LEVEL_VALUES = [8, 12]
MIN_SCORE_VALUES = [-1, 0, 1]


# ═══════════════════════════════════════════════════════════════
# RANKING METRIC — one metric, consistently applied
# ═══════════════════════════════════════════════════════════════

def rank_metric(oos_stats: dict, recent_stats: dict) -> float:
    """Single metric for ranking configs. Higher = better.

    Uses OOS P&L/day as primary metric. This is the most honest
    measure of expected live performance.
    """
    return oos_stats.get("pnl_per_day", 0)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 110)
    print("  BOT BACKTEST — Unbiased parameter sweep")
    print("=" * 110)

    # Load + precompute.
    valid_days, day_caches = load_all_days()
    N = len(valid_days)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    print(f"  Precomputing factor arrays...", flush=True)
    t1 = time.time()
    arrays = {}
    for i, date in enumerate(valid_days):
        arrays[date] = precompute_arrays(day_caches[date])
    print(f"  Done in {time.time()-t1:.0f}s")

    def zone_factory(name, price, drifts):
        return BotZoneTradeReset(price, drifts)

    # Generate all config combinations.
    configs = []
    for tgt_name, tgt_fn in TARGET_CONFIGS.items():
        for stop in STOP_VALUES:
            for mpl in MAX_PER_LEVEL_VALUES:
                for ms in MIN_SCORE_VALUES:
                    label = f"{tgt_name}/S{int(stop)} max={mpl} score>={ms}"
                    configs.append((label, tgt_fn, stop, mpl, ms))

    total_configs = len(configs)
    print(f"\n  Testing {total_configs} configurations\n")

    # Run each config.
    all_results = []

    for ci, (label, target_fn, stop, mpl, min_score) in enumerate(configs):
        print(f"  [{ci+1}/{total_configs}] {label}", flush=True)

        # Simulate without scoring for training data.
        all_trades = []
        trades_by_date = {}
        cw = cl = 0
        for date in valid_days:
            trades, (cw, cl) = simulate_day(
                day_caches[date], arrays[date], zone_factory,
                target_fn, stop, mpl,
                weights=None, min_score=-99,
                streak_state=(cw, cl),
            )
            all_trades.extend(trades)
            trades_by_date[date] = trades

        # Train full-data weights.
        training_data = []
        cw_t = cl_t = 0
        for t in all_trades:
            training_data.append((t.factors, t.outcome, cw_t, cl_t))
            if t.pnl_usd >= 0: cw_t += 1; cl_t = 0
            else: cw_t = 0; cl_t += 1
        w_full = train_weights(training_data)

        # In-sample with scoring.
        is_trades = []
        cw = cl = 0
        for date in valid_days:
            trades, (cw, cl) = simulate_day(
                day_caches[date], arrays[date], zone_factory,
                target_fn, stop, mpl,
                weights=w_full, min_score=min_score,
                streak_state=(cw, cl),
            )
            is_trades.extend(trades)

        # Walk-forward OOS.
        oos_trades = []
        oos_days = 0
        k = INITIAL_TRAIN_DAYS
        while k < N:
            train_days = valid_days[:k]
            test_days = valid_days[k:k + STEP_DAYS]
            if not test_days: break
            oos_days += len(test_days)

            train_t = []
            cw_t = cl_t = 0
            for d in train_days:
                for t in trades_by_date.get(d, []):
                    train_t.append((t.factors, t.outcome, cw_t, cl_t))
                    if t.pnl_usd >= 0: cw_t += 1; cl_t = 0
                    else: cw_t = 0; cl_t += 1
            wt = train_weights(train_t)

            cw = cl = 0
            for d in test_days:
                trades, (cw, cl) = simulate_day(
                    day_caches[d], arrays[d], zone_factory,
                    target_fn, stop, mpl,
                    weights=wt, min_score=min_score,
                    streak_state=(cw, cl),
                )
                oos_trades.extend(trades)
            k += STEP_DAYS

        # Recent 60 days.
        recent = valid_days[-60:]
        rn = len(recent)
        pre = [d for d in valid_days if d < recent[0]]
        pre_t = []
        cw_t = cl_t = 0
        for d in pre:
            for t in trades_by_date.get(d, []):
                pre_t.append((t.factors, t.outcome, cw_t, cl_t))
                if t.pnl_usd >= 0: cw_t += 1; cl_t = 0
                else: cw_t = 0; cl_t += 1
        wr = train_weights(pre_t)
        recent_trades = []
        cw = cl = 0
        for d in recent:
            trades, (cw, cl) = simulate_day(
                day_caches[d], arrays[d], zone_factory,
                target_fn, stop, mpl,
                weights=wr, min_score=min_score,
                streak_state=(cw, cl),
            )
            recent_trades.extend(trades)

        # Compute stats.
        is_s = compute_stats(is_trades, N)
        oos_s = compute_stats(oos_trades, oos_days)
        rec_s = compute_stats(recent_trades, rn)
        rank = rank_metric(oos_s, rec_s)

        all_results.append((label, is_s, oos_s, rec_s, rank))

        # Save.
        sample_targets = {lv: target_fn(lv) for lv in ["IBH","IBL","FIB_EXT_HI_1.272","FIB_EXT_LO_1.272","VWAP"]}
        if len(set(sample_targets.values())) == 1:
            tgt_desc = f"T{int(list(sample_targets.values())[0])}"
            plv = None
        else:
            tgt_desc = "adj-level"
            plv = sample_targets

        save_result(BacktestResult(
            params=BacktestParams(
                zone_type="BotZoneTradeReset", entry_threshold=1.0,
                exit_threshold="trade_reset",
                target=tgt_desc, per_level_targets=plv,
                stop_pts=stop, min_score=min_score,
                weights_type="trained", weights_values=None,
                scoring_factors=["level","combo","time","tick_sweet","tick_low",
                    "entry_count","session_move","streak","vol","approach","density"],
                include_vwap=True, vol_filter_pct=None,
                max_per_level=mpl, timeout_secs=900,
                time_exit_secs=None, trailing_stop=None,
                daily_loss=150.0, max_consec=None,
                data_days=N, data_range=f"{valid_days[0]} to {valid_days[-1]}",
                train_days=INITIAL_TRAIN_DAYS, step_days=STEP_DAYS,
                description=label,
            ),
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            is_trades=is_s["trades"], is_trades_per_day=is_s["trades_per_day"],
            is_wr=is_s["wr"], is_pnl_per_day=is_s["pnl_per_day"], is_max_dd=is_s["max_dd"],
            oos_days=oos_days, oos_trades=oos_s["trades"],
            oos_trades_per_day=oos_s["trades_per_day"],
            oos_wr=oos_s["wr"], oos_pnl_per_day=oos_s["pnl_per_day"], oos_max_dd=oos_s["max_dd"],
            recent_days=rn, recent_trades=rec_s["trades"],
            recent_trades_per_day=rec_s["trades_per_day"],
            recent_wr=rec_s["wr"], recent_pnl_per_day=rec_s["pnl_per_day"],
            recent_max_dd=rec_s["max_dd"],
        ))

    # Sort by ranking metric and display.
    all_results.sort(key=lambda x: x[4], reverse=True)

    print(f"\n{'='*110}")
    print(f"  RESULTS — ranked by OOS P&L/day")
    print(f"{'='*110}\n")
    print(f"  {'#':>3} {'Config':>35s} {'OOS $/d':>8} {'OOS WR':>7} {'OOS DD':>7} "
          f"{'Rec $/d':>8} {'Rec WR':>7} {'IS $/d':>7} {'OOS t/d':>7}")
    print(f"  {'-'*3} {'-'*35} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    for i, (label, is_s, oos_s, rec_s, rank) in enumerate(all_results):
        print(
            f"  {i+1:>3} {label:>35s} "
            f"{oos_s['pnl_per_day']:>+7.1f} {oos_s['wr']:>6.1f}% {oos_s['max_dd']:>6,.0f} "
            f"{rec_s['pnl_per_day']:>+7.1f} {rec_s['wr']:>6.1f}% "
            f"{is_s['pnl_per_day']:>+6.1f} {oos_s['trades_per_day']:>6.1f}"
        )

    # Data-driven recommendations.
    print(f"\n{'='*110}")
    print(f"  RECOMMENDATIONS (data-driven)")
    print(f"{'='*110}\n")

    top = all_results[0]
    print(f"  Best OOS P&L/day: {top[0]} at ${top[2]['pnl_per_day']:+.1f}/day")

    # Best recent-regime (for current market conditions).
    best_recent = max(all_results, key=lambda x: x[3].get("pnl_per_day", -999))
    print(f"  Best recent P&L/day: {best_recent[0]} at ${best_recent[3]['pnl_per_day']:+.1f}/day")

    # Best risk-adjusted (OOS P&L / MaxDD).
    best_risk = max(all_results, key=lambda x: x[2]["pnl_per_day"] / max(x[2]["max_dd"], 1))
    ratio = best_risk[2]["pnl_per_day"] / max(best_risk[2]["max_dd"], 1)
    print(f"  Best risk-adjusted: {best_risk[0]} (P&L/DD ratio: {ratio:.4f})")

    # Flag if different configs win on different metrics.
    if top[0] != best_recent[0]:
        print(f"\n  NOTE: OOS winner and recent winner differ. Consider recent-regime")
        print(f"  performance for live deployment since it reflects current conditions.")

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
