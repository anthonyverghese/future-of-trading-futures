"""Backtest result storage and display.

Saves results as JSON in backtest/results/ with clear parameter metadata.
Results can be queried to see what's been tested without re-running.

The cache is informational — it never blocks running a new backtest.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from .simulate import TradeRecord

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class BacktestParams:
    """All parameters that define a backtest run."""
    # Zone logic
    zone_type: str              # "BotZoneTradeReset", "BotZoneFixedExit", "HumanZone"
    entry_threshold: float      # 1.0 for bot, 7.0 for human
    exit_threshold: str         # "trade_reset", "20.0", "15.0"

    # Target / Stop
    target: str                 # "T8" or "per-level"
    per_level_targets: dict | None  # {"IBH": 14, "IBL": 10, ...} if per-level
    stop_pts: float

    # Scoring
    min_score: int
    weights_type: str           # "trained", "human", "custom"
    weights_values: dict | None # actual weight dict (None if too large)
    scoring_factors: list[str]  # e.g. ["level","combo","time","tick","entry_count","session_move","streak","vol","approach","density"]

    # Levels
    levels: list[str] | None = None  # e.g. ["IBH","IBL","VWAP","FIB_EXT_HI","FIB_EXT_LO","FIB_0.5"]
    include_vwap: bool = True

    # Filters
    vol_filter_pct: float | None = None  # 0.15 or None (disabled)
    max_per_level: int = 12

    # Approach features (Phase 2)
    approach_features: list[str] | None = None  # e.g. ["duration","velocity","volume","deceleration"]

    # Trade management
    timeout_secs: int = 900
    time_exit_secs: int | None = None
    trailing_stop: dict | None = None

    # Risk limits
    daily_loss: float = 100.0
    max_consec: int | None = None

    # Data
    data_days: int = 0
    data_range: str = ""

    # Walk-forward
    train_days: int = 60
    step_days: int = 20

    # Meta
    description: str = ""


@dataclass
class BacktestResult:
    """Results from a single backtest configuration."""
    params: BacktestParams
    timestamp: str           # when the backtest was run

    # In-sample
    is_trades: int
    is_trades_per_day: float
    is_wr: float
    is_pnl_per_day: float
    is_max_dd: float

    # OOS (walk-forward)
    oos_days: int
    oos_trades: int
    oos_trades_per_day: float
    oos_wr: float
    oos_pnl_per_day: float
    oos_max_dd: float

    # Recent regime (last 60 days)
    recent_days: int
    recent_trades: int
    recent_trades_per_day: float
    recent_wr: float
    recent_pnl_per_day: float
    recent_max_dd: float


def compute_stats(
    trades: list[TradeRecord],
    n_days: int,
    dates: list | None = None,
) -> dict:
    """Compute comprehensive stats from a list of trades.

    dates: ordered list of all dates in the period (for daily P&L breakdown).
    """
    if not trades:
        return {"trades": 0, "trades_per_day": 0, "wr": 0,
                "pnl_per_day": 0, "max_dd": 0}
    w = sum(1 for t in trades if t.pnl_usd >= 0)
    l = sum(1 for t in trades if t.pnl_usd < 0)
    total = w + l
    wr = w / total * 100 if total else 0
    total_pnl = sum(t.pnl_usd for t in trades)

    # Max drawdown (trade-by-trade equity curve).
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    stats = {
        "trades": len(trades),
        "trades_per_day": round(len(trades) / n_days, 1) if n_days else 0,
        "wins": w,
        "losses": l,
        "wr": round(wr, 1),
        "total_pnl": round(total_pnl, 2),
        "pnl_per_day": round(total_pnl / n_days, 2) if n_days else 0,
        "max_dd": round(max_dd, 0),
    }

    # Daily breakdown if dates provided.
    if dates:
        from collections import defaultdict
        daily_pnl = defaultdict(float)
        for t in trades:
            daily_pnl[t.date] += t.pnl_usd
        day_pnls = [daily_pnl.get(d, 0.0) for d in dates]

        winning_days = sum(1 for p in day_pnls if p >= 0)
        losing_days = sum(1 for p in day_pnls if p < 0)
        days_below_neg100 = sum(1 for p in day_pnls if p <= -100)

        stats["winning_days"] = winning_days
        stats["losing_days"] = losing_days
        stats["winning_days_pct"] = round(winning_days / len(dates) * 100, 1) if dates else 0
        stats["days_below_neg100"] = days_below_neg100

        # Recent 60 and 30 days.
        if len(dates) >= 60:
            recent_60_dates = set(dates[-60:])
            r60 = [t for t in trades if t.date in recent_60_dates]
            r60_pnl = sum(t.pnl_usd for t in r60)
            r60_w = sum(1 for t in r60 if t.pnl_usd >= 0)
            r60_total = len(r60)
            stats["recent_60d_pnl_per_day"] = round(r60_pnl / 60, 2)
            stats["recent_60d_wr"] = round(r60_w / r60_total * 100, 1) if r60_total else 0

        if len(dates) >= 30:
            recent_30_dates = set(dates[-30:])
            r30 = [t for t in trades if t.date in recent_30_dates]
            r30_pnl = sum(t.pnl_usd for t in r30)
            r30_w = sum(1 for t in r30 if t.pnl_usd >= 0)
            r30_total = len(r30)
            stats["recent_30d_pnl_per_day"] = round(r30_pnl / 30, 2)
            stats["recent_30d_wr"] = round(r30_w / r30_total * 100, 1) if r30_total else 0

        # Quarterly breakdown.
        n = len(dates)
        if n >= 4:
            q = n // 4
            quarterly = {}
            for i, label in enumerate(["Q1_oldest", "Q2", "Q3", "Q4_newest"]):
                q_dates = set(dates[i*q : (i+1)*q if i < 3 else n])
                q_trades = [t for t in trades if t.date in q_dates]
                q_pnl = sum(t.pnl_usd for t in q_trades)
                q_nd = len(q_dates)
                quarterly[label] = round(q_pnl / q_nd, 2) if q_nd else 0
            stats["quarterly_pnl_per_day"] = quarterly

        # Per-level breakdown.
        from collections import defaultdict as dd2
        level_stats = dd2(lambda: {"w": 0, "l": 0, "pnl": 0.0})
        for t in trades:
            ls = level_stats[t.level]
            ls["pnl"] += t.pnl_usd
            if t.pnl_usd >= 0:
                ls["w"] += 1
            else:
                ls["l"] += 1
        per_level = {}
        for lv in sorted(level_stats.keys()):
            s = level_stats[lv]
            total_lv = s["w"] + s["l"]
            per_level[lv] = {
                "trades": total_lv,
                "wr": round(s["w"] / total_lv * 100, 1) if total_lv else 0,
                "pnl_per_day": round(s["pnl"] / n_days, 2) if n_days else 0,
            }
        stats["per_level"] = per_level

    return stats


def save_result(result: BacktestResult) -> str:
    """Save a result to disk. Returns the file path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    desc = result.params.description.replace(" ", "_").replace("/", "-").replace(">=", "gte")[:40] or "backtest"
    filename = f"{desc}_{ts}.json"
    path = RESULTS_DIR / filename

    data = {
        "params": asdict(result.params),
        "timestamp": result.timestamp,
        "in_sample": {
            "trades": result.is_trades,
            "trades_per_day": result.is_trades_per_day,
            "wr": result.is_wr,
            "pnl_per_day": result.is_pnl_per_day,
            "max_dd": result.is_max_dd,
        },
        "oos": {
            "days": result.oos_days,
            "trades": result.oos_trades,
            "trades_per_day": result.oos_trades_per_day,
            "wr": result.oos_wr,
            "pnl_per_day": result.oos_pnl_per_day,
            "max_dd": result.oos_max_dd,
        },
        "recent": {
            "days": result.recent_days,
            "trades": result.recent_trades,
            "trades_per_day": result.recent_trades_per_day,
            "wr": result.recent_wr,
            "pnl_per_day": result.recent_pnl_per_day,
            "max_dd": result.recent_max_dd,
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


def save_trades_data(
    name: str,
    trades: list[TradeRecord],
    daily_context: dict | None = None,
) -> str:
    """Save full trade-level data for post-hoc analysis.

    trades: all TradeRecord objects from the backtest.
    daily_context: optional dict of {date_str: {ibh, ibl, ib_range, ...}}.

    Saved as a separate JSON alongside the summary results.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    desc = name.replace(" ", "_").replace("/", "-")[:40]
    filename = f"{desc}_trades_{ts}.json"
    path = RESULTS_DIR / filename

    trade_rows = []
    for t in trades:
        row = {
            "date": str(t.date),
            "level": t.level,
            "direction": t.direction,
            "entry_count": t.entry_count,
            "outcome": t.outcome,
            "pnl_usd": round(t.pnl_usd, 2),
            "entry_idx": t.entry_idx,
            "exit_idx": t.exit_idx,
            "entry_ns": t.entry_ns,
        }
        if t.factors:
            row["et_mins"] = t.factors.et_mins
            row["tick_rate"] = round(t.factors.tick_rate, 1)
            row["session_move"] = round(t.factors.session_move, 2)
            row["range_30m"] = round(t.factors.range_30m, 2)
            row["approach_speed"] = round(t.factors.approach_speed, 2)
            row["tick_density"] = round(t.factors.tick_density, 2)
        trade_rows.append(row)

    data = {
        "name": name,
        "timestamp": ts,
        "total_trades": len(trades),
        "trades": trade_rows,
    }
    if daily_context:
        data["daily_context"] = daily_context

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


def load_all_results() -> list[dict]:
    """Load all saved results. Returns list of result dicts."""
    if not RESULTS_DIR.exists():
        return []
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except Exception:
            pass
    return results


def display_results(filter_fn=None) -> None:
    """Display saved results in a table. Optional filter function.

    filter_fn: callable(result_dict) → bool. If provided, only show matching.

    Example:
        display_results(lambda r: r["params"]["stop_pts"] == 20)
        display_results(lambda r: "trade_reset" in r["params"]["zone_type"].lower())
    """
    results = load_all_results()
    if filter_fn:
        results = [r for r in results if filter_fn(r)]

    if not results:
        print("  No matching results found.")
        return

    print(f"  {'Description':>30s} {'Zone':>15s} {'T/S':>8s} {'Score':>6s} "
          f"{'IS $/d':>7s} {'OOS $/d':>8s} {'Recent':>8s} {'OOS WR':>7s} {'OOS DD':>7s}")
    print(f"  {'-'*30} {'-'*15} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")

    for r in results:
        p = r["params"]
        oos = r["oos"]
        rec = r["recent"]
        is_ = r["in_sample"]
        zone_short = p["zone_type"].replace("BotZone", "").replace("Human", "H")
        ts = f"{p['target']}/S{int(p['stop_pts'])}"
        print(
            f"  {p['description'][:30]:>30s} {zone_short:>15s} {ts:>8s} "
            f"{'>=' + str(p['min_score']):>6s} "
            f"{is_['pnl_per_day']:>+6.1f} {oos['pnl_per_day']:>+7.1f} "
            f"{rec['pnl_per_day']:>+7.1f} {oos['wr']:>6.1f}% "
            f"{oos['max_dd']:>6,.0f}"
        )


def display_result_detail(result_dict: dict) -> None:
    """Display a single result in detail."""
    p = result_dict["params"]
    print(f"\n  === {p['description']} ===")
    print(f"  Zone: {p['zone_type']} (entry={p['entry_threshold']}pt, exit={p['exit_threshold']})")
    print(f"  Target: {p['target']}, Stop: {p['stop_pts']}")
    if p.get('per_level_targets'):
        print(f"  Per-level targets: {p['per_level_targets']}")
    print(f"  Score >= {p['min_score']}, Max/level: {p['max_per_level']}")
    print(f"  Weights: {p['weights_type']}")
    print(f"  Scoring factors: {', '.join(p.get('scoring_factors', []))}")
    print(f"  VWAP: {'yes' if p.get('include_vwap') else 'no'}, Vol filter: {p.get('vol_filter_pct', 'none')}")
    print(f"  Timeout: {p.get('timeout_secs', 900)}s, Time exit: {p.get('time_exit_secs', 'none')}, Trailing: {p.get('trailing_stop', 'none')}")
    print(f"  Data: {p['data_days']} days ({p['data_range']})")
    print(f"  Walk-forward: {p['train_days']}d train, {p['step_days']}d step")
    consec_str = str(p['max_consec']) if p.get('max_consec') else "none"
    print(f"  Risk: ${p['daily_loss']}/day, consec limit: {consec_str}")
    print(f"  Run: {result_dict['timestamp']}")

    for stage, label in [("in_sample", "In-sample"), ("oos", "OOS"), ("recent", "Recent")]:
        s = result_dict.get(stage, {})
        if not s:
            continue
        tpd = s.get("trades_per_day", 0)
        print(f"\n  {label}:")
        print(f"    Trades: {s.get('trades', 0)} ({tpd:.1f}/d)")
        print(f"    WR: {s.get('wr', 0):.1f}%")
        print(f"    P&L: ${s.get('pnl_per_day', 0):+.1f}/day")
        print(f"    Max DD: ${s.get('max_dd', 0):,.0f}")
