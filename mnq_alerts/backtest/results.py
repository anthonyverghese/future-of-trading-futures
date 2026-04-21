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

    # Filters
    include_vwap: bool
    vol_filter_pct: float | None  # 0.15 or None (disabled)
    max_per_level: int

    # Trade management
    timeout_secs: int           # 900 default
    time_exit_secs: int | None  # None = disabled, 60 = cut losers after 60s
    trailing_stop: dict | None  # None or {"activate": 5, "offset": 3}

    # Risk limits
    daily_loss: float
    max_consec: int

    # Data
    data_days: int
    data_range: str             # "2025-01-02 to 2026-04-15"

    # Walk-forward
    train_days: int
    step_days: int

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


def compute_stats(trades: list[TradeRecord], n_days: int) -> dict:
    """Compute summary stats from a list of trades."""
    if not trades:
        return {"trades": 0, "trades_per_day": 0, "wr": 0,
                "pnl_per_day": 0, "max_dd": 0}
    w = sum(1 for t in trades if t.outcome == "win")
    l = sum(1 for t in trades if t.outcome == "loss")
    d = w + l
    wr = w / d * 100 if d else 0
    pnl = sum(t.pnl_usd for t in trades)
    eq = 10000.0
    peak = eq
    dd = 0.0
    for t in trades:
        eq += t.pnl_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {
        "trades": len(trades),
        "trades_per_day": len(trades) / n_days if n_days else 0,
        "wr": round(wr, 1),
        "pnl_per_day": round(pnl / n_days, 1) if n_days else 0,
        "max_dd": round(dd, 0),
    }


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
    print(f"  Risk: ${p['daily_loss']}/day, {p['max_consec']} consec")
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
