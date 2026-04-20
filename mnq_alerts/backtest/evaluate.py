"""Trade outcome evaluation.

Wraps evaluate_bot_trade from bot_risk_backtest.py with per-level
target support and convenience functions.
"""

from __future__ import annotations

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bot_risk_backtest import evaluate_bot_trade, MULTIPLIER, FEE_PTS
from walk_forward import _eod_cutoff_ns


def evaluate(
    entry_idx: int,
    line_price: float,
    direction: str,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    target_pts: float,
    stop_pts: float,
    timeout_secs: int = 900,
    eod_ns: int | None = None,
) -> tuple[str, int, float]:
    """Evaluate a trade. Returns (outcome, exit_idx, pnl_usd)."""
    out, eidx, pnl_pts = evaluate_bot_trade(
        entry_idx, line_price, direction,
        ts_ns, prices, target_pts, stop_pts,
        timeout_secs, eod_ns,
    )
    return out, eidx, pnl_pts * MULTIPLIER


# Per-level optimal targets (from v5 backtest Stage 0, 2026-04-19).
PER_LEVEL_TARGETS = {
    "IBH": 14,
    "IBL": 10,
    "FIB_EXT_HI_1.272": 5,
    "FIB_EXT_LO_1.272": 8,
    "VWAP": 6,
}


def get_target(level: str, per_level: bool = False, default: float = 8.0) -> float:
    """Get target for a level. Uses per-level if requested."""
    if per_level:
        return float(PER_LEVEL_TARGETS.get(level, default))
    return default
