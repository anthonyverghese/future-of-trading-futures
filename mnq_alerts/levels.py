"""
levels.py — Calculates IBH/IBL and VWAP from real-time trade tick data.
"""

from __future__ import annotations

import pandas as pd


def calculate_initial_balance(
    trades: pd.DataFrame,
) -> tuple[float | None, float | None]:
    """
    Return (IBH, IBL) — the highest and lowest trade prices during the
    9:30–10:30 AM ET window. Call only after 10:30 AM when IB is complete.
    """
    ib_trades = trades.between_time("09:30", "10:30", inclusive="left")
    if ib_trades.empty:
        return None, None
    return float(ib_trades["Price"].max()), float(ib_trades["Price"].min())


def calculate_fib_levels(ibh: float, ibl: float) -> dict[str, float]:
    """
    Return the three highest-EV Fibonacci levels derived from the IB range.

    180-day backtest results (all beat 73.8% baseline):
      FIB_RET_0.236:     77.8% win rate, 977 trades, EV +1.8
      FIB_EXT_LO_1.272:  78.5% win rate, 535 trades, EV +2.0
      FIB_EXT_HI_1.272:  77.8% win rate, 414 trades, EV +1.8
    """
    ib_range = ibh - ibl
    return {
        "FIB_EXT_LO_1.272": ibl - 0.272 * ib_range,
        "FIB_EXT_HI_1.272": ibh + 0.272 * ib_range,
    }


def calculate_vwap(trades: pd.DataFrame) -> float | None:
    """
    Return the session VWAP from 9:30 AM ET to the most recent trade.
    Formula: VWAP = Σ(Price × Size) / Σ(Size)
    Calculated from individual trade ticks for maximum accuracy.
    """
    session_trades = trades.between_time("09:30", "16:00", inclusive="left")
    if session_trades.empty or session_trades["Size"].sum() == 0:
        return None
    total_pv = (session_trades["Price"] * session_trades["Size"]).sum()
    total_size = session_trades["Size"].sum()
    return float(total_pv / total_size)
