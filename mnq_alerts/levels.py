"""
levels.py — Calculates IBH/IBL and VWAP from real-time trade tick data.
"""

from __future__ import annotations

import pandas as pd


def calculate_initial_balance(trades: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    Return (IBH, IBL) — the highest and lowest trade prices during the
    9:30–10:30 AM ET window. Call only after 10:30 AM when IB is complete.
    """
    ib_trades = trades.between_time("09:30", "10:30", inclusive="left")
    if ib_trades.empty:
        return None, None
    return float(ib_trades["Price"].max()), float(ib_trades["Price"].min())


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
