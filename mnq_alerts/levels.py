"""
levels.py — Calculates IBH/IBL and VWAP from intraday bar data.
"""

from __future__ import annotations

import pandas as pd


def calculate_initial_balance(bars: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    Return (IBH, IBL) — the high and low of the 9:30–10:30 AM RTH window.
    Call only after 10:30 AM when the IB period is complete.
    """
    ib_bars = bars.between_time("09:30", "10:29")
    if ib_bars.empty:
        return None, None
    return float(ib_bars["High"].max()), float(ib_bars["Low"].min())


def calculate_vwap(bars: pd.DataFrame) -> float | None:
    """
    Return the session VWAP from 9:30 AM ET to the most recent bar.
    Formula: VWAP = Σ(Typical Price × Volume) / Σ(Volume)
    Typical Price = (High + Low + Close) / 3
    """
    session_bars = bars.between_time("09:30", "15:59")
    if session_bars.empty or session_bars["Volume"].sum() == 0:
        return None

    typical_price = (session_bars["High"] + session_bars["Low"] + session_bars["Close"]) / 3
    vwap = (typical_price * session_bars["Volume"]).cumsum() / session_bars["Volume"].cumsum()
    return float(vwap.iloc[-1])
