"""
levels.py — Price level calculations derived from the Initial Balance range.
"""

from __future__ import annotations


def calculate_fib_levels(ibh: float, ibl: float) -> dict[str, float]:
    """
    Return Fibonacci extension levels derived from the IB range.

    180-day backtest results (all beat 73.8% baseline):
      FIB_EXT_LO_1.272:  78.5% win rate, 535 trades, EV +2.0
      FIB_EXT_HI_1.272:  77.8% win rate, 414 trades, EV +1.8
    """
    ib_range = ibh - ibl
    return {
        "FIB_EXT_LO_1.272": ibl - 0.272 * ib_range,
        "FIB_EXT_HI_1.272": ibh + 0.272 * ib_range,
    }
