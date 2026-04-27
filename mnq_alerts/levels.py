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


def calculate_interior_fibs(ibh: float, ibl: float) -> dict[str, float]:
    """Return interior Fibonacci retracement levels within the IB range.

    331-day backtest: all levels show 74-76% WR at the 1pt entry.
    FIB_0.382 excluded (weakest at 70.3% WR).
    """
    ib_range = ibh - ibl
    return {
        "FIB_0.236": ibl + 0.236 * ib_range,
        "FIB_0.5": ibl + 0.5 * ib_range,
        "FIB_0.618": ibl + 0.618 * ib_range,
        "FIB_0.786": ibl + 0.786 * ib_range,
    }
