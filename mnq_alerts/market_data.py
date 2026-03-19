"""
market_data.py — Real-time MNQ trade data via Databento Live API (GLBX.MDP3 MDP 3.0).

Connects to the Databento live feed and yields TradeMsg records as they arrive.
Accumulates session trades for VWAP and IB calculations. Call reset_session()
at the start of each new trading day.
"""

from __future__ import annotations

import datetime
from typing import Generator

import databento as db
import pandas as pd
import pytz

from config import DATABENTO_API_KEY, DATABENTO_DATASET, DATABENTO_SYMBOL

ET = pytz.timezone("America/New_York")

# Session trade accumulator — reset each new trading day via reset_session().
_trades: pd.DataFrame = pd.DataFrame(columns=["Price", "Size"])
_current_price: float | None = None
_live_client: db.Live | None = None


def reset_session() -> None:
    """Clear accumulated trade data for a new session. Call once per trading day."""
    global _trades, _current_price
    _trades = pd.DataFrame(columns=["Price", "Size"])
    _current_price = None
    print("[market_data] Session trades reset.")


def get_session_trades() -> pd.DataFrame:
    """Return all trades accumulated so far this session."""
    return _trades


def get_current_price() -> float | None:
    """Return the most recent trade price."""
    return _current_price


def _make_client() -> db.Live:
    client = db.Live(key=DATABENTO_API_KEY)
    client.subscribe(
        dataset=DATABENTO_DATASET,
        schema="trades",
        stype_in="continuous",
        symbols=[DATABENTO_SYMBOL],
    )
    return client


def trade_stream() -> Generator[tuple[float, int, datetime.datetime], None, None]:
    """
    Yield (price, size, timestamp_et) for each live trade record.
    Blocks until the next trade arrives. Reconnects automatically on errors.

    Accumulates every yielded trade into the session DataFrame so callers can
    compute VWAP and IB levels from get_session_trades().
    """
    global _trades, _current_price, _live_client

    while True:
        print("[market_data] Connecting to Databento live feed...")
        _live_client = _make_client()
        try:
            for record in _live_client:
                price = record.price / 1_000_000_000
                size = int(record.size)
                ts = pd.Timestamp(record.ts_event, unit="ns", tz="UTC").tz_convert(ET)

                new_row = pd.DataFrame({"Price": [price], "Size": [size]}, index=[ts])
                _trades = pd.concat([_trades, new_row])
                _current_price = price

                yield price, size, ts.to_pydatetime()

        except Exception as exc:
            print(f"[market_data] Feed error: {exc}. Reconnecting...")
        finally:
            try:
                _live_client.stop()
            except Exception:
                pass
            _live_client = None
