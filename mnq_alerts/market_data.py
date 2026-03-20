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
def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {"Price": pd.Series(dtype=float), "Size": pd.Series(dtype=int)},
        index=pd.DatetimeIndex([]),
    )

_trades: pd.DataFrame = _empty_trades()
_current_price: float | None = None
_live_client: db.Live | None = None


def reset_session() -> None:
    """Clear accumulated trade data for a new session. Call once per trading day."""
    global _trades, _current_price
    _trades = _empty_trades()
    _current_price = None
    print("[market_data] Session trades reset.")


def get_session_trades() -> pd.DataFrame:
    """Return all trades accumulated so far this session."""
    return _trades


def get_current_price() -> float | None:
    """Return the most recent trade price."""
    return _current_price


def _make_client(start: datetime.datetime | None = None) -> db.Live:
    client = db.Live(key=DATABENTO_API_KEY)
    kwargs: dict = {
        "dataset":  DATABENTO_DATASET,
        "schema":   "trades",
        "stype_in": "continuous",
        "symbols":  [DATABENTO_SYMBOL],
    }
    if start is not None:
        kwargs["start"] = start.isoformat()
    client.subscribe(**kwargs)
    return client


def trade_stream(session_start: datetime.datetime | None = None) -> Generator[tuple[float, int, datetime.datetime], None, None]:
    """
    Yield (price, size, timestamp_et) for each live trade record.
    Blocks until the next trade arrives. Reconnects automatically on errors.

    Pass session_start to replay historical trades from that timestamp before
    switching to live — use this when starting mid-session so VWAP and IB
    are accurate from the first live tick.

    Accumulates every yielded trade into the session DataFrame so callers can
    compute VWAP and IB levels from get_session_trades().
    """
    global _trades, _current_price, _live_client

    start = session_start  # used only on first connection; cleared after
    while True:
        print("[market_data] Connecting to Databento live feed"
              + (f" (replaying from {start.strftime('%H:%M:%S')} ET)..." if start else "..."))
        _live_client = _make_client(start=start)
        start = None  # reconnects after errors go to live-only
        try:
            for record in _live_client:
                if not isinstance(record, db.TradeMsg):
                    continue
                price = record.price / 1_000_000_000
                size = int(record.size)
                ts = pd.Timestamp(record.ts_event, unit="ns", tz="UTC").tz_convert(ET)

                new_row = pd.DataFrame({"Price": [price], "Size": [size]}, index=[ts])
                _trades = pd.concat([_trades, new_row])
                _current_price = price

                yield price, size, ts.to_pydatetime(warn=False)

        except Exception as exc:
            print(f"[market_data] Feed error: {exc}. Reconnecting...")
        finally:
            try:
                _live_client.stop()
            except Exception:
                pass
            _live_client = None
