"""
market_data_ibkr.py — Real-time MNQ trade data via IBKR (IB Gateway / TWS).

Drop-in replacement for market_data.py (Databento). Provides the same
interface: trade_stream(), reset_session(), load_session_cache(),
get_session_trades(), get_current_price().

Uses ib_insync's reqTickByTickData("AllLast") for tick-level trade data.
Requires an active IB Gateway or TWS connection with market data
subscriptions for CME MNQ futures.

To switch from Databento to IBKR, change the import in main.py:
    # from market_data import ...
    from market_data_ibkr import (
        get_session_trades,
        load_session_cache,
        reset_session,
        trade_stream,
    )

Note: IBKR market data requires a separate subscription ($1/mo for
Level 1 CME Globex futures). The bot's existing IB Gateway connection
already has API access — this adds a market data subscription on top.

IMPORTANT: This module uses a SEPARATE IB connection (different clientId)
from the broker module. This avoids interference between market data
callbacks and order execution callbacks. The broker uses clientId from
config.py (IBKR_CLIENT_ID, default 1); this module uses clientId
IBKR_CLIENT_ID + 1.

Limitation: IBKR tick-by-tick data is live-only — no historical replay.
The session_start parameter is accepted for interface compatibility but
has no effect. On startup, main.py's session cache mechanism provides
trades up to the last cache save; ticks between the last save and now
are lost. This gap is typically <5 minutes.
"""

from __future__ import annotations

import datetime
import time
from typing import Generator

import pandas as pd
import pytz

from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID

ET = pytz.timezone("America/New_York")

# Use a separate clientId for market data to avoid interfering with
# the broker's order execution connection.
_FEED_CLIENT_ID = IBKR_CLIENT_ID + 1
_MNQ_SYMBOL = "MNQ"

# Session trade accumulator — same structure as market_data.py.
_prices: list[float] = []
_sizes: list[int] = []
_timestamps: list[pd.Timestamp] = []
_trades_cache: pd.DataFrame | None = None
_current_price: float | None = None

# Reconnect tracking.
_reconnect_count: int = 0


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {"Price": pd.Series(dtype=float), "Size": pd.Series(dtype=int)},
        index=pd.DatetimeIndex([]),
    )


def reset_session() -> None:
    """Clear accumulated trade data for a new session."""
    global _prices, _sizes, _timestamps, _trades_cache, _current_price
    global _reconnect_count
    _prices = []
    _sizes = []
    _timestamps = []
    _trades_cache = None
    _current_price = None
    _reconnect_count = 0
    print("[market_data] Session trades reset.")


def load_session_cache(trades: pd.DataFrame) -> None:
    """Pre-populate session trades from a cache snapshot."""
    global _prices, _sizes, _timestamps, _trades_cache, _current_price
    if trades.empty:
        return
    # Validate expected columns.
    if "Price" not in trades.columns or "Size" not in trades.columns:
        print(
            f"[market_data] WARNING: cache DataFrame has unexpected columns "
            f"{list(trades.columns)}, expected ['Price', 'Size']. Skipping."
        )
        return
    _prices = trades["Price"].tolist()
    _sizes = trades["Size"].tolist()
    _timestamps = list(trades.index)
    _trades_cache = None
    _current_price = _prices[-1]
    print(
        f"[market_data] Loaded {len(trades)} trades from cache "
        f"(up to {trades.index[-1].strftime('%H:%M:%S')} ET)."
    )


def get_session_trades() -> pd.DataFrame:
    """Return all trades accumulated so far this session."""
    global _trades_cache
    if _trades_cache is not None:
        return _trades_cache
    if not _prices:
        return _empty_trades()
    _trades_cache = pd.DataFrame(
        {"Price": _prices, "Size": _sizes},
        index=pd.DatetimeIndex(_timestamps),
    )
    return _trades_cache


def get_current_price() -> float | None:
    """Return the most recent trade price."""
    return _current_price


def _connect_ibkr():
    """Create a new IB connection for market data (separate from broker)."""
    from ib_insync import IB

    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=_FEED_CLIENT_ID, timeout=20)
    print(
        f"[market_data] Connected to IBKR at "
        f"{IBKR_HOST}:{IBKR_PORT} (clientId={_FEED_CLIENT_ID})"
    )
    return ib


def _resolve_mnq_contract(ib):
    """Qualify the continuous front-month MNQ contract."""
    from ib_insync import ContFuture, Future

    # ContFuture auto-resolves the front month. qualifyContracts fills
    # in the conId which we use to get the specific FUT contract
    # (required for tick data subscriptions).
    contfut = ContFuture(symbol=_MNQ_SYMBOL, exchange="CME", currency="USD")
    qualified = ib.qualifyContracts(contfut)
    if not qualified:
        raise RuntimeError(
            f"Failed to qualify ContFuture for {_MNQ_SYMBOL}"
        )

    fut = Future(conId=qualified[0].conId)
    qualified_fut = ib.qualifyContracts(fut)
    if not qualified_fut:
        raise RuntimeError(
            f"Failed to qualify FUT from conId {qualified[0].conId}"
        )

    contract = qualified_fut[0]
    print(
        f"[market_data] MNQ contract: {contract.localSymbol} "
        f"(conId={contract.conId})"
    )
    return contract


def trade_stream(
    session_start: datetime.datetime | None = None,
) -> Generator[tuple[float, int, datetime.datetime], None, None]:
    """
    Yield (price, size, timestamp_et) for each live MNQ trade.

    Uses IBKR's tick-by-tick "AllLast" data which provides every trade
    print with price, size, and exchange timestamp.

    session_start is accepted for interface compatibility with the
    Databento version but is not used — IBKR tick-by-tick data is
    live-only. The session cache mechanism in main.py handles
    replaying cached trades on startup.
    """
    global _current_price, _trades_cache, _reconnect_count

    if session_start is not None:
        print(
            "[market_data] Note: IBKR feed does not support historical "
            "replay. Session cache provides trades up to the last save. "
            "Live ticks start from now."
        )

    while True:
        ib = None
        ticker = None
        try:
            ib = _connect_ibkr()
            contract = _resolve_mnq_contract(ib)

            # Subscribe to tick-by-tick trade data.
            # "AllLast" includes all trade types (regular + irregular).
            ticker = ib.reqTickByTickData(contract, "AllLast")
            print("[market_data] Subscribed to tick-by-tick AllLast data")

            tick_count = 0
            last_stats_time = time.monotonic()
            skipped_ticks = 0

            while ib.isConnected():
                # Pump the event loop. 0.05s balances responsiveness
                # with CPU usage (~20 checks/sec).
                ib.sleep(0.05)

                # Periodic stats log (every 60s) for diagnosing feed issues.
                now_mono = time.monotonic()
                if now_mono - last_stats_time >= 60.0:
                    elapsed = now_mono - last_stats_time
                    rate = tick_count / elapsed if elapsed > 0 else 0
                    print(
                        f"[market_data] Stats: {tick_count} ticks in last "
                        f"{elapsed:.0f}s ({rate:.0f}/s), "
                        f"{len(_prices)} total session ticks, "
                        f"{skipped_ticks} skipped"
                        + (f", price={_current_price:.2f}" if _current_price else ""),
                        flush=True,
                    )
                    tick_count = 0
                    skipped_ticks = 0
                    last_stats_time = now_mono

                if not ticker.tickByTicks:
                    continue

                for tick in ticker.tickByTicks:
                    price = tick.price
                    size = tick.size

                    # Validate tick data.
                    if price is None or price <= 0:
                        skipped_ticks += 1
                        continue
                    if size is None or size <= 0:
                        skipped_ticks += 1
                        continue

                    # tick.time is a datetime. ib_insync returns it as
                    # UTC-aware in most configurations, but handle the
                    # naive case defensively.
                    tick_time = tick.time
                    if tick_time is None:
                        skipped_ticks += 1
                        continue
                    if tick_time.tzinfo is None:
                        tick_time = pytz.utc.localize(tick_time)
                    ts_et = tick_time.astimezone(ET)

                    # Accumulate into session lists.
                    # .as_unit('ns') ensures nanosecond resolution so
                    # DatetimeIndex.asi8 returns nanoseconds (matching
                    # how cache.save_trades/load_trades round-trips).
                    # Without this, pd.Timestamp(datetime) defaults to
                    # microsecond resolution and asi8 returns microseconds,
                    # breaking the cache.
                    ts_pd = pd.Timestamp(ts_et).as_unit("ns")
                    _prices.append(price)
                    _sizes.append(size)
                    _timestamps.append(ts_pd)
                    _trades_cache = None
                    _current_price = price
                    tick_count += 1

                    yield price, size, ts_et

                # Clear processed ticks so they don't re-process.
                ticker.tickByTicks.clear()

            # If we get here, ib.isConnected() returned False.
            print(
                "[market_data] IBKR connection lost "
                f"(had {len(_prices)} session ticks). Reconnecting..."
            )

        except Exception as exc:
            _reconnect_count += 1
            print(
                f"[market_data] IBKR feed error (reconnect #{_reconnect_count}): "
                f"{type(exc).__name__}: {exc}. Reconnecting in 5s..."
            )
            time.sleep(5)
        finally:
            # Cancel tick subscription before disconnecting.
            if ticker is not None and ib is not None:
                try:
                    ib.cancelTickByTickData(contract, "AllLast")
                except Exception:
                    pass
            if ib is not None:
                try:
                    ib.disconnect()
                    print("[market_data] Disconnected from IBKR")
                except Exception:
                    pass
