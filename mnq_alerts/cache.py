"""
cache.py — Persists session trades to a local SQLite database.

Saves a snapshot every CACHE_INTERVAL_SECONDS so the app can resume
from the last checkpoint on restart instead of replaying from 9:30 AM.
"""

from __future__ import annotations

import datetime
import os
import sqlite3

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")

CACHE_PATH = os.path.join(os.path.dirname(__file__), ".session_cache.db")
CACHE_INTERVAL_SECONDS = 300  # save every 5 minutes


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {"Price": pd.Series(dtype=float), "Size": pd.Series(dtype=int)},
        index=pd.DatetimeIndex([]),
    )


def clear_if_stale() -> None:
    """Delete the cache file if it contains no data for today (leftover from a prior session)."""
    if not os.path.exists(CACHE_PATH):
        return
    today = datetime.datetime.now(ET).date().isoformat()
    try:
        with sqlite3.connect(CACHE_PATH) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE date = ?", (today,)
            ).fetchone()[0]
        if count == 0:
            os.remove(CACHE_PATH)
            print("[cache] Stale cache from previous session cleared.")
    except Exception:
        pass


def save_trades(trades: pd.DataFrame) -> None:
    """Persist today's trades to the local cache database."""
    if trades.empty:
        return
    today = datetime.datetime.now(ET).date().isoformat()

    df = trades.copy()
    df.index = df.index.asi8  # nanoseconds since epoch (UTC)
    df = df.reset_index().rename(columns={"index": "ts_ns", "Price": "price", "Size": "size"})
    df["date"] = today

    with sqlite3.connect(CACHE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades
            (date TEXT, ts_ns INTEGER, price REAL, size INTEGER)
        """)
        conn.execute("DELETE FROM trades WHERE date = ?", (today,))
        df[["date", "ts_ns", "price", "size"]].to_sql(
            "trades", conn, if_exists="append", index=False
        )
    print(f"[cache] Saved {len(trades)} trades.")


def load_trades() -> pd.DataFrame:
    """Load today's cached trades. Returns empty DataFrame if none exist."""
    if not os.path.exists(CACHE_PATH):
        return _empty_trades()
    today = datetime.datetime.now(ET).date().isoformat()
    try:
        with sqlite3.connect(CACHE_PATH) as conn:
            df = pd.read_sql(
                "SELECT ts_ns, price, size FROM trades WHERE date = ? ORDER BY ts_ns",
                conn, params=(today,),
            )
    except Exception:
        return _empty_trades()

    if df.empty:
        return _empty_trades()

    ts_index = pd.to_datetime(df["ts_ns"], unit="ns", utc=True).dt.tz_convert(ET)
    return pd.DataFrame(
        {"Price": df["price"].values, "Size": df["size"].astype(int).values},
        index=ts_index,
    )


def get_replay_start(cached_trades: pd.DataFrame) -> datetime.datetime:
    """
    Return the timestamp to start replaying from.
    If cached trades exist, replay only from 1 second past the last checkpoint
    (the cached trades already cover everything before that point).
    Otherwise replay from 9:30 AM ET today.
    """
    now_et = datetime.datetime.now(ET)
    session_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    if cached_trades.empty:
        return session_open

    last_ts = cached_trades.index[-1].to_pydatetime()
    return last_ts + datetime.timedelta(seconds=1)
