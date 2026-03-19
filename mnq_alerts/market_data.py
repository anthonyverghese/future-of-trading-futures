"""
market_data.py — Fetches real-time MNQ bars from Databento's REST API.

Uses CSV encoding to avoid the databento SDK's Rust extension (incompatible
with Python 3.8). Bars are cached in memory; each poll only requests new bars
since the last fetch to minimize API costs.
"""

from __future__ import annotations

import datetime
import io

import pandas as pd
import pytz
import requests

from config import DATABENTO_API_KEY, DATABENTO_DATASET, DATABENTO_SYMBOL

ET = pytz.timezone("America/New_York")

DATABENTO_API_URL = "https://hist.databento.com/v0/timeseries.get_range"

# Intraday bar cache — accumulates from 9:30 AM, reset each new session.
_bars_cache: pd.DataFrame = pd.DataFrame()


def _reset_cache_if_new_day() -> None:
    """Clear the cache when the calendar date changes."""
    global _bars_cache
    if not _bars_cache.empty:
        if _bars_cache.index[-1].date() != datetime.datetime.now(ET).date():
            _bars_cache = pd.DataFrame()
            print("[market_data] New trading day — bar cache reset.")


def get_todays_bars() -> pd.DataFrame:
    """
    Return all 1-minute OHLCV bars from 9:30 AM ET to now.
    Only fetches bars newer than the last cached bar on each call.
    """
    global _bars_cache

    _reset_cache_if_new_day()

    now_et = datetime.datetime.now(ET)

    if _bars_cache.empty:
        start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    else:
        start = _bars_cache.index[-1] + pd.Timedelta(minutes=1)

    if start >= now_et:
        return _bars_cache  # Nothing new yet

    new_bars = _fetch_bars(start, now_et)

    if not new_bars.empty:
        _bars_cache = (
            pd.concat([_bars_cache, new_bars])
            .pipe(lambda df: df[~df.index.duplicated(keep="last")])
            .sort_index()
        )

    return _bars_cache


def get_current_price(bars: pd.DataFrame | None = None) -> float | None:
    """Return the most recent Close price. Fetches bars if not provided."""
    if bars is None:
        bars = get_todays_bars()
    if bars.empty:
        return None
    return float(bars["Close"].iloc[-1])


def _fetch_bars(start: datetime.datetime, end: datetime.datetime) -> pd.DataFrame:
    """Request OHLCV bars from Databento for the given time range."""
    try:
        resp = requests.get(
            DATABENTO_API_URL,
            auth=(DATABENTO_API_KEY, ""),
            params={
                "dataset":   DATABENTO_DATASET,
                "symbols":   DATABENTO_SYMBOL,
                "schema":    "ohlcv-1m",
                "start":     start.isoformat(),
                "end":       end.isoformat(),
                "stype_in":  "continuous",
                "encoding":  "csv",
            },
            timeout=30,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        # Databento hist API has an ingestion lag; if our end time is ahead of
        # available data, retry using the available_end from the error response.
        if exc.response.status_code == 422:
            try:
                payload = exc.response.json().get("detail", {}).get("payload", {})
                available_end_str = payload.get("available_end")
                if available_end_str:
                    available_end = pd.Timestamp(available_end_str).to_pydatetime()
                    if available_end > start:
                        return _fetch_bars(start, available_end)
            except Exception:
                pass
        print(f"[market_data] HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        return pd.DataFrame()
    except requests.RequestException as exc:
        print(f"[market_data] Request failed: {exc}")
        return pd.DataFrame()

    if not resp.text.strip():
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        print(f"[market_data] CSV parse error: {exc}")
        return pd.DataFrame()

    if df.empty or "ts_event" not in df.columns:
        return pd.DataFrame()

    return _process_bars(df)


def _process_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw Databento CSV into app-ready format:
      - ts_event (nanoseconds) → Eastern DatetimeIndex
      - Prices (fixed-point int) → float  (divide by 1e9)
      - Column names → Title Case
    """
    df["ts_event"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    df = df.set_index("ts_event")

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000_000

    df = df.rename(columns={"open": "Open", "high": "High",
                             "low": "Low", "close": "Close", "volume": "Volume"})

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]

    df.index = df.index.tz_convert(ET)
    return df
