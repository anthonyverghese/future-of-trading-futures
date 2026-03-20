"""
main.py — Entry point. Run with: python main.py

Stream behavior:
  1. Wait for RTH (9:30 AM – 4:00 PM ET, weekdays only)
  2. Connect to Databento Live feed (GLBX.MDP3 MDP 3.0, trades schema)
  3. On each trade tick: recalculate VWAP, check alert levels
  4. At 10:30 AM: lock in IBH and IBL from trade data
  5. State resets automatically each new trading day
"""

from __future__ import annotations

import datetime
import time

import pytz

from alert_manager import AlertManager
from config import (
    ALERT_THRESHOLD_POINTS,
    IB_END_HOUR, IB_END_MIN,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
    MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
)
from levels import calculate_initial_balance, calculate_vwap
from market_data import get_session_trades, reset_session, trade_stream

ET = pytz.timezone("America/New_York")
PT = pytz.timezone("America/Los_Angeles")

MARKET_OPEN  = datetime.time(MARKET_OPEN_HOUR,  MARKET_OPEN_MIN)
MARKET_CLOSE = datetime.time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)
IB_END       = datetime.time(IB_END_HOUR,       IB_END_MIN)

# Throttle console status output to avoid flooding the terminal.
_STATUS_INTERVAL_SECONDS = 5


def is_market_open(now: datetime.datetime) -> bool:
    """Return True if within RTH hours on a weekday."""
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < MARKET_CLOSE


def ib_period_complete(now: datetime.datetime) -> bool:
    """Return True after the 9:30–10:30 AM Initial Balance window has closed."""
    return now.time() >= IB_END


def seconds_until_next_open(now: datetime.datetime) -> float:
    """Return seconds until the next RTH open, accounting for weekends."""
    today_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN,
                             second=0, microsecond=0)
    if now < today_open and now.weekday() < 5:
        return (today_open - now).total_seconds()

    days_ahead = 3 if now.weekday() == 4 else 2 if now.weekday() == 5 else 1
    next_open = (now + datetime.timedelta(days=days_ahead)).replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0
    )
    return (next_open - now).total_seconds()


def run() -> None:
    """Main event-driven loop. Runs until interrupted (Ctrl+C)."""
    print("=" * 55)
    print("  MNQ Alert System — Live Feed (GLBX.MDP3 MDP 3.0)")
    print(f"  Threshold : ±{ALERT_THRESHOLD_POINTS} pts from IBH / IBL / VWAP")
    print(f"  Hours     : 6:30 AM – 1:00 PM PT, weekdays only")
    print("=" * 55)

    # Wait for RTH before opening the live connection.
    while True:
        now = datetime.datetime.now(ET)
        if is_market_open(now):
            break
        now_pt = now.astimezone(PT)
        wait_secs = seconds_until_next_open(now)
        print(f"[{now_pt.strftime('%H:%M:%S')} PT] Market closed. "
              f"Next open in ~{wait_secs / 60:.0f} min.")
        time.sleep(min(wait_secs, 300))

    alert_manager     = AlertManager()
    ib_locked         = False
    ibh: float | None = None
    ibl: float | None = None
    last_session_date = None
    last_status_ts    = 0.0

    # If starting mid-session, replay trades from 9:30 AM so VWAP and IB
    # are accurate from the first live tick rather than starting from scratch.
    now_et = datetime.datetime.now(ET)
    session_start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    for price, size, ts_et in trade_stream(session_start=session_start):
        now    = datetime.datetime.now(ET)
        now_pt = now.astimezone(PT)
        today  = now.date()

        # Skip trades outside RTH — futures trade 24/5 but we only alert during RTH.
        if not is_market_open(now):
            continue

        # Reset session state each new trading day.
        if last_session_date != today:
            reset_session()
            alert_manager     = AlertManager()
            ib_locked         = False
            ibh               = None
            ibl               = None
            last_session_date = today
            print(f"\n[{now_pt.strftime('%Y-%m-%d')}] New session — state reset.")

        trades = get_session_trades()

        # Lock in IBH/IBL once after 10:30 AM ET (fixed for the session).
        # Use the trade's own timestamp so replay doesn't lock IB prematurely.
        if ts_et.time() >= IB_END and not ib_locked:
            ibh, ibl = calculate_initial_balance(trades)
            if ibh is not None and ibl is not None:
                alert_manager.update_levels(ibh=ibh, ibl=ibl, vwap=None)
                print(f"[{now_pt.strftime('%H:%M:%S')} PT] "
                      f"IB locked — IBH: {ibh:.2f}, IBL: {ibl:.2f}")
                ib_locked = True
            else:
                print(f"[{now_pt.strftime('%H:%M:%S')} PT] IB period done but no trade data yet.")

        # Recalculate VWAP on every trade tick for real-time accuracy.
        vwap = calculate_vwap(trades)
        if vwap is not None:
            alert_manager.update_levels(ibh=None, ibl=None, vwap=vwap)

        if ts_et.time() >= IB_END:
            # During replay ts_et lags wall time; only notify for live trades.
            trade_lag = (now - ts_et).total_seconds()
            if trade_lag < 60:
                alert_manager.check_and_notify(price)
            else:
                alert_manager.advance_state(price)

        # Throttle console output — one status line per STATUS_INTERVAL_SECONDS.
        # During replay, show progress instead of stale historical prices.
        now_ts = time.time()
        if now_ts - last_status_ts >= _STATUS_INTERVAL_SECONDS:
            trade_lag = (now - ts_et).total_seconds()
            if trade_lag >= 60:
                print(f"[replaying] {ts_et.strftime('%H:%M:%S')} ET "
                      f"({trade_lag / 60:.0f} min behind live)...")
            else:
                ib_str = (f"IBH: {ibh:.2f} | IBL: {ibl:.2f}" if ib_locked
                          else "IB window active")
                print(f"[{now_pt.strftime('%H:%M:%S')} PT] "
                      f"MNQ: {price:.2f} | "
                      f"VWAP: {f'{vwap:.2f}' if vwap else 'N/A'} | "
                      f"{ib_str}")
            last_status_ts = now_ts


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nMNQ Alert System stopped.")
