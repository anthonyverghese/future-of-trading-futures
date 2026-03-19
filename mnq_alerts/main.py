"""
main.py — Entry point. Run with: python main.py

Loop behavior:
  1. Sleep until RTH (9:30 AM – 4:00 PM ET, weekdays only)
  2. Every 30s: fetch MNQ bars, recalculate VWAP, check alert levels
  3. At 10:30 AM: lock in IBH and IBL for the session
  4. State resets automatically each new trading day
"""

from __future__ import annotations

import datetime
import time

import pytz

from alert_manager import AlertManager
from config import (
    ALERT_THRESHOLD_POINTS,
    CHECK_INTERVAL_SECONDS,
    IB_END_HOUR, IB_END_MIN,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN,
    MARKET_OPEN_HOUR, MARKET_OPEN_MIN,
)
from levels import calculate_initial_balance, calculate_vwap
from market_data import get_current_price, get_todays_bars

ET = pytz.timezone("America/New_York")

MARKET_OPEN  = datetime.time(MARKET_OPEN_HOUR,  MARKET_OPEN_MIN)
MARKET_CLOSE = datetime.time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)
IB_END       = datetime.time(IB_END_HOUR,       IB_END_MIN)


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
    """Main polling loop. Runs until interrupted (Ctrl+C)."""
    print("=" * 55)
    print("  MNQ Alert System")
    print(f"  Threshold : ±{ALERT_THRESHOLD_POINTS} pts from IBH / IBL / VWAP")
    print(f"  Interval  : every {CHECK_INTERVAL_SECONDS}s during RTH")
    print(f"  Hours     : 9:30 AM – 4:00 PM ET, weekdays only")
    print("=" * 55)

    alert_manager     = AlertManager()
    ib_locked         = False
    last_session_date = None

    while True:
        now   = datetime.datetime.now(ET)
        today = now.date()

        # Reset session state each new trading day
        if last_session_date != today:
            alert_manager     = AlertManager()
            ib_locked         = False
            last_session_date = today
            print(f"\n[{now.strftime('%Y-%m-%d')}] New session — state reset.")

        if not is_market_open(now):
            wait_secs = seconds_until_next_open(now)
            print(f"[{now.strftime('%H:%M:%S')} ET] Market closed. "
                  f"Next open in ~{wait_secs / 60:.0f} min.")
            time.sleep(min(wait_secs, 300))
            continue

        bars          = get_todays_bars()
        current_price = get_current_price(bars)

        if current_price is None:
            print(f"[{now.strftime('%H:%M:%S')} ET] No price data yet — retrying...")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        # Lock in IBH/IBL once after 10:30 AM (fixed for the session)
        if ib_period_complete(now) and not ib_locked:
            ibh, ibl = calculate_initial_balance(bars)
            if ibh is not None and ibl is not None:
                alert_manager.update_levels(ibh=ibh, ibl=ibl, vwap=None)
                print(f"[{now.strftime('%H:%M:%S')} ET] "
                      f"IB locked — IBH: {ibh:.2f}, IBL: {ibl:.2f}")
                ib_locked = True
            else:
                print(f"[{now.strftime('%H:%M:%S')} ET] IB period done but no bar data yet.")

        # VWAP drifts throughout the session — recalculate every tick
        vwap = calculate_vwap(bars)
        if vwap is not None:
            alert_manager.update_levels(ibh=None, ibl=None, vwap=vwap)

        if ib_period_complete(now):
            alert_manager.check_and_notify(current_price)

        ib_status = "locked" if ib_locked else "IB window active"
        print(f"[{now.strftime('%H:%M:%S')} ET] "
              f"MNQ: {current_price:.2f} | "
              f"VWAP: {f'{vwap:.2f}' if vwap else 'N/A'} | "
              f"IB: {ib_status}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nMNQ Alert System stopped.")
