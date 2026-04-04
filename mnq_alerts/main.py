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
import sys
import time

import pytz

from alert_manager import AlertManager
from config import (
    ALERT_THRESHOLD_POINTS,
    BOT_EOD_FLATTEN_BUFFER_MIN,
    DISPLAY_TZ,
    IB_END_HOUR,
    IB_END_MIN,
    IBKR_TRADING_ENABLED,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MIN,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MIN,
)
from levels import calculate_fib_levels
from cache import (
    CACHE_INTERVAL_SECONDS,
    clear_if_stale,
    get_daily_summary,
    get_replay_start,
    load_pending_alerts,
    load_recent_outcomes,
    load_trades,
    save_trades,
    upsert_daily_stats,
)
from market_data import (
    get_session_trades,
    load_session_cache,
    reset_session,
    trade_stream,
)
from notifications import send_notification
from outcome_tracker import OutcomeEvaluator

ET = pytz.timezone("America/New_York")
if DISPLAY_TZ:
    LOCAL_TZ = pytz.timezone(DISPLAY_TZ)
    LOCAL_TZ_NAME = datetime.datetime.now(pytz.timezone(DISPLAY_TZ)).strftime("%Z")
else:
    LOCAL_TZ = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    LOCAL_TZ_NAME = (
        datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Z")
    )

MARKET_OPEN = datetime.time(MARKET_OPEN_HOUR, MARKET_OPEN_MIN)
MARKET_CLOSE = datetime.time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN)
IB_END = datetime.time(IB_END_HOUR, IB_END_MIN)
# Flatten any open bot position this many minutes before MARKET_CLOSE.
_EOD_FLATTEN_TIME = (
    datetime.datetime.combine(datetime.date.today(), MARKET_CLOSE)
    - datetime.timedelta(minutes=BOT_EOD_FLATTEN_BUFFER_MIN)
).time()

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
    today_open = now.replace(
        hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0
    )
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
    print(
        f"  Threshold : ±{ALERT_THRESHOLD_POINTS} pts from IBH / IBL / VWAP / Fib levels"
    )
    market_open_local = (
        datetime.datetime.now(ET)
        .replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN)
        .astimezone(LOCAL_TZ)
    )
    market_close_local = (
        datetime.datetime.now(ET)
        .replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MIN)
        .astimezone(LOCAL_TZ)
    )
    print(
        f"  Hours     : {market_open_local.strftime('%I:%M %p')} – "
        f"{market_close_local.strftime('%I:%M %p')} {LOCAL_TZ_NAME}, weekdays only"
    )
    print("=" * 55)

    # Wait for RTH before opening the live connection.
    while True:
        now = datetime.datetime.now(ET)
        if is_market_open(now):
            break
        now_pt = now.astimezone(LOCAL_TZ)
        wait_secs = seconds_until_next_open(now)
        print(
            f"[{now_pt.strftime('%H:%M:%S')} {LOCAL_TZ_NAME}] Market closed. "
            f"Next open in ~{wait_secs / 60:.0f} min."
        )
        time.sleep(min(wait_secs, 300))

    # Clear stale cache from a previous session before loading.
    clear_if_stale()
    cached_trades = load_trades()
    session_start = get_replay_start(cached_trades)

    # If replaying from session open, the replay covers the full session —
    # don't seed from a partial cache or it will be double-counted.
    replaying_full_session = session_start.time() <= MARKET_OPEN
    use_cache = not cached_trades.empty and not replaying_full_session
    if not cached_trades.empty and replaying_full_session:
        print(
            "[cache] Partial cache — replaying full session (cache not used for seeding)."
        )
    if use_cache:
        load_session_cache(cached_trades)

    alert_manager = AlertManager()
    prior_outcomes = load_recent_outcomes()
    evaluator = OutcomeEvaluator(prior_outcomes)
    evaluator.restore(load_pending_alerts(datetime.datetime.now(ET).date().isoformat()))
    print(
        f"[streak] Loaded outcomes "
        f"(wins: {evaluator.consecutive_wins}, "
        f"losses: {evaluator.consecutive_losses})"
    )

    # Bot trader — separate zone tracking (1pt threshold) and order execution.
    # Disabled by default; set IBKR_TRADING_ENABLED=true in .env to activate.
    bot = None
    if IBKR_TRADING_ENABLED:
        from bot_trader import BotTrader

        bot = BotTrader()
        if not bot.connect():
            send_notification(
                "Bot Connection Failed",
                "Could not connect to IB Gateway. Bot will retry on first trade.",
            )

    ib_locked = False
    ibh: float | None = None
    ibl: float | None = None
    day_open: float | None = None
    last_session_date = datetime.datetime.now(ET).date() if use_cache else None
    last_status_ts = 0.0
    last_cache_ts = 0.0
    session_closed = False
    bot_eod_flattened = False

    # Incremental VWAP: O(1) per tick instead of scanning the full DataFrame.
    vwap_sum_pv = 0.0  # Σ(price × size)
    vwap_sum_vol = 0  # Σ(size)
    vwap: float | None = None

    # Incremental IB: track high/low during 9:30–10:30 without DataFrame scan.
    ib_high = -float("inf")
    ib_low = float("inf")
    ib_has_trades = False

    # Seed incremental accumulators from cached trades so VWAP/IB are
    # correct from the first live tick (no need to re-derive from replay).
    if use_cache:
        rth = cached_trades.between_time("09:30", "16:00", inclusive="left")
        if not rth.empty:
            vwap_sum_pv = float((rth["Price"] * rth["Size"]).sum())
            vwap_sum_vol = int(rth["Size"].sum())
            if vwap_sum_vol > 0:
                vwap = vwap_sum_pv / vwap_sum_vol
                alert_manager.update_levels(ibh=None, ibl=None, vwap=vwap)
            day_open = float(rth["Price"].iloc[0])
        ib = cached_trades.between_time("09:30", "10:30", inclusive="left")
        if not ib.empty:
            ib_high = float(ib["Price"].max())
            ib_low = float(ib["Price"].min())
            ib_has_trades = True
        # If cache contains trades past 10:30, IB is already locked.
        last_cached_time = cached_trades.index[-1].to_pydatetime(warn=False).time()
        if last_cached_time >= IB_END and ib_has_trades:
            ibh, ibl = ib_high, ib_low
            ib_locked = True
            alert_manager.update_levels(ibh=ibh, ibl=ibl, vwap=vwap)
            fib_levels = calculate_fib_levels(ibh, ibl)
            alert_manager.update_fib_levels(fib_levels)
            print(f"[cache] IB already locked — IBH: {ibh:.2f}, IBL: {ibl:.2f}")
            alert_manager.restore_zone_state()
        print(
            f"[cache] Seeded VWAP from {len(rth)} cached trades "
            f"(VWAP: {f'{vwap:.2f}' if vwap else 'N/A'})"
        )

    # Tick rate: count trades in a rolling window for live ticks only.
    tick_times: list[datetime.datetime] = []

    for price, size, ts_et in trade_stream(session_start=session_start):
        now = datetime.datetime.now(ET)
        now_pt = now.astimezone(LOCAL_TZ)
        today = now.date()

        # Pre-close EOD flatten: close any open bot position a few minutes
        # before 4pm so fills complete before the session ends (avoids
        # overnight margin on stalled positions). Blocks new bot trades too.
        if (
            not bot_eod_flattened
            and bot is not None
            and bot.is_connected
            and last_session_date == today
            and now.time() >= _EOD_FLATTEN_TIME
            and now.time() < MARKET_CLOSE
        ):
            bot.eod_flatten()
            bot_eod_flattened = True

        # Close session once market shuts — must check before the is_market_open
        # guard below, otherwise post-market trades hit `continue` and this
        # block is never reached.
        if (
            not session_closed
            and last_session_date == today
            and now.time() >= MARKET_CLOSE
        ):
            evaluator.close_session()
            session_closed = True

            summary = get_daily_summary(today.isoformat())
            total = sum(summary.values())
            if total > 0:
                wr = (
                    summary["correct"]
                    / (summary["correct"] + summary["incorrect"])
                    * 100
                    if (summary["correct"] + summary["incorrect"]) > 0
                    else 0
                )
                send_notification(
                    f"MNQ Daily Summary — {today.strftime('%m/%d')}",
                    f"{total} alerts today\n"
                    f"✓ {summary['correct']} correct\n"
                    f"✗ {summary['incorrect']} incorrect\n"
                    f"? {summary['inconclusive']} inconclusive\n"
                    f"Win rate: {wr:.0f}%",
                )

            # Bot daily summary notification + close.
            if bot is not None and bot.is_connected:
                send_notification(
                    f"Bot Daily Summary — {today.strftime('%m/%d')}",
                    bot.daily_summary,
                )
                bot.close_session()

            save_trades(get_session_trades())
            print(
                f"[{now_pt.strftime('%H:%M:%S')} {LOCAL_TZ_NAME}] "
                f"Market closed. Shutting down."
            )
            sys.exit(0)

        # Skip trades outside RTH — futures trade 24/5 but we only alert during RTH.
        if not is_market_open(now):
            continue

        # Reset session state each new trading day.
        if last_session_date != today:
            reset_session()
            alert_manager = AlertManager()
            evaluator = OutcomeEvaluator(load_recent_outcomes())
            ib_locked = False
            ibh = None
            ibl = None
            day_open = None
            last_session_date = today
            last_cache_ts = 0.0
            session_closed = False
            bot_eod_flattened = False
            if bot is not None:
                bot.reset_daily_state()
            vwap_sum_pv = 0.0
            vwap_sum_vol = 0
            vwap = None
            ib_high = -float("inf")
            ib_low = float("inf")
            ib_has_trades = False
            tick_times = []
            print(f"\n[{now_pt.strftime('%Y-%m-%d')}] New session — state reset.")

        # Record opening price for session context scoring.
        if day_open is None:
            day_open = price

        trade_time = ts_et.time()

        # Update incremental VWAP (9:30 AM – 4:00 PM ET).
        if MARKET_OPEN <= trade_time < MARKET_CLOSE:
            vwap_sum_pv += price * size
            vwap_sum_vol += size
            if vwap_sum_vol > 0:
                vwap = vwap_sum_pv / vwap_sum_vol
                alert_manager.update_levels(ibh=None, ibl=None, vwap=vwap)
                if bot is not None:
                    bot.update_levels(vwap=vwap)

        # Update incremental IB high/low (9:30–10:30 AM ET).
        if not ib_locked and MARKET_OPEN <= trade_time < IB_END:
            if price > ib_high:
                ib_high = price
            if price < ib_low:
                ib_low = price
            ib_has_trades = True

        # Lock in IBH/IBL once after 10:30 AM ET (fixed for the session).
        # Use the trade's own timestamp so replay doesn't lock IB prematurely.
        if trade_time >= IB_END and not ib_locked:
            if ib_has_trades:
                ibh, ibl = ib_high, ib_low
                alert_manager.update_levels(ibh=ibh, ibl=ibl, vwap=None)
                print(
                    f"[{now_pt.strftime('%H:%M:%S')} {LOCAL_TZ_NAME}] "
                    f"IB locked — IBH: {ibh:.2f}, IBL: {ibl:.2f}"
                )
                ib_locked = True
                if bot is not None:
                    bot.update_levels(ibh=ibh, ibl=ibl)
                upsert_daily_stats(today.isoformat(), ibh=ibh, ibl=ibl)
                fib_levels = calculate_fib_levels(ibh, ibl)
                alert_manager.update_fib_levels(fib_levels)
                if bot is not None:
                    bot.update_fib_levels(fib_levels)
                fib_str = ", ".join(f"{k}: {v:.2f}" for k, v in fib_levels.items())
                print(
                    f"[{now_pt.strftime('%H:%M:%S')} {LOCAL_TZ_NAME}] Fib levels: {fib_str}"
                )
            else:
                print(
                    f"[{now_pt.strftime('%H:%M:%S')} {LOCAL_TZ_NAME}] IB period done but no trade data yet."
                )

        # Pump ib_insync event loop so fill callbacks fire.
        if bot is not None:
            bot.process_events()

        if trade_time >= IB_END:
            # During replay ts_et lags wall time; only notify for live trades.
            trade_lag = (now - ts_et).total_seconds()
            if trade_lag < 60:
                # Compute tick rate from rolling window (O(1) amortized).
                tick_times.append(ts_et)
                window_start = ts_et - datetime.timedelta(minutes=3)
                # Trim old entries from the front.
                while tick_times and tick_times[0] < window_start:
                    tick_times.pop(0)
                tick_rate = len(tick_times) / 3.0

                session_move = price - day_open if day_open is not None else None
                fired, all_zone_entries = alert_manager.check_and_notify(
                    price,
                    now_et=trade_time,
                    tick_rate=tick_rate,
                    session_move_pts=session_move,
                    consecutive_wins=evaluator.consecutive_wins,
                    consecutive_losses=evaluator.consecutive_losses,
                    trade_ts=ts_et,
                )
                fired_levels = set()
                for alert_id, line_name, line_price, direction in fired:
                    evaluator.add(
                        alert_id, line_price, direction, ts_et, today.isoformat()
                    )
                    fired_levels.add((line_name, line_price, direction))
                # Bot checks its own zones (1pt entry) and submits orders.
                if bot is not None:
                    bot.on_tick(price)
                # Track suppressed zone entries for streak computation —
                # matches how the backtest computes streaks across ALL entries.
                for line_name, line_price, direction in all_zone_entries:
                    if (line_name, line_price, direction) not in fired_levels:
                        evaluator.add_untracked(line_price, direction, ts_et)
                evaluator.update(price, ts_et)
            else:
                alert_manager.advance_state(price)
                if bot is not None:
                    bot.advance_zones(price)
                # Evaluate pending outcomes during replay too — a restart
                # mid-evaluation needs the replayed trades to determine
                # whether the target/stop was hit while the app was down.
                evaluator.update(price, ts_et)

        # Save trade cache every CACHE_INTERVAL_SECONDS.
        now_ts = time.time()
        if now_ts - last_cache_ts >= CACHE_INTERVAL_SECONDS:
            save_trades(get_session_trades())
            last_cache_ts = now_ts

        # Throttle console output — one status line per STATUS_INTERVAL_SECONDS.
        if now_ts - last_status_ts >= _STATUS_INTERVAL_SECONDS:
            trade_lag = (now - ts_et).total_seconds()
            if trade_lag >= 60:
                print(
                    f"[replaying] {ts_et.strftime('%H:%M:%S')} ET "
                    f"({trade_lag / 60:.0f} min behind live) | "
                    f"VWAP: {f'{vwap:.2f}' if vwap else 'N/A'}"
                )
            else:
                if ib_locked:
                    ib_str = f"IBH: {ibh:.2f} | IBL: {ibl:.2f}"
                elif ib_has_trades:
                    ib_str = f"IB forming — H: {ib_high:.2f} | L: {ib_low:.2f}"
                else:
                    ib_str = "IB window active (no trades yet)"
                print(
                    f"[{now_pt.strftime('%H:%M:%S')} {LOCAL_TZ_NAME}] "
                    f"MNQ: {price:.2f} | "
                    f"VWAP: {f'{vwap:.2f}' if vwap else 'N/A'} | "
                    f"{ib_str}"
                )
            last_status_ts = now_ts


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nMNQ Alert System stopped.")
