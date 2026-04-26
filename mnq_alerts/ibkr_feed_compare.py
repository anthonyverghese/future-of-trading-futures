"""
ibkr_feed_compare.py — Compare IBKR tick data against Databento in real time.

Runs the IBKR feed in a background thread alongside the active Databento feed.
Logs periodic comparisons of tick count, VWAP, price, and latency so we can
verify IBKR data quality before switching.

Usage: import and call start_comparison() from main.py after market open.
Call stop_comparison() at session close. Logs appear as [ibkr-compare].

Does NOT interfere with the active Databento feed or trading logic.
"""

from __future__ import annotations

import datetime
import threading
import time

import pandas as pd
import pytz

from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID

ET = pytz.timezone("America/New_York")

_FEED_CLIENT_ID = IBKR_CLIENT_ID + 1
_MNQ_SYMBOL = "MNQ"

# IBKR shadow accumulator (separate from the active Databento accumulator).
_ibkr_prices: list[float] = []
_ibkr_sizes: list[int] = []
_ibkr_timestamps: list[datetime.datetime] = []
_ibkr_tick_count: int = 0
_ibkr_skipped: int = 0
_ibkr_last_price: float | None = None

# Control.
_thread: threading.Thread | None = None
_stop_event = threading.Event()


def start_comparison() -> None:
    """Start the IBKR shadow feed in a background thread."""
    global _thread
    if _thread is not None and _thread.is_alive():
        print("[ibkr-compare] Already running")
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_run_ibkr_feed, daemon=True)
    _thread.start()
    print("[ibkr-compare] Started IBKR shadow feed for comparison")


def stop_comparison() -> None:
    """Stop the IBKR shadow feed."""
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=10)
    print(
        f"[ibkr-compare] Stopped. Total IBKR ticks: {_ibkr_tick_count}, "
        f"skipped: {_ibkr_skipped}"
    )


def get_ibkr_stats() -> dict:
    """Return current IBKR feed stats for comparison logging."""
    vwap = None
    if _ibkr_prices and _ibkr_sizes:
        total_pv = sum(p * s for p, s in zip(_ibkr_prices, _ibkr_sizes))
        total_vol = sum(_ibkr_sizes)
        if total_vol > 0:
            vwap = total_pv / total_vol
    return {
        "tick_count": _ibkr_tick_count,
        "skipped": _ibkr_skipped,
        "last_price": _ibkr_last_price,
        "vwap": vwap,
        "total_trades": len(_ibkr_prices),
    }


def log_comparison(databento_trades: pd.DataFrame) -> None:
    """Log a comparison between Databento and IBKR feeds.

    Call periodically from main.py (e.g., every 5 minutes or on each
    status log).
    """
    ibkr = get_ibkr_stats()
    db_count = len(databento_trades) if not databento_trades.empty else 0

    # Databento VWAP.
    db_vwap = None
    if not databento_trades.empty:
        pv = (databento_trades["Price"] * databento_trades["Size"]).sum()
        vol = databento_trades["Size"].sum()
        if vol > 0:
            db_vwap = pv / vol

    # Tick count difference.
    count_diff = ibkr["total_trades"] - db_count
    count_pct = (
        abs(count_diff) / max(db_count, 1) * 100
    )

    # VWAP difference.
    vwap_diff = None
    if db_vwap is not None and ibkr["vwap"] is not None:
        vwap_diff = ibkr["vwap"] - db_vwap

    # Price difference.
    price_diff = None
    db_last = None
    if not databento_trades.empty:
        db_last = float(databento_trades["Price"].iloc[-1])
    if db_last is not None and ibkr["last_price"] is not None:
        price_diff = ibkr["last_price"] - db_last

    ibkr_vwap = ibkr["vwap"]
    ibkr_last = ibkr["last_price"]
    ibkr_total = ibkr["total_trades"]

    db_vwap_s = f"{db_vwap:.2f}" if db_vwap else "N/A"
    ibkr_vwap_s = f"{ibkr_vwap:.2f}" if ibkr_vwap else "N/A"
    vwap_diff_s = f"{vwap_diff:+.2f}" if vwap_diff is not None else "N/A"
    db_last_s = f"{db_last:.2f}" if db_last else "N/A"
    ibkr_last_s = f"{ibkr_last:.2f}" if ibkr_last else "N/A"
    price_diff_s = f"{price_diff:+.2f}" if price_diff is not None else "N/A"

    print(
        f"[ibkr-compare] "
        f"Ticks: DB={db_count:,} IBKR={ibkr_total:,} "
        f"diff={count_diff:+,} ({count_pct:.1f}%) | "
        f"VWAP: DB={db_vwap_s} IBKR={ibkr_vwap_s} diff={vwap_diff_s} | "
        f"Last: DB={db_last_s} IBKR={ibkr_last_s} diff={price_diff_s} | "
        f"IBKR skipped: {ibkr['skipped']}",
        flush=True,
    )


def _run_ibkr_feed() -> None:
    """Background thread: connect to IBKR and accumulate ticks."""
    global _ibkr_tick_count, _ibkr_skipped, _ibkr_last_price

    from ib_insync import IB, ContFuture, Future

    while not _stop_event.is_set():
        ib = None
        try:
            print(
                f"[ibkr-compare] Connecting to IBKR at "
                f"{IBKR_HOST}:{IBKR_PORT} (clientId={_FEED_CLIENT_ID})..."
            )
            ib = IB()
            ib.connect(
                IBKR_HOST, IBKR_PORT,
                clientId=_FEED_CLIENT_ID, timeout=20,
            )
            print("[ibkr-compare] Connected to IBKR")

            # Qualify contract.
            print(f"[ibkr-compare] Qualifying {_MNQ_SYMBOL}...")
            contfut = ContFuture(
                symbol=_MNQ_SYMBOL, exchange="CME", currency="USD",
            )
            qualified = ib.qualifyContracts(contfut)
            if not qualified:
                print("[ibkr-compare] ERROR: Failed to qualify ContFuture")
                time.sleep(30)
                continue
            fut = Future(conId=qualified[0].conId)
            qualified_fut = ib.qualifyContracts(fut)
            if not qualified_fut:
                print("[ibkr-compare] ERROR: Failed to qualify FUT")
                time.sleep(30)
                continue
            contract = qualified_fut[0]
            print(
                f"[ibkr-compare] Contract: {contract.localSymbol} "
                f"(conId={contract.conId})"
            )

            # Subscribe.
            ticker = ib.reqTickByTickData(contract, "AllLast")
            print("[ibkr-compare] Subscribed to AllLast ticks")

            first_tick = True
            last_log_time = time.monotonic()
            interval_ticks = 0

            while ib.isConnected() and not _stop_event.is_set():
                ib.sleep(0.05)

                if not ticker.tickByTicks:
                    # Log if no ticks after 60s.
                    if (
                        time.monotonic() - last_log_time >= 60.0
                        and _ibkr_tick_count == 0
                    ):
                        print(
                            "[ibkr-compare] WARNING: no ticks received "
                            "after 60s — check CME subscription"
                        )
                        last_log_time = time.monotonic()
                    continue

                batch = list(ticker.tickByTicks)
                ticker.tickByTicks.clear()

                for tick in batch:
                    price = tick.price
                    size = tick.size
                    if (
                        price is None or price <= 0
                        or size is None or size <= 0
                    ):
                        _ibkr_skipped += 1
                        continue

                    tick_time = tick.time
                    if tick_time is None:
                        _ibkr_skipped += 1
                        continue

                    if first_tick:
                        if tick_time.tzinfo is None:
                            tick_time_utc = pytz.utc.localize(tick_time)
                        else:
                            tick_time_utc = tick_time.astimezone(pytz.utc)
                        lag = (
                            datetime.datetime.now(pytz.utc) - tick_time_utc
                        ).total_seconds()
                        print(
                            f"[ibkr-compare] First tick: "
                            f"price={price:.2f}, size={size}, "
                            f"lag={lag:.1f}s"
                        )
                        first_tick = False

                    _ibkr_prices.append(price)
                    _ibkr_sizes.append(size)
                    _ibkr_last_price = price
                    _ibkr_tick_count += 1
                    interval_ticks += 1

                # Periodic stats (every 5 min).
                now_mono = time.monotonic()
                if now_mono - last_log_time >= 300.0:
                    elapsed = now_mono - last_log_time
                    rate = interval_ticks / elapsed if elapsed > 0 else 0
                    print(
                        f"[ibkr-compare] IBKR stats: {interval_ticks} ticks "
                        f"in {elapsed:.0f}s ({rate:.0f}/s), "
                        f"{_ibkr_tick_count} total, "
                        f"{_ibkr_skipped} skipped"
                        + (
                            f", price={_ibkr_last_price:.2f}"
                            if _ibkr_last_price
                            else ""
                        ),
                        flush=True,
                    )
                    interval_ticks = 0
                    last_log_time = now_mono

            if not _stop_event.is_set():
                print("[ibkr-compare] IBKR connection lost. Reconnecting...")

        except Exception as exc:
            if not _stop_event.is_set():
                print(
                    f"[ibkr-compare] Error: {type(exc).__name__}: {exc}. "
                    f"Retrying in 30s..."
                )
                # Wait with periodic stop checks.
                for _ in range(60):
                    if _stop_event.is_set():
                        break
                    time.sleep(0.5)
        finally:
            if ib is not None:
                try:
                    ib.cancelTickByTickData(contract, "AllLast")
                except Exception:
                    pass
                try:
                    ib.disconnect()
                except Exception:
                    pass
