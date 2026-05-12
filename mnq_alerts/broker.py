"""
broker.py — IBKR order execution with risk management.

Submits bracket orders (market entry + limit target + stop loss) when alerts fire.
Disabled by default — set IBKR_TRADING_ENABLED=true in .env to activate.

Risk controls:
  - One position at a time (no stacking)
  - Daily loss limit ($200 — stops trading for the day)
  - 15-min position timeout (close at market if no target/stop hit)

Connection: IB Gateway or TWS must be running with API access enabled.
Default port 4002 (IB Gateway paper), 4001 (IB Gateway live).
"""

from __future__ import annotations

import datetime
import threading
import time
from dataclasses import dataclass

from config import (
    BOT_FILL_TIMEOUT_SECS,
    BOT_TIMEOUT_SECS,
    DAILY_LOSS_LIMIT_USD,
    IBKR_ACCOUNT,
    IBKR_CLIENT_ID,
    IBKR_HOST,
    IBKR_PORT,
    IBKR_TRADING_ENABLED,
)

# ib_insync is only imported when trading is enabled to avoid hard dependency.
if IBKR_TRADING_ENABLED:
    from ib_insync import IB, Contract, LimitOrder, MarketOrder, Order, StopOrder

from cache import (
    close_open_bot_trades,
    load_bot_daily_risk_state,
    log_bot_trade_entry,
    mark_open_bot_trades_orphaned,
    update_bot_trade_exit,
)

MNQ_EXCHANGE = "CME"
MNQ_SYMBOL = "MNQ"
MNQ_MULTIPLIER = "2"  # $2 per point
MNQ_CURRENCY = "USD"
MNQ_POINT_VALUE = 2.0  # $2 per point for P&L calculation
ORDER_QTY = 1  # 1 contract per alert


@dataclass
class TradeResult:
    """Result of a bracket order submission."""

    success: bool
    order_id: int | None = None
    entry_price: float | None = None
    target_price: float | None = None
    stop_price: float | None = None
    error: str | None = None


class IBKRBroker:
    """Manages IBKR connection, order submission, and risk controls.

    Thread-safe: ib_insync event loop runs on a background thread.
    Call connect() once at startup, then submit_bracket() per alert.

    IMPORTANT: call process_events() periodically (e.g. on each trade tick)
    so ib_insync can dispatch fill callbacks. Without this, the broker
    won't know when positions close.

    Risk state resets daily via reset_daily_state().
    """

    def __init__(self) -> None:
        self._ib: IB | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._contract: Contract | None = None  # cached qualified contract
        self._was_connected = False  # tracks if we ever connected successfully
        self._connect_attempted = False  # True after first connect() call
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._heartbeat_interval_secs = 300.0  # log connection state every 5 min
        self._last_heartbeat_time: float = 0.0
        self._last_reconnect_time: float = 0.0  # monotonic timestamp
        self._reconnect_interval_secs = 60.0  # retry every 60s, not every tick

        # Risk management state — resets each trading day.
        self._daily_pnl_usd: float = 0.0
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        self._trades_today: int = 0
        self._wins_today: int = 0
        self._losses_today: int = 0
        self._stopped_for_day: bool = False
        self._stop_reason: str = ""

        # Position tracking — one position at a time.
        self._position_open: bool = False
        self._pending_direction: str | None = None  # "up" or "down"
        self._pending_line_price: float | None = None
        self._pending_entry_fill: float | None = None  # actual fill price
        self._pending_target_pts: float = 8.0
        self._pending_stop_pts: float = 25.0
        self._pending_level_name: str | None = None
        self._pending_target_price: float | None = None
        self._pending_stop_price: float | None = None
        self._pending_db_trade_id: int | None = None  # row id in bot_trades table
        self._pending_exit_reason: str | None = None  # set before close orders
        self._pending_score: int | None = None  # bot entry score
        self._pending_trend_60m: float | None = None  # 60m trend at entry
        self._pending_entry_count: int | None = None  # which retest of this level
        self._pending_parent_order_id: int | None = None  # parent orderId (for recovery)
        self._pending_entry_limit: float | None = None  # limit order price
        self._pending_range_30m: float | None = None  # 30m range at entry
        self._pending_tick_rate: float | None = None  # tick rate at entry
        self._pending_session_move_pct: float | None = None  # session move % at entry
        self._position_opened_at: float | None = None  # monotonic() timestamp

    def connect(self) -> bool:
        """Connect to IB Gateway / TWS. Returns True on success."""
        self._connect_attempted = True
        if not IBKR_TRADING_ENABLED:
            print("[broker] Trading disabled (IBKR_TRADING_ENABLED=false)")
            return False
        try:
            self._ib = IB()
            self._ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
            self._connected = True
            self._was_connected = True
            self._reconnect_attempts = 0

            # Verify we're on the expected account (safety check).
            managed = self._ib.managedAccounts()
            if IBKR_ACCOUNT:
                if IBKR_ACCOUNT not in managed:
                    print(
                        f"[broker] FATAL: Expected account {IBKR_ACCOUNT} "
                        f"but connected to {managed}. Aborting."
                    )
                    self._ib.disconnect()
                    self._connected = False
                    return False
                print(f"[broker] Account verified: {IBKR_ACCOUNT}")
            else:
                print(
                    f"[broker] WARNING: IBKR_ACCOUNT not set. Connected accounts: {managed}"
                )

            # Register fill callback for risk tracking.
            self._ib.orderStatusEvent += self._on_order_status

            # Re-hydrate risk counters from today's closed trades so a
            # mid-day restart doesn't hand the bot a fresh daily budget
            # or clear an existing consecutive-loss stop.
            self._restore_daily_state()

            # Pre-qualify the MNQ contract so we don't do it on every order.
            self._contract = self._resolve_contract()

            # Adopt any open MNQ position left by a previous session so we
            # don't silently stack a second entry on top of it. Runs after
            # contract resolution because _defensive_flatten needs it.
            self._reconcile_open_position()
            if self._contract:
                print(
                    f"[broker] MNQ contract qualified: "
                    f"{self._contract.localSymbol} "
                    f"(secType={self._contract.secType})"
                )
            else:
                print("[broker] WARNING: Could not qualify MNQ contract")

            print(
                f"[broker] Connected to IBKR at {IBKR_HOST}:{IBKR_PORT} "
                f"(client {IBKR_CLIENT_ID})"
            )
            print(
                f"[broker] Risk limits: ${DAILY_LOSS_LIMIT_USD:.0f}/day, "
                f"1 position at a time"
            )
            return True
        except Exception as exc:
            print(f"[broker] Connection failed: {exc}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            self._contract = None
            print("[broker] Disconnected from IBKR")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    def reconnect(self) -> bool:
        """Attempt to reconnect after disconnection or initial connection failure.

        Retries up to max_reconnect_attempts, then gives up (until reset_daily_state
        clears the counter for the next trading day).
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            if self._reconnect_attempts == self._max_reconnect_attempts:
                print(
                    f"[broker] Reconnect failed after "
                    f"{self._max_reconnect_attempts} attempts — giving up"
                )
                self._reconnect_attempts += 1  # only print once
            return False
        self._reconnect_attempts += 1
        print(
            f"[broker] Connection lost — reconnecting "
            f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})..."
        )
        try:
            if self._ib:
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
            self._ib = IB()
            self._ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
            self._connected = True
            self._reconnect_attempts = 0
            self._ib.orderStatusEvent += self._on_order_status
            self._contract = self._resolve_contract()
            print(
                f"[broker] Reconnected to IBKR "
                f"(contract: {self._contract.localSymbol if self._contract else 'N/A'})"
            )
            return True
        except Exception as exc:
            print(f"[broker] Reconnect failed: {exc}")
            self._connected = False
            return False

    def process_events(self) -> None:
        """Pump the ib_insync event loop so callbacks fire.

        Must be called periodically from the main loop (e.g. on each
        trade tick). Without this, orderStatusEvent callbacks won't
        dispatch and the broker won't know when positions close.

        Also detects disconnections (or initial connection failures) and
        attempts to reconnect with rate limiting (every 60s, not every tick).

        Emits a periodic heartbeat log line (every 5 min) so post-session
        investigation can confirm connection state at any point. Without
        this, today's (2026-04-15) afternoon disconnect went undetected
        because the journal had rotated all skip/block messages.
        """
        now = time.monotonic()

        # Periodic heartbeat — always emit regardless of connection state.
        if now - self._last_heartbeat_time >= self._heartbeat_interval_secs:
            self._last_heartbeat_time = now
            connected = self.is_connected
            if connected:
                print(
                    f"[broker] heartbeat: CONNECTED | "
                    f"position_open={self._position_open} | "
                    f"{self.daily_stats} | "
                    f"consec_losses={self._consecutive_losses} | "
                    f"reconnects={self._reconnect_attempts}/"
                    f"{self._max_reconnect_attempts}"
                    + (
                        f" | STOPPED: {self._stop_reason}"
                        if self._stopped_for_day
                        else ""
                    ),
                    flush=True,
                )
            else:
                print(
                    f"[broker] heartbeat: DISCONNECTED | "
                    f"reconnects={self._reconnect_attempts}/"
                    f"{self._max_reconnect_attempts} | "
                    f"was_connected={self._was_connected}",
                    flush=True,
                )

        if self._connect_attempted and not self.is_connected:
            if now - self._last_reconnect_time >= self._reconnect_interval_secs:
                self._last_reconnect_time = now
                self.reconnect()
            return
        if self._ib and self.is_connected:
            try:
                self._ib.sleep(0)  # non-blocking: process pending events
            except Exception:
                pass

    def _restore_daily_state(self) -> None:
        """Repopulate risk counters from today's closed bot_trades.

        Only called from connect(), not reconnect(): a mid-process socket
        recovery should trust in-memory state (which already reflects fills
        via orderStatusEvent), while a fresh process has zeroed counters
        that would silently bypass the daily cap.
        """
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        today = now.strftime("%Y-%m-%d")
        try:
            state = load_bot_daily_risk_state(today)
        except Exception as exc:
            print(f"[broker] Failed to restore daily state: {exc}")
            return
        if state["trades"] == 0:
            return
        with self._lock:
            self._daily_pnl_usd = state["pnl_usd"]
            self._trades_today = state["trades"]
            self._wins_today = state["wins"]
            self._losses_today = state["losses"]
            self._consecutive_losses = state["consecutive_losses"]
            self._consecutive_wins = state.get("consecutive_wins", 0)
            # Eagerly set stopped_for_day so the startup banner and the
            # first can_trade() call reflect the restored state.
            if self._daily_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
                self._stopped_for_day = True
                self._stop_reason = (
                    f"Daily loss limit hit (restored: "
                    f"${self._daily_pnl_usd:+.2f})"
                )
        print(
            f"[broker] Restored daily state: {self._trades_today} trades "
            f"({self._wins_today}W/{self._losses_today}L), "
            f"P&L ${self._daily_pnl_usd:+.2f}, "
            f"consec wins {self._consecutive_wins} losses {self._consecutive_losses}"
            + (f" | STOPPED: {self._stop_reason}" if self._stopped_for_day else "")
        )

    def _reconcile_open_position(self) -> None:
        """Flatten any leftover MNQ positions and cancel stray orders on startup.

        A restart mid-trade orphans bracket orders from the old process,
        leaving a naked position that can't be tracked. Rather than trying
        to adopt the position (which has caused repeated issues), we simply
        flatten everything and start clean. Daily risk state is already
        restored from the DB so P&L tracking is preserved.
        """
        if not self._ib:
            return
        # Positions and openTrades are populated asynchronously after
        # connect. Pump the event loop briefly so both are ready.
        try:
            self._ib.sleep(1.0)
        except Exception:
            pass

        # Cancel any open MNQ orders first (before flattening positions,
        # so stray bracket orders don't interfere).
        try:
            stray = [
                t
                for t in self._ib.openTrades()
                if t.contract.symbol == MNQ_SYMBOL
            ]
        except Exception:
            stray = []
        if stray:
            print(
                f"[broker] Startup: cancelling {len(stray)} stray MNQ order(s)"
            )
            for trade in stray:
                try:
                    self._ib.cancelOrder(trade.order)
                except Exception as exc:
                    print(f"[broker] Stray cancel error: {exc}")
            try:
                self._ib.sleep(1.0)
            except Exception:
                pass

        # Flatten any open MNQ positions.
        try:
            mnq_positions = [
                p
                for p in self._ib.positions()
                if p.contract.symbol == MNQ_SYMBOL and p.position != 0
            ]
        except Exception as exc:
            print(f"[broker] Failed to query positions during reconcile: {exc}")
            return

        if mnq_positions:
            for pos in mnq_positions:
                qty = int(pos.position)
                action = "BUY" if qty < 0 else "SELL"
                direction = "up" if qty > 0 else "down"
                try:
                    multiplier = float(pos.contract.multiplier or MNQ_MULTIPLIER)
                except (TypeError, ValueError):
                    multiplier = float(MNQ_MULTIPLIER)
                entry_price = pos.avgCost / multiplier if multiplier else 0.0
                print(
                    f"[broker] Startup: flattening leftover {pos.contract.localSymbol} "
                    f"qty={qty} entry~{entry_price:.2f} @ market"
                )
                try:
                    contract = self._contract or pos.contract
                    order = MarketOrder(action, abs(qty))
                    trade = self._ib.placeOrder(contract, order)
                    self._ib.sleep(2.0)
                    fill_price = trade.orderStatus.avgFillPrice or 0.0
                    status = trade.orderStatus.status
                    print(
                        f"[broker] Startup flatten: {action} {abs(qty)} "
                        f"{pos.contract.localSymbol} fill={fill_price:.2f} "
                        f"status={status}"
                    )
                    # Record P&L from the flatten.
                    if status == "Filled" and fill_price > 0 and entry_price > 0:
                        if direction == "up":
                            pnl_pts = fill_price - entry_price
                        else:
                            pnl_pts = entry_price - fill_price
                        pnl_usd = pnl_pts * MNQ_POINT_VALUE - 1.24
                        self._daily_pnl_usd += pnl_usd
                        self._trades_today += 1
                        if pnl_usd >= 0:
                            self._wins_today += 1
                            self._consecutive_losses = 0
                            self._consecutive_wins += 1
                        else:
                            self._losses_today += 1
                            self._consecutive_losses += 1
                            self._consecutive_wins = 0
                        print(
                            f"[broker] Startup flatten P&L: ${pnl_usd:+.2f} "
                            f"(entry~{entry_price:.2f} exit={fill_price:.2f}) | "
                            f"{self.daily_stats}"
                        )
                        # Update the matching DB row with actual exit details.
                        try:
                            now = datetime.datetime.now(
                                datetime.timezone.utc
                            ).astimezone()
                            today = now.strftime("%Y-%m-%d")
                            updated = close_open_bot_trades(
                                date_str=today,
                                exit_time=now.strftime("%H:%M:%S %Z"),
                                exit_price=fill_price,
                                pnl_usd=pnl_usd,
                                exit_reason="startup_flatten",
                            )
                            if updated:
                                print(
                                    f"[broker] Updated {updated} open bot_trades "
                                    f"row(s) with flatten P&L"
                                )
                        except Exception as exc:
                            print(f"[broker] Startup DB update error: {exc}")
                except Exception as exc:
                    print(f"[broker] Startup flatten error: {exc}")
            # Verify.
            try:
                self._ib.sleep(1.0)
                remaining = [
                    p
                    for p in self._ib.positions()
                    if p.contract.symbol == MNQ_SYMBOL and p.position != 0
                ]
                if remaining:
                    print(
                        f"[broker] WARNING: {len(remaining)} MNQ position(s) "
                        f"still open after startup flatten"
                    )
                else:
                    print("[broker] Startup reconcile: all positions flattened")
            except Exception:
                pass
        else:
            if stray:
                print("[broker] Startup reconcile: no positions, stray orders cancelled")
            else:
                print("[broker] Startup reconcile: no positions or stray orders")

    def _cancel_all_mnq(self, context: str = "") -> int:
        """Cancel all open MNQ orders using both local and server-side queries.

        openTrades() only sees orders placed by this client session.
        reqAllOpenOrders() queries IBKR's server for any open orders on
        the account, catching orphans from crashed sessions or manual
        TWS entries. Both return List[Trade].

        cancelOrder() takes an Order (not Trade) and is asynchronous —
        it sends the cancel request but confirmation arrives via
        orderStatusEvent callback. We verify cancellation by pumping
        the event loop and re-checking.
        """
        if not self._ib:
            return 0
        cancelled = 0
        seen_order_ids: set[int] = set()

        # Pass 1: orders tracked by this ib_insync session.
        try:
            for trade in self._ib.openTrades():
                if trade.contract.symbol == MNQ_SYMBOL:
                    oid = trade.order.orderId
                    seen_order_ids.add(oid)
                    ref = getattr(trade.order, "orderRef", "") or ""
                    status = trade.orderStatus.status
                    print(
                        f"[broker] {context}: cancelling order {oid} "
                        f"({ref}, status={status})",
                        flush=True,
                    )
                    self._ib.cancelOrder(trade.order)
                    cancelled += 1
        except Exception as exc:
            print(
                f"[broker] {context}: openTrades cancel error: {exc}",
                flush=True,
            )

        # Pass 2: server-side sweep for orphaned orders from other
        # sessions (e.g. after a crash/restart).
        try:
            server_trades = self._ib.reqAllOpenOrders()
            for trade in server_trades:
                oid = trade.order.orderId
                if (
                    oid not in seen_order_ids
                    and trade.contract.symbol == MNQ_SYMBOL
                ):
                    ref = getattr(trade.order, "orderRef", "") or ""
                    status = trade.orderStatus.status
                    print(
                        f"[broker] {context}: cancelling ORPHANED order "
                        f"{oid} ({ref}, status={status})",
                        flush=True,
                    )
                    self._ib.cancelOrder(trade.order)
                    cancelled += 1
        except Exception as exc:
            print(
                f"[broker] {context}: reqAllOpenOrders cancel error: {exc}",
                flush=True,
            )

        # Pump the event loop so cancel confirmations arrive, then
        # verify no MNQ orders remain.
        if cancelled:
            try:
                self._ib.sleep(0.5)
            except Exception:
                pass
            try:
                remaining = [
                    t for t in self._ib.openTrades()
                    if t.contract.symbol == MNQ_SYMBOL
                ]
                if remaining:
                    for t in remaining:
                        print(
                            f"[broker] {context}: WARNING — order "
                            f"{t.order.orderId} still open after cancel "
                            f"(status={t.orderStatus.status})",
                            flush=True,
                        )
                else:
                    print(
                        f"[broker] {context}: all {cancelled} cancel(s) "
                        f"confirmed",
                        flush=True,
                    )
            except Exception:
                pass

        return cancelled

    def _has_mnq_position(self) -> bool:
        """Check if any MNQ position exists on IBKR."""
        try:
            return any(
                p.contract.symbol == MNQ_SYMBOL and p.position != 0
                for p in self._ib.positions()
            )
        except Exception:
            return False

    def _close_mnq_at_market(self, order_ref: str = "bot-close") -> list:
        """Submit market order(s) to close all open MNQ positions.

        Returns the list of submitted close Trade objects (empty list
        if no open MNQ position was found). Callers that previously
        treated this as bool can use `if returned_list:` — the truthy
        semantics are preserved.

        Does not wait for fill — caller is responsible for awaiting
        the close if it needs the fill price (e.g. _defensive_flatten
        needs the exit price to attribute P&L to orphaned rows).
        """
        closed_trades: list = []
        try:
            for pos in self._ib.positions():
                if pos.contract.symbol == MNQ_SYMBOL and pos.position != 0:
                    action = "SELL" if pos.position > 0 else "BUY"
                    qty = abs(int(pos.position))
                    contract = self._contract or pos.contract
                    close_order = MarketOrder(action, qty)
                    close_order.orderRef = order_ref
                    close_trade = self._ib.placeOrder(contract, close_order)
                    closed_trades.append(close_trade)
        except Exception as exc:
            print(f"[broker] Close at market error: {exc}", flush=True)
        return closed_trades

    def _defensive_flatten(self, reason: str) -> None:
        """Cancel all open MNQ orders and market-close any MNQ position.

        Used by reconcile when we can't trust the linkage between an open
        position and the bot's record of it. Does NOT set _pending_*
        state — the flatten is intentionally blind and its fill is
        ignored by _on_order_status (because _position_open is False).

        After the close market order fills, attributes the exit price/
        time/P&L to any orphaned bot_trades rows so the audit trail
        and daily P&L include the recovery, not just NULLs.
        """
        if not self._ib:
            return
        self._cancel_all_mnq(f"Defensive flatten ({reason})")
        close_trades = self._close_mnq_at_market("bot-defensive-flatten")
        if close_trades:
            print(
                f"[broker] Defensive flatten ({reason}): "
                f"closing MNQ position at market",
                flush=True,
            )
        else:
            print(
                f"[broker] Defensive flatten ({reason}): "
                f"no MNQ position found on IBKR"
            )

        # Wait for the close market order(s) to fill so we can attribute
        # the exit price to the orphaned row(s). Market orders typically
        # fill in <500ms; cap the wait at 5s.
        exit_price: float | None = None
        if close_trades:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    self._ib.sleep(0.1)
                except Exception:
                    break
                if all(
                    t.orderStatus.status == "Filled" for t in close_trades
                ):
                    break
            # Aggregate the fill price across all close legs (usually 1).
            # avgFillPrice is the source of truth for market fills.
            filled = [
                t for t in close_trades
                if t.orderStatus.status == "Filled"
                and t.orderStatus.avgFillPrice
            ]
            if filled:
                exit_price = float(filled[-1].orderStatus.avgFillPrice)
            else:
                print(
                    f"[broker] Defensive flatten ({reason}): close order(s) "
                    f"did not fill within 5s — orphan row(s) will lack "
                    f"exit price",
                    flush=True,
                )

        # Sweep any 'open' bot_trades rows for today so they don't stay
        # stuck forever. Pass the close fill's exit info if we have it
        # so P&L is attributed instead of left NULL.
        try:
            now = datetime.datetime.now(datetime.timezone.utc).astimezone()
            today = now.strftime("%Y-%m-%d")
            exit_time_str = (
                now.strftime("%H:%M:%S %Z") if exit_price is not None else None
            )
            updated = mark_open_bot_trades_orphaned(
                today, exit_time=exit_time_str, exit_price=exit_price
            )
            if updated:
                attribution = (
                    f"with exit @ {exit_price:.2f}"
                    if exit_price is not None
                    else "without exit price (close fill not observed)"
                )
                print(
                    f"[broker] Marked {updated} open bot_trades row(s) "
                    f"as 'orphaned' {attribution}"
                )
        except Exception as exc:
            print(f"[broker] Orphan sweep error: {exc}")

    def reset_daily_state(self) -> None:
        """Reset risk counters for a new trading day.

        Before clearing position state, checks IBKR for any leftover
        MNQ position (e.g. from a failed flatten) and closes it.
        """
        # Check for leftover positions before resetting flags.
        if self._ib:
            try:
                has_leftover = False
                for p in self._ib.positions():
                    if p.contract.symbol == MNQ_SYMBOL and p.position != 0:
                        print(
                            f"[broker] Daily reset: IBKR has leftover MNQ "
                            f"position ({p.position}) — flattening"
                        )
                        self._defensive_flatten("daily reset leftover")
                        has_leftover = True
                        break
                if not has_leftover:
                    print("[broker] Daily reset: no leftover IBKR positions")
            except Exception as e:
                print(f"[broker] Daily reset: position check failed: {e}")

        with self._lock:
            prev_pnl = self._daily_pnl_usd
            prev_trades = self._trades_today
            self._daily_pnl_usd = 0.0
            self._consecutive_losses = 0
            self._consecutive_wins = 0
            self._trades_today = 0
            self._wins_today = 0
            self._losses_today = 0
            self._stopped_for_day = False
            self._stop_reason = ""
            self._position_open = False
            self._pending_direction = None
            self._pending_line_price = None
            self._pending_entry_fill = None
            self._pending_parent_order_id = None
            self._pending_db_trade_id = None
            self._position_opened_at = None
            self._reconnect_attempts = 0  # retry connection each new day
            if prev_trades > 0:
                print(
                    f"[broker] Daily reset (yesterday: {prev_trades} trades, "
                    f"${prev_pnl:+.2f})"
                )
            else:
                print("[broker] Daily risk state reset")

    def can_trade(self) -> tuple[bool, str]:
        """Check if risk limits allow a new trade.

        Returns (allowed, reason). If not allowed, reason explains why.
        """
        if self._stopped_for_day:
            return False, self._stop_reason
        if self._position_open:
            return False, "Position already open (1 at a time)"
        if self._daily_pnl_usd <= -DAILY_LOSS_LIMIT_USD:
            self._stopped_for_day = True
            self._stop_reason = (
                f"Daily loss limit hit (${self._daily_pnl_usd:+.2f} "
                f">= -${DAILY_LOSS_LIMIT_USD:.0f})"
            )
            return False, self._stop_reason
        return True, ""

    @property
    def daily_stats(self) -> str:
        """Human-readable daily trading stats."""
        return (
            f"{self._trades_today} trades "
            f"({self._wins_today}W/{self._losses_today}L), "
            f"P&L: ${self._daily_pnl_usd:+.2f}"
        )

    def _on_order_status(self, trade) -> None:
        """Callback when an order status changes.

        Dispatches filled orders to the entry-recording path or the
        position-closing / P&L-recording path based on the orderRef tag
        we set at submission time. Every bot-submitted order carries a
        tag of the form "bot-<parent_id>-<role>" where role is one of
        parent / target / stop / close. Defensive flattens use the tag
        "bot-defensive-flatten" and are intentionally ignored (no DB row
        to update, no P&L to compute).

        The previous implementation dispatched on ``order.parentId``: a
        non-zero value meant "child/close." That approach broke for
        manual-close market orders because IBKR interpreted the sentinel
        parentId=1 as a reference to a real order and rejected it with
        "Error 135: Can't find order with id = 1" — leaving positions
        stuck open past their timeout.
        """
        if trade.contract.symbol != MNQ_SYMBOL:
            return

        # Handle cancelled/inactive parent (entry) limit orders: clear
        # _position_open so the bot isn't blocked from new trades. With
        # limit entries (not market), the parent can go unfilled if price
        # moves away before the limit is hit.
        if trade.orderStatus.status in ("Cancelled", "Inactive"):
            order_ref = (trade.order.orderRef or "").strip()
            if order_ref.startswith("bot-") and order_ref.endswith("-parent"):
                with self._lock:
                    if self._position_open and self._pending_entry_fill is None:
                        self._position_open = False
                        self._position_opened_at = None
                        print(
                            f"[broker] Entry limit order {trade.orderStatus.status.lower()} "
                            f"— clearing position state"
                        )
            return

        if trade.orderStatus.status != "Filled":
            return

        order = trade.order
        order_ref = (order.orderRef or "").strip()

        # Defensive flattens are intentionally untracked — the position
        # they close was orphaned (no matching bot_trades row), so there
        # is nothing to update and no P&L to attribute.
        if order_ref == "bot-defensive-flatten":
            return

        # Parse the role suffix from the orderRef. Expected format:
        # "bot-<parent_order_id>-<role>".
        role: str | None = None
        if order_ref.startswith("bot-"):
            parts = order_ref.rsplit("-", 1)
            if len(parts) == 2:
                role = parts[1]

        # Any non-bot order (e.g. manually placed in TWS) is ignored.
        if role is None:
            return

        if role == "parent":
            # Parent (entry) order filled — record the actual fill price.
            with self._lock:
                # Idempotency guard. IBKR can emit the parent-filled
                # orderStatus event more than once for the same order
                # during state-transition races (e.g. PreSubmitted →
                # Filled while a cancel is in-flight). Without this
                # guard, the second emit would re-INSERT a bot_trades
                # row AND overwrite _pending_db_trade_id, orphaning the
                # first row in the database. This is the root cause of
                # the phantom-DB-row bug behind both bracket_race_close
                # and defensive_flatten duplicates.
                if self._pending_db_trade_id is not None:
                    return
                self._pending_entry_fill = trade.orderStatus.avgFillPrice
                print(f"[broker] Entry filled @ {self._pending_entry_fill:.2f}")
                # Log entry to database.
                try:
                    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
                    self._pending_db_trade_id = log_bot_trade_entry(
                        date_str=now.strftime("%Y-%m-%d"),
                        entry_time=now.strftime("%H:%M:%S %Z"),
                        level_name=self._pending_level_name or "unknown",
                        direction=self._pending_direction or "unknown",
                        line_price=self._pending_line_price or 0.0,
                        entry_price=self._pending_entry_fill,
                        target_price=self._pending_target_price or 0.0,
                        stop_price=self._pending_stop_price or 0.0,
                        score=self._pending_score,
                        trend_60m=self._pending_trend_60m,
                        entry_count=self._pending_entry_count,
                        parent_order_id=self._pending_parent_order_id,
                        range_30m=self._pending_range_30m,
                        tick_rate=self._pending_tick_rate,
                        session_move_pct=self._pending_session_move_pct,
                        entry_limit=self._pending_entry_limit,
                    )
                except Exception as e:
                    print(f"[broker] Error logging trade entry to DB: {e}")
            return

        # Only target / stop / close roles run the P&L path. Unknown
        # roles are ignored defensively.
        if role not in ("target", "stop", "close"):
            return

        with self._lock:
            if not self._position_open:
                return  # Already processed or spurious

            exit_price = trade.orderStatus.avgFillPrice

            # Determine exit reason: use explicit reason if set (timeout/eod),
            # otherwise infer from order type.
            if self._pending_exit_reason:
                exit_reason = self._pending_exit_reason
                self._pending_exit_reason = None
            elif order.orderType == "LMT":
                exit_reason = "target"
            elif order.orderType == "STP":
                exit_reason = "stop"
            else:
                exit_reason = "market"

            self._record_trade_close(exit_price, exit_reason)
            self.can_trade()  # updates _stopped_for_day if limits hit

            # Cancel any stale sibling orders from THIS bracket only.
            # Using _cancel_all_mnq here would risk cancelling a NEW
            # bracket's orders if the bot re-entered immediately.
            closed_parent_id = self._pending_parent_order_id
            if closed_parent_id is not None:
                try:
                    for t in self._ib.openTrades():
                        if (
                            t.contract.symbol == MNQ_SYMBOL
                            and t.order.parentId == closed_parent_id
                        ):
                            ref = getattr(t.order, "orderRef", "") or ""
                            print(
                                f"[broker] Post-close cleanup: cancelling "
                                f"stale order {t.order.orderId} ({ref})",
                                flush=True,
                            )
                            self._ib.cancelOrder(t.order)
                except Exception as e:
                    print(f"[broker] Post-close cleanup error: {e}")

    def _record_trade_close(
        self, exit_price: float, exit_reason: str
    ) -> float:
        """Record P&L and update state for a closed position.

        Must be called with self._lock held. Returns pnl_usd.
        Used by _on_order_status and _verify_and_failsafe_close.
        """
        self._position_open = False
        self._position_opened_at = None
        self._trades_today += 1

        entry = self._pending_entry_fill or self._pending_line_price
        if self._pending_direction and entry is not None:
            if self._pending_direction == "up":
                pnl_pts = exit_price - entry
            else:
                pnl_pts = entry - exit_price
            pnl_usd = pnl_pts * MNQ_POINT_VALUE - 1.24
        else:
            # Fallback: estimate from target/stop pts if entry unknown.
            if exit_reason == "target":
                pnl_usd = self._pending_target_pts * MNQ_POINT_VALUE - 1.24
            elif exit_reason == "stop":
                pnl_usd = -(self._pending_stop_pts * MNQ_POINT_VALUE + 1.24)
            else:
                pnl_usd = 0.0

        self._daily_pnl_usd += pnl_usd

        if pnl_usd >= 0:
            self._wins_today += 1
            self._consecutive_losses = 0
            self._consecutive_wins += 1
        else:
            self._losses_today += 1
            self._consecutive_losses += 1
            self._consecutive_wins = 0

        print(
            f"[broker] Trade closed: "
            f"{'WIN' if pnl_usd >= 0 else 'LOSS'} ${pnl_usd:+.2f} | "
            f"Day: {self.daily_stats} | "
            f"Consec wins: {self._consecutive_wins} losses: {self._consecutive_losses}"
        )

        if self._pending_db_trade_id is not None:
            try:
                now = datetime.datetime.now(datetime.timezone.utc).astimezone()
                update_bot_trade_exit(
                    trade_id=self._pending_db_trade_id,
                    exit_time=now.strftime("%H:%M:%S %Z"),
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    outcome="win" if pnl_usd >= 0 else "loss",
                    exit_reason=exit_reason,
                )
            except Exception as e:
                print(f"[broker] Error logging trade exit to DB: {e}")
            self._pending_db_trade_id = None

        return pnl_usd

    def _verify_and_failsafe_close(self, reason: str) -> None:
        """After submitting a close order, verify the position actually closed.

        Pumps the ib_insync event loop for up to 10 seconds, checking
        positions() periodically. If the MNQ position is still open after
        that window, cancels any remaining MNQ orders and submits a raw
        market order directly (no orderRef dependency, no callback
        dependency). If even that fails, sends a Pushover alert.

        Called from eod_flatten() and check_position_timeout() as a
        last line of defense. Added after 2026-04-15 when a parentId=1
        sentinel bug left a short position stuck open for 45 minutes
        past its timeout.
        """
        if not self._ib:
            return

        # Quick check: if IBKR already has no MNQ position, we're done.
        try:
            self._ib.sleep(0.5)  # brief pump to process pending events
            if not self._has_mnq_position():
                return
        except Exception:
            pass

        # Wait up to 10s for the normal close (via _on_order_status).
        for _ in range(5):
            try:
                self._ib.sleep(2.0)
            except Exception:
                pass
            if not self._has_mnq_position():
                return

        # Position still open after 10s — the primary close order failed
        # or was rejected. Cancel any stray orders and try a direct close.
        print(
            f"[broker] {reason} close not confirmed after 10s "
            f"— attempting failsafe close",
            flush=True,
        )
        self._cancel_all_mnq("Failsafe")

        try:
            for pos in self._ib.positions():
                if pos.contract.symbol == MNQ_SYMBOL and pos.position != 0:
                    action = "SELL" if pos.position > 0 else "BUY"
                    qty = abs(int(pos.position))
                    contract = self._contract or pos.contract
                    order = MarketOrder(action, qty)
                    trade = self._ib.placeOrder(contract, order)
                    # Wait up to 10s for fill
                    for _ in range(20):
                        self._ib.sleep(0.5)
                        if trade.orderStatus.status == "Filled":
                            fill_price = trade.orderStatus.avgFillPrice
                            print(
                                f"[broker] Failsafe close filled @ "
                                f"{fill_price:.2f}",
                                flush=True,
                            )
                            with self._lock:
                                self._record_trade_close(
                                    fill_price, f"failsafe_{reason}"
                                )
                            return
                    print(
                        "[broker] WARNING: Failsafe close order not "
                        "filled within 10s",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[broker] Failsafe close error: {exc}", flush=True)

        # Even the failsafe failed — alert the user
        print(
            "[broker] CRITICAL: Position could not be closed. "
            "Manual intervention required.",
            flush=True,
        )
        try:
            from notifications import send_notification, PRIORITY_HIGH

            send_notification(
                title="CRITICAL: Position Not Closed",
                message=(
                    f"{reason} close failed and failsafe also failed. "
                    f"MNQ position is still open. "
                    f"Close manually via IBKR Mobile or VNC immediately."
                ),
                priority=PRIORITY_HIGH,
            )
        except Exception:
            pass

    def eod_flatten(self) -> None:
        """Pre-close safety: flatten any open position and block new trades.

        Called a few minutes before MARKET_CLOSE to avoid overnight margin
        requirements. Uses the same reverse-market-order approach as the
        timeout path so the fill callback records P&L normally.

        As a safety net, also checks IBKR's actual positions — if IBKR
        has an MNQ position but our internal flag is False (e.g. from a
        race condition), we flatten it anyway to avoid overnight exposure.
        """
        if not self.is_connected:
            return
        with self._lock:
            if self._stopped_for_day and not self._position_open:
                # Even if we think we're done, check IBKR for phantom positions.
                pass
            if not self._stopped_for_day:
                self._stopped_for_day = True
                self._stop_reason = "Pre-close EOD flatten (no new trades)"
                print(f"[broker] {self._stop_reason}")
            if self._position_open and self._position_opened_at is not None:
                direction = self._pending_direction
                self._position_opened_at = None  # prevent timeout path racing
            else:
                direction = None

        # Safety net: check IBKR's actual positions regardless of our flag.
        ibkr_position = None
        try:
            for p in self._ib.positions():
                if p.contract.symbol == MNQ_SYMBOL and p.position != 0:
                    ibkr_position = p
                    break
        except Exception as e:
            print(f"[broker] EOD flatten: error checking IBKR positions: {e}")

        if direction is None and ibkr_position is None:
            # No position, but sweep any stale orders (e.g. unfilled
            # entry limits that could fill after hours).
            stale = self._cancel_all_mnq("EOD order sweep")
            if stale:
                print(f"[broker] EOD flatten: no position but cancelled {stale} stale order(s)")
            else:
                print("[broker] EOD flatten: no position, no orders")
            return

        if direction is None and ibkr_position is not None:
            # Phantom position: IBKR has it but we don't know about it.
            # Use _defensive_flatten which checks IBKR's actual positions
            # directly — the normal close path would skip it because
            # _position_open is False.
            print(
                f"[broker] EOD flatten: IBKR has untracked MNQ position "
                f"({ibkr_position.position}) — flattening defensively"
            )
            self._defensive_flatten("EOD phantom position")
            return

        if self._contract is None:
            return
        self._cancel_all_mnq("EOD flatten")

        # Re-check under lock: a TP/stop fill may have closed the position
        # while we were cancelling. Submitting a reverse market order at this
        # point would open a new position in the opposite direction.
        with self._lock:
            if not self._position_open:
                print(
                    "[broker] Position closed before EOD market order "
                    "— skipping flatten (race with TP/stop fill)"
                )
                return

        # Final safety: verify IBKR actually has a position.
        try:
            ibkr_has_position = any(
                p.contract.symbol == MNQ_SYMBOL and p.position != 0
                for p in self._ib.positions()
            )
            if not ibkr_has_position:
                print(
                    "[broker] EOD flatten: _position_open=True but IBKR has "
                    "no MNQ position — clearing flag, sweeping orders"
                )
                with self._lock:
                    self._position_open = False
                    self._pending_entry_fill = None
                stale = self._cancel_all_mnq("EOD no-position sweep")
                if stale:
                    print(f"[broker] EOD flatten: cancelled {stale} stale order(s)")
                return
        except Exception as exc:
            print(f"[broker] EOD flatten: position check failed: {exc}")

        close_action = "SELL" if direction == "up" else "BUY"
        close_order = MarketOrder(close_action, ORDER_QTY)
        # Tag with orderRef so _on_order_status routes the fill through
        # the close/P&L path. Using parentId=1 as a sentinel (the prior
        # approach) made IBKR reject the order with error 135 because
        # IBKR treats parentId as a real order reference.
        parent_id_tag = self._pending_parent_order_id or "unknown"
        close_order.orderRef = f"bot-{parent_id_tag}-close"
        self._pending_exit_reason = "eod_flatten"
        try:
            self._ib.placeOrder(self._contract, close_order)
            print(f"[broker] EOD flatten: {close_action} {ORDER_QTY} MNQ @ market")
        except Exception as exc:
            print(f"[broker] EOD flatten failed: {exc}")

        self._verify_and_failsafe_close("eod_flatten")

    def check_position_timeout(self) -> bool:
        """If an open position has exceeded BOT_TIMEOUT_SECS, close it at market.

        Matches the 15-min window in bot_risk_backtest.py: if neither target nor
        stop hits in time, we exit (at whatever price the market gives us) so
        the position slot frees up for the next signal. Returns True if a
        timeout close was issued.

        Cancels TP/stop children (without touching position_open state) and
        submits a reverse market order. The fill callback will record P&L and
        clear position_open like a normal close.
        """
        if not self.is_connected:
            return False
        with self._lock:
            if not self._position_open or self._position_opened_at is None:
                return False
            elapsed = time.monotonic() - self._position_opened_at
            if elapsed < BOT_TIMEOUT_SECS:
                return False
            direction = self._pending_direction
            # Clear the timeout marker so we only issue one close.
            self._position_opened_at = None
            print(
                f"[broker] Position timeout ({elapsed:.0f}s > {BOT_TIMEOUT_SECS}s) "
                f"— cancelling bracket and closing at market"
            )

        contract = self._contract
        if contract is None or direction is None:
            return False

        # Cancel the TP + stop children so they don't race with the close.
        self._cancel_all_mnq("Timeout")

        # Re-check under lock: a TP/stop fill may have arrived between the
        # first check and now (ib_insync runs callbacks during placeOrder/
        # cancelOrder). If the position is already closed, submitting a
        # reverse market order here would OPEN a new position in the
        # opposite direction, which must not happen.
        with self._lock:
            if not self._position_open:
                print(
                    "[broker] Position closed before timeout market order "
                    "— skipping close (race with TP/stop fill)"
                )
                return False

        # Final safety: verify IBKR actually has a position. If our flag
        # says open but IBKR has nothing (dropped fill callback), submitting
        # a reverse order would accidentally open a new position.
        try:
            ibkr_has_position = any(
                p.contract.symbol == MNQ_SYMBOL and p.position != 0
                for p in self._ib.positions()
            )
            if not ibkr_has_position:
                print(
                    "[broker] Timeout: _position_open=True but IBKR has "
                    "no MNQ position — clearing flag, sweeping orders"
                )
                with self._lock:
                    self._position_open = False
                    self._pending_entry_fill = None
                # Sweep any stale orders that might still be open.
                stale = self._cancel_all_mnq("Timeout no-position sweep")
                if stale:
                    print(f"[broker] Timeout: cancelled {stale} stale order(s)")
                return False
        except Exception as exc:
            print(f"[broker] Timeout: position check failed: {exc}")
            # Proceed with close — safer to close a real position than
            # to skip and leave it open overnight.

        # Submit reverse market order. orderRef "bot-<parent>-close"
        # routes the fill through the close/P&L path in
        # _on_order_status. Previously used parentId=1 as a sentinel,
        # which IBKR rejected with "Error 135: Can't find order with
        # id = 1" and left the position stuck open.
        close_action = "SELL" if direction == "up" else "BUY"
        close_order = MarketOrder(close_action, ORDER_QTY)
        parent_id_tag = self._pending_parent_order_id or "unknown"
        close_order.orderRef = f"bot-{parent_id_tag}-close"
        self._pending_exit_reason = "timeout"
        try:
            self._ib.placeOrder(contract, close_order)
            print(f"[broker] Timeout close: {close_action} {ORDER_QTY} MNQ @ market")
        except Exception as exc:
            print(f"[broker] Timeout close failed: {exc}")
            return False

        self._verify_and_failsafe_close("timeout")
        return True

    def _resolve_contract(self) -> Contract | None:
        """Qualify the MNQ contract so IBKR knows the exact instrument.

        Uses CONTFUT (continuous future) which IBKR resolves to the
        actual front-month FUT contract during qualification. Verifies
        the resolved contract is tradeable (secType = FUT).
        """
        contract = Contract()
        contract.symbol = MNQ_SYMBOL
        contract.secType = "CONTFUT"
        contract.exchange = MNQ_EXCHANGE
        contract.multiplier = MNQ_MULTIPLIER
        contract.currency = MNQ_CURRENCY
        try:
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                print("[broker] Could not qualify MNQ CONTFUT contract")
                return None
            resolved = qualified[0]
            # CONTFUT should resolve to FUT after qualification.
            # If still CONTFUT, re-qualify as FUT using the resolved conId.
            if resolved.secType == "CONTFUT":
                print("[broker] CONTFUT not resolved to FUT — re-qualifying via conId")
                fut = Contract(conId=resolved.conId)
                fut_qualified = self._ib.qualifyContracts(fut)
                if fut_qualified and fut_qualified[0].secType == "FUT":
                    resolved = fut_qualified[0]
                else:
                    print(
                        "[broker] WARNING: Could not resolve to tradeable FUT contract"
                    )
            return resolved
        except Exception as exc:
            print(f"[broker] Contract qualification failed: {exc}")
            return None

    def submit_bracket(
        self,
        direction: str,
        current_price: float,
        line_price: float,
        level_name: str,
        qty: int = ORDER_QTY,
        score: int | None = None,
        trend_60m: float | None = None,
        entry_count: int | None = None,
        target_pts: float = 8.0,
        stop_pts: float = 25.0,
        entry_limit_buffer: float = 4.0,
        range_30m: float | None = None,
        tick_rate: float | None = None,
        session_move_pct: float | None = None,
    ) -> TradeResult:
        """Submit a bracket order: market entry + limit target + stop loss.

        direction: 'up' → BUY, 'down' → SELL
        current_price: price at alert time (for logging)
        line_price: the level price (target/stop measured from here)
        level_name: e.g. 'IBL', 'VWAP' (for logging)
        score: bot entry score (stored on bot_trades for analysis)
        trend_60m: 60-minute price trend at entry
        entry_count: which retest of this level

        Checks risk limits before submitting. Returns TradeResult with
        success=False if limits are exceeded.
        """
        if not IBKR_TRADING_ENABLED:
            return TradeResult(success=False, error="Trading disabled")

        if not self.is_connected:
            print("[broker] Not connected — attempting reconnect...")
            if not self.connect():
                return TradeResult(success=False, error="Not connected to IBKR")

        # Pre-entry drift check: our in-memory _position_open flag drives
        # the "1 position at a time" guarantee, but it can diverge from
        # IBKR's actual state if a fill event is dropped, a manual flatten
        # clears the flag without confirmation, or a startup race happens.
        # Query IBKR directly and reconcile before risking a second entry.
        try:
            ibkr_has_mnq_position = any(
                p.contract.symbol == MNQ_SYMBOL and p.position != 0
                for p in self._ib.positions()
            )
        except Exception as exc:
            return TradeResult(
                success=False,
                error=f"Pre-entry position check failed: {exc}",
            )

        with self._lock:
            if ibkr_has_mnq_position and not self._position_open:
                # Dangerous drift: IBKR holds a position we don't know
                # about. Flatten it immediately so the bot can recover
                # and resume trading, rather than staying stuck all day.
                print(
                    "[broker] DRIFT: IBKR has an MNQ position but "
                    "_position_open=False. Flattening untracked position."
                )
                self._defensive_flatten("pre-entry drift")
                return TradeResult(
                    success=False,
                    error="State drift — flattened untracked MNQ position",
                )

            if not ibkr_has_mnq_position and self._position_open:
                # Safe drift: the bot thinks there's a position but IBKR
                # has none. The fill event for a close was probably
                # dropped. Clear the flag and pending state so this
                # entry can proceed cleanly.
                print(
                    "[broker] DRIFT: _position_open=True but IBKR has "
                    "no MNQ position. Clearing stale pending state, "
                    "sweeping orders, and allowing entry."
                )
                self._position_open = False
                self._pending_direction = None
                self._pending_line_price = None
                self._pending_entry_fill = None
                self._pending_parent_order_id = None
                self._pending_db_trade_id = None
                self._pending_level_name = None
                self._pending_target_price = None
                self._pending_stop_price = None
                self._pending_exit_reason = None
                self._pending_entry_limit = None
                self._pending_range_30m = None
                self._pending_tick_rate = None
                self._pending_session_move_pct = None
                self._position_opened_at = None
                # Sweep stale orders from the phantom position.
                stale = self._cancel_all_mnq("Drift safe-clear sweep")
                if stale:
                    print(f"[broker] DRIFT: cancelled {stale} stale order(s)")

            # Check risk limits.
            allowed, reason = self.can_trade()
            if not allowed:
                print(f"[broker] Trade blocked: {reason}")
                return TradeResult(success=False, error=reason)

            # Stash analytics fields so the entry-fill callback can persist them.
            self._pending_score = score
            self._pending_trend_60m = trend_60m
            self._pending_entry_count = entry_count
            self._pending_range_30m = range_30m
            self._pending_tick_rate = tick_rate
            self._pending_session_move_pct = session_move_pct

            result, parent, parent_trade = self._submit_bracket_locked(
                direction, current_price, line_price, level_name, qty,
                target_pts, stop_pts, entry_limit_buffer,
            )

        # Lock is released. If submission failed, return immediately.
        if not result.success or parent is None:
            return result

        # Wait for entry limit to fill. sleep() pumps the event loop,
        # which fires callbacks that acquire self._lock — must NOT hold
        # the lock here (threading.Lock is not reentrant).
        parent_id = result.order_id
        filled = False
        n_checks = int(BOT_FILL_TIMEOUT_SECS / 0.25)
        try:
            for _ in range(n_checks):
                self._ib.sleep(0.25)
                if parent_trade.orderStatus.status == "Filled":
                    filled = True
                    break
        except Exception:
            pass

        if not filled:
            print(
                f"[broker] Entry limit not filled — "
                f"cancelling bracket (order {parent_id})",
                flush=True,
            )
            try:
                self._ib.cancelOrder(parent)
            except Exception as exc:
                print(
                    f"[broker] WARNING: cancel of unfilled entry "
                    f"{parent_id} failed: {exc}",
                    flush=True,
                )
            # Wait for IBKR to give us a definitive resolution on the
            # parent order. After cancelOrder, the order ends up in one
            # of two terminal states:
            #   • Cancelled / ApiCancelled / Inactive — clean cancel
            #   • Filled — race: fill arrived during/before cancel
            # The previous fixed 0.5s sleep was a guess at IBKR's
            # async latency. On 2026-05-05 the fill-status notification
            # arrived ~600ms after cancel was issued — after the 0.5s
            # window — leaving the bot to incorrectly clear state and
            # orphan the position. Polling orderStatus.status here is
            # event-driven on actual IBKR state, with a 3s deadline.
            final_status: str | None = None
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    self._ib.sleep(0.1)
                except Exception:
                    break
                status = parent_trade.orderStatus.status
                if status in (
                    "Cancelled", "ApiCancelled", "Inactive", "Filled"
                ):
                    final_status = status
                    break

            if final_status == "Filled":
                # Read the fill price from the Trade object — this is
                # IBKR's source of truth, not our async-set
                # _pending_entry_fill flag (which may not have been
                # populated by the callback yet, depending on event
                # ordering inside ib_insync).
                fill_price = float(
                    parent_trade.orderStatus.avgFillPrice or 0.0
                )
                if fill_price == 0.0 and parent_trade.fills:
                    fill_price = float(
                        parent_trade.fills[-1].execution.price
                    )
                return self._handle_fill_during_cancel(parent_id, fill_price)

            # Belt-and-suspenders: even if order status reads cancelled
            # (or never resolved), check IBKR's actual position state
            # before declaring the trade dead. If a position somehow
            # exists — e.g., status update reordered ahead of the
            # position update, or the deadline expired before any
            # terminal status — treat as fill-during-cancel and
            # flatten via the same recovery path.
            if self._has_mnq_position():
                with self._lock:
                    entry_fill = self._pending_entry_fill or 0.0
                print(
                    f"[broker] WARNING: order {parent_id} "
                    f"status={final_status} but IBKR has MNQ position "
                    f"— treating as fill-during-cancel",
                    flush=True,
                )
                return self._handle_fill_during_cancel(parent_id, entry_fill)

            return self._handle_entry_cancelled(parent_id)

        return result

    def _handle_fill_during_cancel(
        self, parent_id: int, entry_fill: float
    ) -> "TradeResult":
        """Handle case where entry filled during cancel attempt.

        Cancels bracket children, closes position at market, and clears
        state. Returns TradeResult(success=False). IBKR can cascade the
        parent cancel to children, leaving position unprotected — so we
        always close at market (2026-05-01 incident).
        """
        # Cancel any surviving bracket children.
        try:
            all_open = self._ib.openTrades()
            children = [
                t for t in all_open
                if t.contract.symbol == MNQ_SYMBOL
                and t.order.parentId == parent_id
            ]
            if children:
                print(
                    f"[broker] Fill-during-cancel: cancelling "
                    f"{len(children)} bracket children",
                    flush=True,
                )
                for c in children:
                    try:
                        self._ib.cancelOrder(c.order)
                    except Exception:
                        pass
                self._ib.sleep(0.5)
        except Exception as exc:
            print(
                f"[broker] Fill-during-cancel: error checking "
                f"children: {exc}",
                flush=True,
            )

        # Close position if IBKR still has one.
        try:
            self._ib.sleep(0.5)
        except Exception:
            pass
        if self._has_mnq_position():
            print(
                f"[broker] WARNING: Entry {parent_id} filled "
                f"@ {entry_fill:.2f} during cancel "
                f"— closing at market (bracket unreliable)",
                flush=True,
            )
            self._pending_exit_reason = "bracket_race_close"
            self._close_mnq_at_market(f"bot-{parent_id}-close")
            try:
                # Give _on_order_status time to process the fill.
                self._ib.sleep(1.0)
            except Exception:
                pass
        else:
            print(
                f"[broker] Fill-during-cancel: no IBKR "
                f"position — already closed",
                flush=True,
            )

        # Ensure _position_open is cleared regardless.
        with self._lock:
            if self._position_open:
                self._position_open = False
                self._position_opened_at = None
                print(
                    f"[broker] Fill-during-cancel: cleared "
                    f"stuck position state",
                    flush=True,
                )

        return TradeResult(
            success=False,
            error="Entry filled during cancel — closed",
        )

    def _handle_entry_cancelled(self, parent_id: int) -> "TradeResult":
        """Handle case where entry was truly cancelled (no fill).

        Clears position state and cancels orphaned bracket children.
        """
        with self._lock:
            self._position_open = False
            self._position_opened_at = None
            self._pending_entry_fill = None
            self._pending_entry_limit = None
            self._pending_range_30m = None
            self._pending_tick_rate = None
            self._pending_session_move_pct = None
        # Cancel children outside lock to avoid deadlock.
        try:
            for t in self._ib.openTrades():
                if (
                    t.contract.symbol == MNQ_SYMBOL
                    and t.order.parentId == parent_id
                ):
                    ref = getattr(t.order, "orderRef", "") or ""
                    print(
                        f"[broker] Entry cancel cleanup: "
                        f"cancelling child {t.order.orderId} ({ref})",
                        flush=True,
                    )
                    self._ib.cancelOrder(t.order)
        except Exception as e:
            print(f"[broker] Entry cancel cleanup error: {e}")
        return TradeResult(
            success=False,
            error="Entry limit not filled",
        )

    def _submit_bracket_locked(
        self,
        direction: str,
        current_price: float,
        line_price: float,
        level_name: str,
        qty: int,
        target_pts: float = 8.0,
        stop_pts: float = 25.0,
        entry_limit_buffer: float = 4.0,
    ) -> tuple["TradeResult", "Order | None", "object | None"]:
        """Internal bracket submission (must hold self._lock).

        Returns (result, parent_order, parent_trade) so the caller can
        check fill status and cancel if needed after releasing the lock.
        """
        action = "BUY" if direction == "up" else "SELL"
        reverse_action = "SELL" if direction == "up" else "BUY"

        # Target and stop prices based on direction.
        if direction == "up":
            target_price = line_price + target_pts
            stop_price = line_price - stop_pts
        else:
            target_price = line_price - target_pts
            stop_price = line_price + stop_pts

        # Round to MNQ tick size (0.25).
        target_price = round(target_price * 4) / 4
        stop_price = round(stop_price * 4) / 4

        contract = self._contract
        if contract is None:
            # Try to re-qualify if we don't have a cached contract.
            contract = self._resolve_contract()
            self._contract = contract
        if contract is None:
            return (
                TradeResult(success=False, error="Contract resolution failed"),
                None,
                None,
            )

        try:
            # Build bracket: limit entry + limit take-profit + stop loss.
            # Entry limit caps slippage at entry_limit_buffer pts from line.
            #
            # Must pre-assign orderId to the parent so children can reference
            # it via parentId BEFORE placeOrder is called.
            if direction == "up":
                entry_limit = line_price + entry_limit_buffer
            else:
                entry_limit = line_price - entry_limit_buffer
            entry_limit = round(entry_limit * 4) / 4  # MNQ tick size
            self._pending_entry_limit = entry_limit
            parent = LimitOrder(action, qty, entry_limit)
            parent.orderId = self._ib.client.getReqId()
            parent.orderRef = f"bot-{parent.orderId}-parent"
            parent.transmit = False

            take_profit = LimitOrder(reverse_action, qty, target_price)
            take_profit.parentId = parent.orderId
            take_profit.orderRef = f"bot-{parent.orderId}-target"
            take_profit.transmit = False

            stop_loss = StopOrder(reverse_action, qty, stop_price)
            stop_loss.parentId = parent.orderId
            stop_loss.orderRef = f"bot-{parent.orderId}-stop"
            stop_loss.transmit = True  # transmit last to send all at once

            # Submit all three legs. placeOrder is a synchronous socket
            # write — it does NOT pump the event loop, so it's safe to
            # call while holding self._lock.
            parent_trade = self._ib.placeOrder(contract, parent)
            self._ib.placeOrder(contract, take_profit)
            self._ib.placeOrder(contract, stop_loss)

            parent_id = parent.orderId

            # Set position state optimistically so callbacks during the
            # fill-check loop below can record the entry fill and route
            # P&L correctly. If the fill doesn't come, we clear this.
            self._position_open = True
            self._pending_direction = direction
            self._pending_line_price = line_price
            self._pending_target_pts = target_pts
            self._pending_stop_pts = stop_pts
            self._pending_level_name = level_name
            self._pending_target_price = target_price
            self._pending_stop_price = stop_price
            self._pending_db_trade_id = None
            self._pending_parent_order_id = parent_id
            self._position_opened_at = time.monotonic()

            print(
                f"[broker] {action} {qty} MNQ @ limit {entry_limit:.2f} | "
                f"target {target_price:.2f} (+{target_pts}) | "
                f"stop {stop_price:.2f} (-{stop_pts}) | "
                f"level {level_name} @ {line_price:.2f} | "
                f"order {parent_id} | {self.daily_stats}"
            )

        except Exception as exc:
            error_msg = f"Order submission failed: {exc}"
            print(f"[broker] {error_msg}")
            return (
                TradeResult(success=False, error=error_msg),
                None,
                None,
            )

        return (
            TradeResult(
                success=True,
                order_id=parent_id,
                entry_price=current_price,
                target_price=target_price,
                stop_price=stop_price,
            ),
            parent,
            parent_trade,
        )

    def get_positions(self) -> list[dict]:
        """Return current MNQ positions."""
        if not self.is_connected:
            return []
        positions = []
        for pos in self._ib.positions():
            if pos.contract.symbol == MNQ_SYMBOL:
                positions.append(
                    {
                        "symbol": pos.contract.symbol,
                        "quantity": pos.position,
                        "avg_cost": pos.avgCost,
                    }
                )
        return positions

    def cancel_all_mnq_orders(self) -> int:
        """Cancel all open MNQ orders. Returns count of orders cancelled."""
        if not self.is_connected:
            return 0
        cancelled = self._cancel_all_mnq("cancel_all_mnq_orders")
        self._position_open = False
        return cancelled

    def flatten_positions(self) -> bool:
        """Close all MNQ positions with market orders."""
        if not self.is_connected:
            return False
        for pos in self._ib.positions():
            if pos.contract.symbol == MNQ_SYMBOL and pos.position != 0:
                action = "SELL" if pos.position > 0 else "BUY"
                qty = abs(pos.position)
                order = MarketOrder(action, qty)
                contract = self._contract or pos.contract
                self._ib.placeOrder(contract, order)
                print(f"[broker] Flattening: {action} {qty} MNQ @ market")
        self._position_open = False
        return True

    def session_close(self) -> None:
        """Final safety close at market end — mirrors eod_flatten with failsafe.

        Called from bot_trader.close_session() at 4:00 PM ET as the last
        line of defense. If eod_flatten already closed the position at
        3:58 PM, this is a no-op. If the position is still open (eod
        flatten failed, was skipped, or a new position snuck in), this
        path cancels remaining orders, submits a tracked reverse market
        order with proper orderRef/P&L routing, and verifies the close
        via _verify_and_failsafe_close before disconnecting.
        """
        if not self.is_connected:
            return

        with self._lock:
            has_tracked_position = self._position_open
            direction = self._pending_direction
            if has_tracked_position:
                self._position_opened_at = None  # prevent timeout path racing

        # Safety net: check IBKR's actual positions regardless of our flag.
        ibkr_has_position = False
        try:
            ibkr_has_position = any(
                p.contract.symbol == MNQ_SYMBOL and p.position != 0
                for p in self._ib.positions()
            )
        except Exception as e:
            print(f"[broker] Session close: position check failed: {e}")

        if not has_tracked_position and not ibkr_has_position:
            stale = self._cancel_all_mnq("Session close order sweep")
            if stale:
                print(f"[broker] Session close: no position but cancelled {stale} stale order(s)")
            else:
                print("[broker] Session close: no position, no orders")
            return

        if not has_tracked_position and ibkr_has_position:
            # Phantom position — use defensive flatten.
            print(
                "[broker] Session close: IBKR has untracked MNQ position "
                "— flattening defensively"
            )
            self._defensive_flatten("session_close phantom")
            return

        if direction is None or self._contract is None:
            print(
                "[broker] Session close: no direction/contract — "
                "falling back to failsafe"
            )
            self._verify_and_failsafe_close("session_close")
            return

        self._cancel_all_mnq("Session close")

        # Re-check: a TP/stop fill may have closed the position while
        # we were cancelling.
        with self._lock:
            if not self._position_open:
                print(
                    "[broker] Position closed before session-close market "
                    "order — skipping (race with TP/stop fill)"
                )
                return

        # Verify IBKR still has a position before submitting close order.
        try:
            if not any(
                p.contract.symbol == MNQ_SYMBOL and p.position != 0
                for p in self._ib.positions()
            ):
                print(
                    "[broker] Session close: _position_open=True but IBKR "
                    "has no MNQ position — clearing flag, sweeping orders"
                )
                with self._lock:
                    self._position_open = False
                    self._pending_entry_fill = None
                stale = self._cancel_all_mnq("Session close no-position sweep")
                if stale:
                    print(f"[broker] Session close: cancelled {stale} stale order(s)")
                return
        except Exception:
            pass  # proceed with close if check fails

        close_action = "SELL" if direction == "up" else "BUY"
        close_order = MarketOrder(close_action, ORDER_QTY)
        parent_id_tag = self._pending_parent_order_id or "unknown"
        close_order.orderRef = f"bot-{parent_id_tag}-close"
        self._pending_exit_reason = "session_close"
        try:
            self._ib.placeOrder(self._contract, close_order)
            print(
                f"[broker] Session close: {close_action} {ORDER_QTY} MNQ @ market"
            )
        except Exception as exc:
            print(f"[broker] Session close order failed: {exc}")

        self._verify_and_failsafe_close("session_close")
