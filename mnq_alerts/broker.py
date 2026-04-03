"""
broker.py — IBKR order execution with risk management.

Submits bracket orders (market entry + limit target + stop loss) when alerts fire.
Disabled by default — set IBKR_TRADING_ENABLED=true in .env to activate.

Risk controls (validated over 214 days in bot_risk_backtest.py):
  - One position at a time (no stacking)
  - Daily loss limit ($150 default — stops trading for the day)
  - Consecutive loss limit (3 default — stops trading for the day)

Connection: IB Gateway or TWS must be running with API access enabled.
Default port 4002 (IB Gateway paper), 4001 (IB Gateway live).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from config import (
    BOT_STOP_POINTS,
    BOT_TARGET_POINTS,
    DAILY_LOSS_LIMIT_USD,
    IBKR_ACCOUNT,
    IBKR_CLIENT_ID,
    IBKR_HOST,
    IBKR_PORT,
    IBKR_TRADING_ENABLED,
    MAX_CONSECUTIVE_LOSSES,
)

# ib_insync is only imported when trading is enabled to avoid hard dependency.
if IBKR_TRADING_ENABLED:
    from ib_insync import IB, Contract, LimitOrder, MarketOrder, Order, StopOrder


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

        # Risk management state — resets each trading day.
        self._daily_pnl_usd: float = 0.0
        self._consecutive_losses: int = 0
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
        self._pending_target_pts: float = BOT_TARGET_POINTS
        self._pending_stop_pts: float = BOT_STOP_POINTS

    def connect(self) -> bool:
        """Connect to IB Gateway / TWS. Returns True on success."""
        if not IBKR_TRADING_ENABLED:
            print("[broker] Trading disabled (IBKR_TRADING_ENABLED=false)")
            return False
        try:
            self._ib = IB()
            self._ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID)
            self._connected = True

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

            # Pre-qualify the MNQ contract so we don't do it on every order.
            self._contract = self._resolve_contract()
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
                f"{MAX_CONSECUTIVE_LOSSES} consec losses, 1 position at a time"
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

    def process_events(self) -> None:
        """Pump the ib_insync event loop so callbacks fire.

        Must be called periodically from the main loop (e.g. on each
        trade tick). Without this, orderStatusEvent callbacks won't
        dispatch and the broker won't know when positions close.
        """
        if self._ib and self.is_connected:
            try:
                self._ib.sleep(0)  # non-blocking: process pending events
            except Exception:
                pass

    def reset_daily_state(self) -> None:
        """Reset risk counters for a new trading day."""
        with self._lock:
            prev_pnl = self._daily_pnl_usd
            prev_trades = self._trades_today
            self._daily_pnl_usd = 0.0
            self._consecutive_losses = 0
            self._trades_today = 0
            self._wins_today = 0
            self._losses_today = 0
            self._stopped_for_day = False
            self._stop_reason = ""
            self._position_open = False
            self._pending_direction = None
            self._pending_line_price = None
            self._pending_entry_fill = None
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
        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._stopped_for_day = True
            self._stop_reason = (
                f"{self._consecutive_losses} consecutive losses "
                f"(limit: {MAX_CONSECUTIVE_LOSSES})"
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

        Tracks fills on take-profit and stop-loss orders to update
        daily P&L, consecutive losses, and position state.
        """
        if trade.contract.symbol != MNQ_SYMBOL:
            return
        if trade.orderStatus.status != "Filled":
            return

        order = trade.order
        # Child orders close the position; parent opens it.
        if order.parentId == 0:
            # Parent (entry) order filled — record the actual fill price.
            with self._lock:
                self._pending_entry_fill = trade.orderStatus.avgFillPrice
                print(f"[broker] Entry filled @ {self._pending_entry_fill:.2f}")
            return

        with self._lock:
            if not self._position_open:
                return  # Already processed or spurious

            self._position_open = False
            self._trades_today += 1

            # Compute P&L from actual entry fill → exit fill.
            exit_price = trade.orderStatus.avgFillPrice
            entry_price = self._pending_entry_fill or self._pending_line_price
            if self._pending_direction and entry_price is not None:
                if self._pending_direction == "up":
                    pnl_pts = exit_price - entry_price
                else:
                    pnl_pts = entry_price - exit_price
                pnl_usd = pnl_pts * MNQ_POINT_VALUE - 0.54  # subtract round-trip fee
            else:
                # Fallback: infer from order type.
                if order.orderType == "LMT":
                    pnl_pts = self._pending_target_pts
                    pnl_usd = (pnl_pts - 0.27) * MNQ_POINT_VALUE
                else:
                    pnl_pts = -self._pending_stop_pts
                    pnl_usd = -(self._pending_stop_pts + 0.27) * MNQ_POINT_VALUE

            self._daily_pnl_usd += pnl_usd

            if pnl_usd >= 0:
                self._wins_today += 1
                self._consecutive_losses = 0
                outcome = "WIN"
            else:
                self._losses_today += 1
                self._consecutive_losses += 1
                outcome = "LOSS"

            print(
                f"[broker] Trade closed: {outcome} ${pnl_usd:+.2f} | "
                f"Day: {self.daily_stats} | "
                f"Consec losses: {self._consecutive_losses}"
            )

            # Check if we should stop for the day.
            self.can_trade()  # updates _stopped_for_day if limits hit

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
            if resolved.secType not in ("FUT", "CONTFUT"):
                print(
                    f"[broker] WARNING: Unexpected secType after qualifying: "
                    f"{resolved.secType}"
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
    ) -> TradeResult:
        """Submit a bracket order: market entry + limit target + stop loss.

        direction: 'up' → BUY, 'down' → SELL
        current_price: price at alert time (for logging)
        line_price: the level price (target/stop measured from here)
        level_name: e.g. 'IBL', 'VWAP' (for logging)

        Checks risk limits before submitting. Returns TradeResult with
        success=False if limits are exceeded.
        """
        if not IBKR_TRADING_ENABLED:
            return TradeResult(success=False, error="Trading disabled")

        if not self.is_connected:
            print("[broker] Not connected — attempting reconnect...")
            if not self.connect():
                return TradeResult(success=False, error="Not connected to IBKR")

        with self._lock:
            # Check risk limits.
            allowed, reason = self.can_trade()
            if not allowed:
                print(f"[broker] Trade blocked: {reason}")
                return TradeResult(success=False, error=reason)

            return self._submit_bracket_locked(
                direction, current_price, line_price, level_name, qty
            )

    def _submit_bracket_locked(
        self,
        direction: str,
        current_price: float,
        line_price: float,
        level_name: str,
        qty: int,
    ) -> TradeResult:
        """Internal bracket submission (must hold self._lock)."""
        action = "BUY" if direction == "up" else "SELL"
        reverse_action = "SELL" if direction == "up" else "BUY"

        # Target and stop prices based on direction.
        if direction == "up":
            target_price = line_price + BOT_TARGET_POINTS
            stop_price = line_price - BOT_STOP_POINTS
        else:
            target_price = line_price - BOT_TARGET_POINTS
            stop_price = line_price + BOT_STOP_POINTS

        # Round to MNQ tick size (0.25).
        target_price = round(target_price * 4) / 4
        stop_price = round(stop_price * 4) / 4

        contract = self._contract
        if contract is None:
            # Try to re-qualify if we don't have a cached contract.
            contract = self._resolve_contract()
            self._contract = contract
        if contract is None:
            return TradeResult(success=False, error="Contract resolution failed")

        try:
            # Build bracket manually: market entry + limit take-profit + stop loss.
            # Using market order (not limit) for the entry so the bot gets
            # filled immediately — a limit at current_price can miss if the
            # market moves even slightly.
            #
            # Must pre-assign orderId to the parent so children can reference
            # it via parentId BEFORE placeOrder is called.
            parent = MarketOrder(action, qty)
            parent.orderId = self._ib.client.getReqId()
            parent.transmit = False

            take_profit = LimitOrder(reverse_action, qty, target_price)
            take_profit.parentId = parent.orderId
            take_profit.transmit = False

            stop_loss = StopOrder(reverse_action, qty, stop_price)
            stop_loss.parentId = parent.orderId
            stop_loss.transmit = True  # transmit last to send all at once

            # Submit all three legs.
            self._ib.placeOrder(contract, parent)
            self._ib.placeOrder(contract, take_profit)
            self._ib.placeOrder(contract, stop_loss)

            parent_id = parent.orderId

            # Track position state for risk management.
            self._position_open = True
            self._pending_direction = direction
            self._pending_line_price = line_price
            self._pending_target_pts = BOT_TARGET_POINTS
            self._pending_stop_pts = BOT_STOP_POINTS

            print(
                f"[broker] {action} {qty} MNQ @ market | "
                f"target {target_price:.2f} (+{BOT_TARGET_POINTS}) | "
                f"stop {stop_price:.2f} (-{BOT_STOP_POINTS}) | "
                f"level {level_name} @ {line_price:.2f} | "
                f"order {parent_id} | {self.daily_stats}"
            )

            return TradeResult(
                success=True,
                order_id=parent_id,
                entry_price=current_price,
                target_price=target_price,
                stop_price=stop_price,
            )

        except Exception as exc:
            error_msg = f"Order submission failed: {exc}"
            print(f"[broker] {error_msg}")
            return TradeResult(success=False, error=error_msg)

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
        cancelled = 0
        for trade in self._ib.openTrades():
            if trade.contract.symbol == MNQ_SYMBOL:
                self._ib.cancelOrder(trade.order)
                cancelled += 1
        if cancelled:
            print(f"[broker] Cancelled {cancelled} open MNQ orders")
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
                self._ib.placeOrder(pos.contract, order)
                print(f"[broker] Flattening: {action} {qty} MNQ @ market")
        self._position_open = False
        return True
