"""BacktestBroker — drop-in replacement for IBKRBroker for backtesting.

Implements the same interface that BotTrader expects, but resolves
trades from historical tick data instead of submitting to IBKR.

This ensures the backtest uses the exact same entry logic as production
(BotTrader.on_tick), eliminating drift between live and backtest code.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot_risk_backtest import evaluate_bot_trade, MULTIPLIER, FEE_PTS
from broker import TradeResult, MNQ_POINT_VALUE


@dataclass
class BacktestTradeRecord:
    """Record of a completed backtest trade."""
    level: str
    direction: str
    entry_count: int
    outcome: str       # win, loss, timeout
    pnl_usd: float
    entry_idx: int
    exit_idx: int
    entry_ns: int


class BacktestBroker:
    """Mock broker that resolves trades from tick data.

    Drop-in replacement for IBKRBroker in BotTrader. The key method is
    submit_bracket(), which evaluates the trade outcome synchronously
    from tick data instead of submitting to IBKR.

    Usage:
        broker = BacktestBroker(full_prices, full_ts_ns, eod_cutoff_ns)
        bot = BotTrader.__new__(BotTrader)
        bot._broker = broker
        # ... register zones, feed ticks ...
    """

    def __init__(
        self,
        full_prices: np.ndarray,
        full_ts_ns: np.ndarray,
        eod_cutoff_ns: int,
        daily_loss: float = 200.0,
        timeout_secs: int = 900,
    ) -> None:
        self._full_prices = full_prices
        self._full_ts_ns = full_ts_ns
        self._eod_cutoff_ns = eod_cutoff_ns
        self._daily_loss = daily_loss
        self._timeout_secs = timeout_secs

        # State matching IBKRBroker's interface
        self._position_open = False
        self._consecutive_losses = 0
        self._daily_pnl_usd = 0.0
        self._trades_today = 0
        self._wins_today = 0
        self._losses_today = 0
        self._stopped_for_day = False
        self._stop_reason = ""
        self._connected = True

        # Track current position for tick-level exit detection
        self._pos_exit_idx = -1

        # Current tick index (updated by simulate loop)
        self._current_tick_idx = 0

        # Completed trades
        self.trades: list[BacktestTradeRecord] = []

        # Pending trade info (for P&L on close)
        self._pending_level: str = ""
        self._pending_direction: str = ""
        self._pending_entry_count: int = 0
        self._pending_entry_idx: int = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def daily_stats(self) -> str:
        return (
            f"{self._trades_today} trades "
            f"({self._wins_today}W/{self._losses_today}L), "
            f"P&L: ${self._daily_pnl_usd:+.2f}"
        )

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def process_events(self) -> None:
        """Check if position should close at current tick."""
        if self._position_open and self._current_tick_idx >= self._pos_exit_idx:
            self._close_position()

    def check_position_timeout(self) -> bool:
        return False  # Timeout handled in evaluate_bot_trade

    def can_trade(self) -> tuple[bool, str]:
        if self._stopped_for_day:
            return False, self._stop_reason
        if self._position_open:
            return False, "Position already open"
        if self._daily_pnl_usd <= -self._daily_loss:
            self._stopped_for_day = True
            self._stop_reason = (
                f"Daily loss limit hit (${self._daily_pnl_usd:+.2f} "
                f">= -${self._daily_loss:.0f})"
            )
            return False, self._stop_reason
        return True, ""

    def submit_bracket(
        self,
        direction: str,
        current_price: float,
        line_price: float,
        level_name: str,
        score: int = 0,
        trend_60m: float = 0.0,
        entry_count: int = 1,
        target_pts: float = 8.0,
        stop_pts: float = 25.0,
        entry_limit_buffer: float = 4.0,
        range_30m: float | None = None,
        tick_rate: float = 0.0,
        session_move_pct: float = 0.0,
    ) -> TradeResult:
        """Evaluate trade from tick data and set position state."""
        entry_idx = self._current_tick_idx

        # Evaluate using the same function as the original backtest
        outcome, exit_idx, pnl_pts = evaluate_bot_trade(
            entry_idx,
            line_price,
            direction,
            self._full_ts_ns,
            self._full_prices,
            target_pts,
            stop_pts,
            self._timeout_secs,
            self._eod_cutoff_ns,
        )

        pnl_usd = pnl_pts * MULTIPLIER

        # Set position as open — BotTrader will skip zones until close
        self._position_open = True
        self._pos_exit_idx = exit_idx

        # Store pending info for close
        self._pending_level = level_name
        self._pending_direction = direction
        self._pending_entry_count = entry_count
        self._pending_entry_idx = entry_idx
        self._pending_outcome = outcome
        self._pending_pnl_usd = pnl_usd

        return TradeResult(
            success=True,
            order_id=entry_idx,
            entry_price=current_price,
        )

    def _close_position(self) -> None:
        """Close the current position and record the trade."""
        self._position_open = False
        self._trades_today += 1
        self._daily_pnl_usd += self._pending_pnl_usd

        if self._pending_pnl_usd >= 0:
            self._wins_today += 1
            self._consecutive_losses = 0
        else:
            self._losses_today += 1
            self._consecutive_losses += 1

        self.trades.append(BacktestTradeRecord(
            level=self._pending_level,
            direction=self._pending_direction,
            entry_count=self._pending_entry_count,
            outcome=self._pending_outcome,
            pnl_usd=self._pending_pnl_usd,
            entry_idx=self._pending_entry_idx,
            exit_idx=self._pos_exit_idx,
            entry_ns=int(self._full_ts_ns[self._pending_entry_idx]),
        ))

    def reset_daily_state(self) -> None:
        self._position_open = False
        self._daily_pnl_usd = 0.0
        self._trades_today = 0
        self._wins_today = 0
        self._losses_today = 0
        self._consecutive_losses = 0
        self._stopped_for_day = False
        self._stop_reason = ""
        self._pos_exit_idx = -1
        self.trades = []

    def eod_flatten(self) -> None:
        if self._position_open:
            self._close_position()

    def session_close(self) -> None:
        self.eod_flatten()
