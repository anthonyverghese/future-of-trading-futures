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


def _simulate_limit_fill(
    full_prices: np.ndarray,
    full_ts_ns: np.ndarray,
    fire_idx: int,
    direction: str,
    line_price: float,
    entry_limit_buffer: float,
    latency_ms: float,
    timeout_secs: float,
) -> tuple[int | None, float | None]:
    """Simulate a limit-order fill from tick data.

    Returns (fill_idx, fill_price). Returns (None, None) if the limit
    is not satisfied within `timeout_secs` of the fire time.

    The bot's limit price is `line_price - buffer` for SELL ("down")
    and `line_price + buffer` for BUY ("up"). Order arrives at the
    matching engine after `latency_ms` of network latency. From there,
    we walk forward and fill at the first tick where the limit is
    satisfied (price >= limit for SELL, <= limit for BUY).
    """
    if direction == "down":
        limit = line_price - entry_limit_buffer
    else:
        limit = line_price + entry_limit_buffer

    fire_ns = int(full_ts_ns[fire_idx])
    start_ns = fire_ns + int(latency_ms * 1_000_000)
    end_ns = fire_ns + int(timeout_secs * 1_000_000_000)

    start_idx = int(np.searchsorted(full_ts_ns, start_ns, side="left"))
    end_idx = int(np.searchsorted(full_ts_ns, end_ns, side="right"))
    end_idx = min(end_idx, len(full_prices))
    if start_idx >= end_idx:
        return None, None

    seg = full_prices[start_idx:end_idx]
    if direction == "down":
        hit = np.where(seg >= limit)[0]
    else:
        hit = np.where(seg <= limit)[0]
    if len(hit) == 0:
        return None, None

    fill_idx = start_idx + int(hit[0])
    return fill_idx, float(full_prices[fill_idx])


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
        simulate_slippage: bool = False,
        latency_ms: float = 100.0,
        fill_timeout_secs: float = 3.0,
    ) -> None:
        self._full_prices = full_prices
        self._full_ts_ns = full_ts_ns
        self._eod_cutoff_ns = eod_cutoff_ns
        self._daily_loss = daily_loss
        self._timeout_secs = timeout_secs
        # Slippage modeling parameters. When `simulate_slippage` is True,
        # submit_bracket runs a limit-fill simulation on the tick data
        # (matching the production limit-order behavior) and computes
        # P&L from the actual fill price instead of `line_price`. The
        # default `False` preserves the historical slippage-blind
        # behavior, so any caller that doesn't opt in is unaffected.
        self._simulate_slippage = simulate_slippage
        self._latency_ms = latency_ms
        self._fill_timeout_secs = fill_timeout_secs

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
        """Evaluate trade from tick data and set position state.

        With `simulate_slippage=True` the broker runs a limit-fill
        simulation: the order is exposed at `line_price ± entry_limit_buffer`
        with `latency_ms` of network latency, and we walk forward up to
        `fill_timeout_secs` looking for the first tick that satisfies
        the limit. Trades that don't fill within that window return
        `TradeResult(success=False)` so the bot can free its cap slot
        and apply its failed-fill cooldown — matching the live behavior.
        """
        fire_idx = self._current_tick_idx

        if self._simulate_slippage and entry_limit_buffer > 0:
            fill_idx, fill_price = _simulate_limit_fill(
                self._full_prices, self._full_ts_ns,
                fire_idx, direction, line_price, entry_limit_buffer,
                self._latency_ms, self._fill_timeout_secs,
            )
            if fill_idx is None:
                return TradeResult(
                    success=False,
                    order_id=fire_idx,
                    entry_price=current_price,
                    error="Limit not filled within timeout",
                )
            entry_idx = fill_idx
            # Slippage-aware: compute P&L from the actual limit-fill price.
            pnl_entry_price = fill_price
            returned_entry_price = fill_price
        else:
            entry_idx = fire_idx
            # Slippage-blind: compute P&L assuming entry at the line, to
            # preserve compatibility with the historical backtest behavior
            # (evaluate_bot_trade returned target_pts - FEE_PTS for wins
            # and -(stop_pts + FEE_PTS) for losses, both implicitly
            # entry-at-line). The TradeResult still reports the fire-tick
            # price to the caller — same as before.
            pnl_entry_price = line_price
            returned_entry_price = current_price

        # Walk forward to determine target/stop/timeout outcome. Target
        # and stop prices are absolute (line ± points), independent of
        # where we filled, so the outcome is determined by the price
        # path between the entry tick (fill_idx in slippage mode) and
        # the trade window's end.
        outcome, exit_idx, _eval_pnl_pts = evaluate_bot_trade(
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

        # Compute P&L from the actual entry price. In slippage-blind
        # mode this equals `_eval_pnl_pts * MULTIPLIER` because the
        # entry price is the line price. In slippage mode the entry
        # price is the limit-fill price and the slippage is reflected
        # in the per-trade P&L.
        if outcome == "win":
            target_price = (line_price - target_pts
                            if direction == "down" else line_price + target_pts)
            raw_pnl_pts = (pnl_entry_price - target_price
                           if direction == "down" else target_price - pnl_entry_price)
        elif outcome == "loss":
            stop_price = (line_price + stop_pts
                          if direction == "down" else line_price - stop_pts)
            raw_pnl_pts = (pnl_entry_price - stop_price
                           if direction == "down" else stop_price - pnl_entry_price)
        else:  # timeout
            exit_price = float(self._full_prices[exit_idx])
            raw_pnl_pts = (pnl_entry_price - exit_price
                           if direction == "down" else exit_price - pnl_entry_price)
        pnl_pts = raw_pnl_pts - FEE_PTS
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
            entry_price=returned_entry_price,
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
