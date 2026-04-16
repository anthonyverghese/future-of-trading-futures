"""Tests for broker module: IBKRBroker risk management and order logic."""

import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub ib_insync before importing broker so it doesn't need the real package.
_ib_mod = MagicMock()
_ib_mod.IB = MagicMock
_ib_mod.Contract = type("Contract", (), {"__init__": lambda self: None})
_ib_mod.MarketOrder = lambda action, qty: SimpleNamespace(
    action=action, totalQuantity=qty, orderId=0, transmit=True, parentId=0
)
_ib_mod.LimitOrder = lambda action, qty, price: SimpleNamespace(
    action=action,
    totalQuantity=qty,
    lmtPrice=price,
    orderId=0,
    transmit=True,
    parentId=0,
)
_ib_mod.StopOrder = lambda action, qty, price: SimpleNamespace(
    action=action,
    totalQuantity=qty,
    auxPrice=price,
    orderId=0,
    transmit=True,
    parentId=0,
)
_ib_mod.Order = MagicMock
sys.modules["ib_insync"] = _ib_mod

import pytest

# Force IBKR_TRADING_ENABLED=true so broker imports ib_insync stubs.
with patch.dict(os.environ, {"IBKR_TRADING_ENABLED": "true"}):
    import importlib
    import config as _cfg

    _cfg.IBKR_TRADING_ENABLED = True
    import broker as _broker_mod

    importlib.reload(_broker_mod)

from broker import IBKRBroker, TradeResult, MNQ_POINT_VALUE, MNQ_SYMBOL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_broker(**overrides) -> IBKRBroker:
    """Create a broker with risk state directly set (no IBKR connection)."""
    b = IBKRBroker()
    for k, v in overrides.items():
        setattr(b, f"_{k}", v)
    return b


def _fake_trade(
    symbol, status, order_type="MKT", parent_id=0, fill_price=20000.0,
    order_ref=None,
):
    """Build a fake ib_insync Trade object for _on_order_status.

    order_ref defaults to "bot-0-parent" for entries (parent_id=0) and
    "bot-0-target" for closes (parent_id!=0), matching the orderRef-based
    dispatch introduced in commit 9834b19.
    """
    if order_ref is None:
        order_ref = "bot-0-parent" if parent_id == 0 else "bot-0-target"
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol),
        orderStatus=SimpleNamespace(status=status, avgFillPrice=fill_price),
        order=SimpleNamespace(
            orderType=order_type, parentId=parent_id, orderRef=order_ref,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Risk gate — can_trade()
# ---------------------------------------------------------------------------


class TestCanTrade:
    def test_allows_trade_when_no_limits_hit(self):
        b = _make_broker()
        allowed, reason = b.can_trade()
        assert allowed is True
        assert reason == ""

    def test_blocks_when_position_open(self):
        b = _make_broker(position_open=True)
        allowed, reason = b.can_trade()
        assert allowed is False
        assert "Position already open" in reason

    def test_blocks_when_stopped_for_day(self):
        b = _make_broker(stopped_for_day=True, stop_reason="test stop")
        allowed, reason = b.can_trade()
        assert allowed is False
        assert reason == "test stop"

    def test_blocks_at_daily_loss_limit(self):
        b = _make_broker(daily_pnl_usd=-150.0)
        allowed, reason = b.can_trade()
        assert allowed is False
        assert "Daily loss limit" in reason
        assert b._stopped_for_day is True

    def test_blocks_at_consecutive_loss_limit(self):
        from config import MAX_CONSECUTIVE_LOSSES

        b = _make_broker(consecutive_losses=MAX_CONSECUTIVE_LOSSES)
        allowed, reason = b.can_trade()
        assert allowed is False
        assert "consecutive losses" in reason
        assert b._stopped_for_day is True

    def test_allows_just_under_daily_limit(self):
        b = _make_broker(daily_pnl_usd=-149.99)
        allowed, _ = b.can_trade()
        assert allowed is True

    def test_allows_at_two_consecutive_losses(self):
        b = _make_broker(consecutive_losses=2)
        allowed, _ = b.can_trade()
        assert allowed is True


# ---------------------------------------------------------------------------
# 2. Daily reset
# ---------------------------------------------------------------------------


class TestResetDailyState:
    def test_clears_all_counters(self):
        b = _make_broker(
            daily_pnl_usd=-100.0,
            consecutive_losses=2,
            trades_today=5,
            wins_today=3,
            losses_today=2,
            stopped_for_day=True,
            stop_reason="limit hit",
            position_open=True,
            pending_direction="up",
            pending_line_price=20000.0,
            pending_entry_fill=20001.0,
        )
        b.reset_daily_state()

        assert b._daily_pnl_usd == 0.0
        assert b._consecutive_losses == 0
        assert b._trades_today == 0
        assert b._wins_today == 0
        assert b._losses_today == 0
        assert b._stopped_for_day is False
        assert b._stop_reason == ""
        assert b._position_open is False
        assert b._pending_direction is None
        assert b._pending_line_price is None
        assert b._pending_entry_fill is None

    def test_can_trade_after_reset(self):
        b = _make_broker(stopped_for_day=True, consecutive_losses=3)
        b.reset_daily_state()
        allowed, _ = b.can_trade()
        assert allowed is True


# ---------------------------------------------------------------------------
# 3. Order status callback — _on_order_status
# ---------------------------------------------------------------------------


class TestOnOrderStatus:
    def test_ignores_non_mnq_symbols(self):
        b = _make_broker(position_open=True)
        trade = _fake_trade("ES", "Filled")
        b._on_order_status(trade)
        assert b._trades_today == 0  # not processed

    def test_ignores_non_filled_status(self):
        b = _make_broker(position_open=True)
        trade = _fake_trade(MNQ_SYMBOL, "Submitted")
        b._on_order_status(trade)
        assert b._trades_today == 0

    def test_parent_fill_records_entry_price(self):
        b = _make_broker(position_open=True)
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=0, fill_price=20005.0)
        b._on_order_status(trade)
        assert b._pending_entry_fill == 20005.0
        assert b._position_open is True  # parent doesn't close position

    def test_child_fill_win_up_direction(self):
        """Take-profit fill on a long (up) trade → win."""
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_entry_fill=20000.0,
            pending_line_price=20000.0,
        )
        # Exit at 20012 → +12 pts → $24 - $0.54 fee = $23.46
        trade = _fake_trade(
            MNQ_SYMBOL, "Filled", order_type="LMT", parent_id=100, fill_price=20012.0
        )
        b._on_order_status(trade)

        assert b._position_open is False
        assert b._trades_today == 1
        assert b._wins_today == 1
        assert b._losses_today == 0
        assert b._consecutive_losses == 0
        assert b._daily_pnl_usd == pytest.approx(12.0 * MNQ_POINT_VALUE - 1.24)

    def test_child_fill_loss_up_direction(self):
        """Stop-loss fill on a long (up) trade → loss."""
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_entry_fill=20000.0,
            pending_line_price=20000.0,
        )
        # Exit at 19975 → -25 pts → -$50 - $0.54 fee = -$50.54
        trade = _fake_trade(
            MNQ_SYMBOL, "Filled", order_type="STP", parent_id=100, fill_price=19975.0
        )
        b._on_order_status(trade)

        assert b._position_open is False
        assert b._losses_today == 1
        assert b._consecutive_losses == 1
        assert b._daily_pnl_usd == pytest.approx(-25.0 * MNQ_POINT_VALUE - 1.24)

    def test_child_fill_win_down_direction(self):
        """Take-profit fill on a short (down) trade → win."""
        b = _make_broker(
            position_open=True,
            pending_direction="down",
            pending_entry_fill=20000.0,
            pending_line_price=20000.0,
        )
        # Exit at 19988 → +12 pts (short) → $24 - $0.54 = $23.46
        trade = _fake_trade(
            MNQ_SYMBOL, "Filled", order_type="LMT", parent_id=100, fill_price=19988.0
        )
        b._on_order_status(trade)

        assert b._wins_today == 1
        assert b._daily_pnl_usd == pytest.approx(12.0 * MNQ_POINT_VALUE - 1.24)

    def test_child_fill_loss_down_direction(self):
        """Stop-loss fill on a short (down) trade → loss."""
        b = _make_broker(
            position_open=True,
            pending_direction="down",
            pending_entry_fill=20000.0,
            pending_line_price=20000.0,
        )
        # Exit at 20025 → -25 pts (short)
        trade = _fake_trade(
            MNQ_SYMBOL, "Filled", order_type="STP", parent_id=100, fill_price=20025.0
        )
        b._on_order_status(trade)

        assert b._losses_today == 1
        assert b._daily_pnl_usd == pytest.approx(-25.0 * MNQ_POINT_VALUE - 1.24)

    def test_consecutive_losses_accumulate(self):
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_entry_fill=20000.0,
        )
        # First loss
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=100, fill_price=19975.0)
        b._on_order_status(trade)
        assert b._consecutive_losses == 1

        # Second loss — re-open position
        b._position_open = True
        b._pending_direction = "up"
        b._pending_entry_fill = 20000.0
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=101, fill_price=19975.0)
        b._on_order_status(trade)
        assert b._consecutive_losses == 2

    def test_win_resets_consecutive_losses(self):
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_entry_fill=20000.0,
            consecutive_losses=2,
        )
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=100, fill_price=20012.0)
        b._on_order_status(trade)
        assert b._consecutive_losses == 0

    def test_ignores_child_fill_when_no_position(self):
        """Spurious fill after position already closed — should be ignored."""
        b = _make_broker(position_open=False)
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=100, fill_price=20012.0)
        b._on_order_status(trade)
        assert b._trades_today == 0

    def test_stops_for_day_after_hitting_loss_limit(self):
        """After enough losses, can_trade() triggered from callback stops the day."""
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_entry_fill=20000.0,
            daily_pnl_usd=-100.0,  # already down $100
        )
        # Loss of $50.54 → total -$150.54, exceeds $150 limit
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=100, fill_price=19975.0)
        b._on_order_status(trade)

        # The callback calls can_trade() which sets _stopped_for_day
        assert b._stopped_for_day is True


# ---------------------------------------------------------------------------
# 4. Bracket order price calculation
# ---------------------------------------------------------------------------


class TestBracketPrices:
    def test_up_direction_prices(self):
        """BUY bracket: target above, stop below line price."""
        b = _make_broker()
        b._connected = True
        b._contract = SimpleNamespace(symbol=MNQ_SYMBOL, secType="FUT")

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.client.getReqId.return_value = 42
        b._ib = mock_ib

        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("up", 20003.0, 20000.0, "IBL")

        assert result.success is True
        assert result.target_price == 20010.0  # 20000 + 10
        assert result.stop_price == 19975.0  # 20000 - 25
        assert result.order_id == 42

    def test_down_direction_prices(self):
        """SELL bracket: target below, stop above line price."""
        b = _make_broker()
        b._connected = True
        b._contract = SimpleNamespace(symbol=MNQ_SYMBOL, secType="FUT")

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.client.getReqId.return_value = 43
        b._ib = mock_ib

        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("down", 19997.0, 20000.0, "IBH")

        assert result.success is True
        assert result.target_price == 19990.0  # 20000 - 10
        assert result.stop_price == 20025.0  # 20000 + 25
        assert result.order_id == 43

    def test_tick_rounding(self):
        """Prices round to MNQ tick size (0.25)."""
        b = _make_broker()
        b._connected = True
        b._contract = SimpleNamespace(symbol=MNQ_SYMBOL, secType="FUT")

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.client.getReqId.return_value = 44
        b._ib = mock_ib

        # Line at 20001.33 → target 20011.33 → rounds to 20011.25
        #                   → stop 19976.33 → rounds to 19976.25
        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("up", 20002.0, 20001.33, "VWAP")

        assert result.target_price == 20011.25
        assert result.stop_price == 19976.25

    def test_sets_position_tracking_state(self):
        """After successful bracket, position tracking state is set."""
        b = _make_broker()
        b._connected = True
        b._contract = SimpleNamespace(symbol=MNQ_SYMBOL, secType="FUT")

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.client.getReqId.return_value = 45
        b._ib = mock_ib

        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            b.submit_bracket("up", 20003.0, 20000.0, "IBL")

        assert b._position_open is True
        assert b._pending_direction == "up"
        assert b._pending_line_price == 20000.0

    def test_blocked_when_position_open(self):
        """Cannot submit when a position is already open."""
        b = _make_broker(position_open=True)
        b._connected = True

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        # Pre-entry drift check queries positions — must return a matching
        # MNQ position so the flag isn't cleared by "safe drift" logic.
        mock_ib.positions.return_value = [
            SimpleNamespace(
                contract=SimpleNamespace(symbol=MNQ_SYMBOL),
                position=1,
            )
        ]
        b._ib = mock_ib

        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("up", 20003.0, 20000.0, "IBL")

        assert result.success is False
        assert "Position already open" in result.error


# ---------------------------------------------------------------------------
# 5. Daily stats string
# ---------------------------------------------------------------------------


class TestDailyStats:
    def test_initial_stats(self):
        b = _make_broker()
        assert b.daily_stats == "0 trades (0W/0L), P&L: $+0.00"

    def test_after_trades(self):
        b = _make_broker(
            trades_today=3, wins_today=2, losses_today=1, daily_pnl_usd=-3.62
        )
        assert "3 trades" in b.daily_stats
        assert "2W/1L" in b.daily_stats
        assert "$-3.62" in b.daily_stats


# ---------------------------------------------------------------------------
# 6. Submit bracket without connection
# ---------------------------------------------------------------------------


class TestSubmitWithoutConnection:
    def test_returns_error_when_trading_disabled(self):
        with patch.object(_broker_mod, "IBKR_TRADING_ENABLED", False):
            b = _make_broker()
            result = b.submit_bracket("up", 20000.0, 20000.0, "IBL")
            assert result.success is False
            assert "disabled" in result.error.lower()

    def test_returns_error_when_contract_missing(self):
        b = _make_broker(contract=None)
        b._connected = True

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.client.getReqId.return_value = 50
        mock_ib.qualifyContracts.return_value = []  # contract resolution fails
        b._ib = mock_ib

        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("up", 20000.0, 20000.0, "IBL")

        assert result.success is False
        assert "Contract" in result.error


# ---------------------------------------------------------------------------
# 7. P&L uses actual fill price, not line price
# ---------------------------------------------------------------------------


class TestPnlFromActualFill:
    def test_pnl_from_entry_fill_not_line_price(self):
        """P&L should be computed from actual entry fill, not line_price."""
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_line_price=20000.0,
            pending_entry_fill=20002.0,  # filled 2 pts above line
        )
        # Exit at 20012 → profit = 20012 - 20002 = 10 pts (not 12)
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=100, fill_price=20012.0)
        b._on_order_status(trade)

        expected_pnl = 10.0 * MNQ_POINT_VALUE - 1.24
        assert b._daily_pnl_usd == pytest.approx(expected_pnl)

    def test_pnl_falls_back_to_line_price_without_fill(self):
        """If entry fill wasn't recorded, fall back to line_price."""
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_line_price=20000.0,
            pending_entry_fill=None,  # no fill recorded
        )
        trade = _fake_trade(MNQ_SYMBOL, "Filled", parent_id=100, fill_price=20012.0)
        b._on_order_status(trade)

        expected_pnl = 12.0 * MNQ_POINT_VALUE - 1.24
        assert b._daily_pnl_usd == pytest.approx(expected_pnl)


# ---------------------------------------------------------------------------
# Helpers for orderRef-based tests
# ---------------------------------------------------------------------------


def _fake_trade_with_ref(symbol, status, order_ref, fill_price=20000.0, order_type="MKT"):
    """Build a fake Trade with an orderRef tag (for dispatch tests)."""
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol),
        orderStatus=SimpleNamespace(status=status, avgFillPrice=fill_price),
        order=SimpleNamespace(orderType=order_type, parentId=0, orderRef=order_ref),
    )


# ---------------------------------------------------------------------------
# 8. OrderRef-based dispatch — _on_order_status
# ---------------------------------------------------------------------------


class TestOrderRefDispatch:
    def test_parent_role_logs_entry(self):
        """Parent fill (orderRef=bot-30-parent) records entry price."""
        b = _make_broker(position_open=True)
        trade = _fake_trade_with_ref(MNQ_SYMBOL, "Filled", "bot-30-parent", fill_price=20005.0)
        b._on_order_status(trade)
        assert b._pending_entry_fill == 20005.0

    def test_target_role_closes_position(self):
        """Target fill (orderRef=bot-30-target) closes position and computes P&L."""
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_entry_fill=20000.0,
            pending_line_price=20000.0,
        )
        trade = _fake_trade_with_ref(
            MNQ_SYMBOL, "Filled", "bot-30-target", fill_price=20012.0, order_type="LMT"
        )
        b._on_order_status(trade)
        assert b._position_open is False
        assert b._trades_today == 1
        assert b._daily_pnl_usd == pytest.approx(12.0 * MNQ_POINT_VALUE - 1.24)

    def test_stop_role_closes_position(self):
        """Stop fill (orderRef=bot-30-stop) closes position and computes P&L."""
        b = _make_broker(
            position_open=True,
            pending_direction="up",
            pending_entry_fill=20000.0,
            pending_line_price=20000.0,
        )
        trade = _fake_trade_with_ref(
            MNQ_SYMBOL, "Filled", "bot-30-stop", fill_price=19975.0, order_type="STP"
        )
        b._on_order_status(trade)
        assert b._position_open is False
        assert b._trades_today == 1
        assert b._daily_pnl_usd == pytest.approx(-25.0 * MNQ_POINT_VALUE - 1.24)

    def test_close_role_closes_position(self):
        """Manual close (orderRef=bot-30-close) closes position."""
        b = _make_broker(
            position_open=True,
            pending_direction="down",
            pending_entry_fill=20000.0,
            pending_line_price=20000.0,
        )
        trade = _fake_trade_with_ref(
            MNQ_SYMBOL, "Filled", "bot-30-close", fill_price=19990.0
        )
        b._on_order_status(trade)
        assert b._position_open is False
        assert b._trades_today == 1

    def test_defensive_flatten_ignored(self):
        """Defensive flatten (orderRef=bot-defensive-flatten) is untracked."""
        b = _make_broker(position_open=True)
        trade = _fake_trade_with_ref(
            MNQ_SYMBOL, "Filled", "bot-defensive-flatten", fill_price=20000.0
        )
        b._on_order_status(trade)
        assert b._position_open is True
        assert b._trades_today == 0

    def test_unknown_role_ignored(self):
        """Unknown role (orderRef=bot-30-unknown) is ignored."""
        b = _make_broker(position_open=True)
        trade = _fake_trade_with_ref(
            MNQ_SYMBOL, "Filled", "bot-30-unknown", fill_price=20000.0
        )
        b._on_order_status(trade)
        assert b._position_open is True
        assert b._trades_today == 0

    def test_no_orderref_ignored(self):
        """Empty or None orderRef (not a bot order) is ignored."""
        b = _make_broker(position_open=True)
        # Empty string
        trade = _fake_trade_with_ref(MNQ_SYMBOL, "Filled", "", fill_price=20000.0)
        b._on_order_status(trade)
        assert b._position_open is True
        assert b._trades_today == 0
        # None
        trade = _fake_trade_with_ref(MNQ_SYMBOL, "Filled", None, fill_price=20000.0)
        b._on_order_status(trade)
        assert b._position_open is True
        assert b._trades_today == 0

    def test_non_mnq_ignored(self):
        """Non-MNQ symbol is ignored regardless of orderRef."""
        b = _make_broker(position_open=True)
        trade = _fake_trade_with_ref(
            "ES", "Filled", "bot-30-target", fill_price=20000.0
        )
        b._on_order_status(trade)
        assert b._position_open is True
        assert b._trades_today == 0

    def test_non_filled_ignored(self):
        """Non-Filled status is ignored."""
        b = _make_broker(position_open=True)
        trade = _fake_trade_with_ref(
            MNQ_SYMBOL, "Submitted", "bot-30-target", fill_price=20000.0
        )
        b._on_order_status(trade)
        assert b._position_open is True
        assert b._trades_today == 0


# ---------------------------------------------------------------------------
# 9. Pre-entry drift check in submit_bracket()
# ---------------------------------------------------------------------------


class TestPreEntryDriftCheck:
    def _make_connected_broker(self, **overrides):
        """Create a broker wired up for submit_bracket() calls."""
        b = _make_broker(**overrides)
        b._connected = True
        b._was_connected = True
        b._connect_attempted = True
        b._contract = SimpleNamespace(symbol=MNQ_SYMBOL, secType="FUT")
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.client.getReqId.return_value = 99
        b._ib = mock_ib
        return b

    def test_dangerous_drift_refuses_entry(self):
        """IBKR has MNQ position but broker thinks none — refuse entry."""
        b = self._make_connected_broker(position_open=False)
        b._ib.positions.return_value = [
            SimpleNamespace(
                contract=SimpleNamespace(symbol=MNQ_SYMBOL),
                position=1,
            )
        ]
        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("up", 20003.0, 20000.0, "IBL")
        assert result.success is False
        assert "drift" in result.error.lower()

    def test_safe_drift_clears_flag(self):
        """Broker thinks position open but IBKR has none — clear and proceed."""
        b = self._make_connected_broker(position_open=True)
        b._ib.positions.return_value = []
        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("up", 20003.0, 20000.0, "IBL")
        assert b._position_open is True  # re-set by submit_bracket after clearing
        assert result.success is True

    def test_no_drift_proceeds(self):
        """No position on either side — entry proceeds normally."""
        b = self._make_connected_broker(position_open=False)
        b._ib.positions.return_value = []
        with patch.object(
            type(b), "is_connected", new_callable=lambda: property(lambda self: True)
        ):
            result = b.submit_bracket("up", 20003.0, 20000.0, "IBL")
        assert result.success is True


# ---------------------------------------------------------------------------
# 10. Restore daily state from DB — _restore_daily_state()
# ---------------------------------------------------------------------------


class TestRestoreDailyState:
    @patch("broker.load_bot_daily_risk_state")
    def test_restores_counters_from_db(self, mock_load):
        mock_load.return_value = {
            "pnl_usd": -50.0,
            "trades": 3,
            "wins": 2,
            "losses": 1,
            "consecutive_losses": 1,
        }
        b = _make_broker()
        b._restore_daily_state()
        assert b._daily_pnl_usd == -50.0
        assert b._trades_today == 3
        assert b._wins_today == 2
        assert b._losses_today == 1
        assert b._consecutive_losses == 1
        assert b._stopped_for_day is False

    @patch("broker.load_bot_daily_risk_state")
    def test_sets_stopped_when_loss_limit_hit(self, mock_load):
        mock_load.return_value = {
            "pnl_usd": -160.0,
            "trades": 4,
            "wins": 1,
            "losses": 3,
            "consecutive_losses": 2,
        }
        b = _make_broker()
        b._restore_daily_state()
        assert b._stopped_for_day is True
        assert "loss limit" in b._stop_reason.lower()

    @patch("broker.load_bot_daily_risk_state")
    def test_sets_stopped_when_consec_losses_hit(self, mock_load):
        mock_load.return_value = {
            "pnl_usd": -80.0,
            "trades": 5,
            "wins": 0,
            "losses": 5,
            "consecutive_losses": 5,
        }
        b = _make_broker()
        b._restore_daily_state()
        assert b._stopped_for_day is True
        assert "consecutive losses" in b._stop_reason.lower()

    @patch("broker.load_bot_daily_risk_state")
    def test_noop_when_no_trades(self, mock_load):
        mock_load.return_value = {
            "pnl_usd": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "consecutive_losses": 0,
        }
        b = _make_broker()
        b._restore_daily_state()
        assert b._daily_pnl_usd == 0.0
        assert b._trades_today == 0
        assert b._stopped_for_day is False


# ---------------------------------------------------------------------------
# 11. Heartbeat in process_events()
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_heartbeat_prints_when_interval_elapsed(self, capsys):
        """Heartbeat fires when enough time has passed since last one."""
        b = _make_broker()
        b._connect_attempted = True
        b._connected = True
        b._was_connected = True
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        b._ib = mock_ib
        # Guarantee heartbeat fires by zeroing the interval.
        b._heartbeat_interval_secs = 0
        b._last_heartbeat_time = 0.0
        b.process_events()
        captured = capsys.readouterr()
        assert "[broker] heartbeat:" in captured.out

    def test_heartbeat_skipped_when_recent(self, capsys):
        """No heartbeat when last one was very recent."""
        b = _make_broker()
        b._connect_attempted = True
        b._connected = True
        b._was_connected = True
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        b._ib = mock_ib
        # Set last heartbeat to now so interval hasn't elapsed.
        import time as _time

        b._last_heartbeat_time = _time.monotonic()
        b.process_events()
        captured = capsys.readouterr()
        assert "[broker] heartbeat:" not in captured.out
