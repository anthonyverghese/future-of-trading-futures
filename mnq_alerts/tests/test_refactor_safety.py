"""Refactor safety tests — must pass before AND after refactoring.

These tests exercise integration between components and key behaviors
that are likely to break during structural refactoring. They serve as
a safety net: if all these pass after a refactor, the refactor is
functionally equivalent.

Categories:
1. BotTrader on_tick flow: zone entry → filters → broker submission
2. Broker risk management: loss limits, position tracking, P&L
3. Broker order lifecycle: submit → fill → close → state cleanup
4. Bot + Broker integration: trade close resets zones, cooldown fires
5. Human alert flow: zone entry → scoring → notification
6. Backtest simulation: simulate_day produces consistent results
"""

import sys
import os
import datetime
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytz
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub ib_insync before importing broker/bot_trader.
_ib_mod = MagicMock()
_ib_mod.IB = MagicMock
_ib_mod.Contract = type("Contract", (), {"__init__": lambda self: None})
_ib_mod.MarketOrder = lambda action, qty: SimpleNamespace(
    action=action, totalQuantity=qty, orderId=0, transmit=True, parentId=0
)
_ib_mod.LimitOrder = lambda action, qty, price: SimpleNamespace(
    action=action, totalQuantity=qty, lmtPrice=price,
    orderId=0, transmit=True, parentId=0,
)
_ib_mod.StopOrder = lambda action, qty, price: SimpleNamespace(
    action=action, totalQuantity=qty, auxPrice=price,
    orderId=0, transmit=True, parentId=0,
)
_ib_mod.Order = MagicMock
sys.modules["ib_insync"] = _ib_mod

import importlib

with patch.dict(os.environ, {"IBKR_TRADING_ENABLED": "true"}):
    import config as _cfg
    _cfg.IBKR_TRADING_ENABLED = True
    import broker as _broker_mod
    importlib.reload(_broker_mod)
    import bot_trader as _bt_mod
    importlib.reload(_bt_mod)

from bot_trader import BotZone, BotTrader, bot_entry_score
from broker import IBKRBroker, MNQ_POINT_VALUE, MNQ_SYMBOL
from config import BOT_ENTRY_THRESHOLD, BOT_PER_LEVEL_TS, BOT_PER_LEVEL_MAX_ENTRIES

_ET = pytz.timezone("America/New_York")


def _make_broker():
    """Create a broker with stubbed internals for testing."""
    b = IBKRBroker.__new__(IBKRBroker)
    b._ib = None
    b._contract = None
    b._connected = False
    b._connect_attempted = False
    b._was_connected = False
    b._position_open = False
    b._position_opened_at = None
    b._pending_entry_fill = None
    b._pending_direction = None
    b._pending_line_price = None
    b._pending_target_price = None
    b._pending_stop_price = None
    b._pending_level_name = None
    b._pending_score = None
    b._pending_trend_60m = None
    b._pending_entry_count = None
    b._pending_parent_order_id = None
    b._pending_db_trade_id = None
    b._pending_range_30m = None
    b._pending_tick_rate = None
    b._pending_session_move_pct = None
    b._pending_entry_limit = None
    b._daily_pnl_usd = 0.0
    b._trades_today = 0
    b._wins_today = 0
    b._losses_today = 0
    b._consecutive_losses = 0
    b._stopped_for_day = False
    b._stop_reason = ""
    b._last_heartbeat_time = 0.0
    b._heartbeat_interval_secs = 300
    b._reconnect_attempts = 0
    b._max_reconnect_attempts = 5
    b._last_reconnect_time = 0.0
    b._reconnect_interval_secs = 60
    import threading
    b._lock = threading.Lock()
    return b


def _make_bot_trader():
    """Create a BotTrader with stubbed broker for testing."""
    bt = BotTrader.__new__(BotTrader)
    bt._broker = MagicMock()
    bt._broker.is_connected = True
    bt._broker._position_open = False
    bt._broker._consecutive_losses = 0
    bt._zones = {}
    bt._price_window = deque()
    bt._price_window_5m = deque()
    bt._price_5m_ago = None
    bt._level_trade_counts = {}
    bt._active_trade_level = None
    bt._level_cooldown_until = {}
    bt._global_cooldown_until = 0.0
    bt._adaptive_caps_restored = True
    bt._adaptive_caps_until_et_mins = 0
    bt._adaptive_accepted_trades = 0
    bt._adaptive_any_loss = False
    return bt


# ---------------------------------------------------------------------------
# 1. BotTrader on_tick flow
# ---------------------------------------------------------------------------


class TestBotTraderOnTickFlow:
    """Tests that on_tick correctly chains: zone → filters → broker."""

    def test_zone_entry_calls_submit_bracket(self):
        """When zone fires and all filters pass, submit_bracket is called."""
        bt = _make_bot_trader()
        bt._zones = {"FIB_0.236": BotZone("FIB_0.236", 27800.0)}
        bt._broker.can_trade.return_value = (True, "")
        bt._broker.submit_bracket.return_value = SimpleNamespace(
            success=True, order_id=1, entry_price=27800.0
        )

        now_et = datetime.time(11, 0)
        bt.on_tick(27800.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_called_once()
        call_kwargs = bt._broker.submit_bracket.call_args
        assert call_kwargs[1]["level_name"] == "FIB_0.236"

    def test_ibh_buy_blocked_by_direction_filter(self):
        """IBH BUY should be blocked — only SELL allowed."""
        bt = _make_bot_trader()
        bt._zones = {"IBH": BotZone("IBH", 27900.0)}
        bt._broker.can_trade.return_value = (True, "")

        # Price above IBH → direction = "up" (BUY) → should be blocked
        now_et = datetime.time(11, 0)
        bt.on_tick(27900.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_not_called()

    def test_ibh_sell_allowed(self):
        """IBH SELL should pass the direction filter."""
        bt = _make_bot_trader()
        bt._zones = {"IBH": BotZone("IBH", 27900.0)}
        bt._broker.can_trade.return_value = (True, "")
        bt._broker.submit_bracket.return_value = SimpleNamespace(
            success=True, order_id=1, entry_price=27899.5
        )

        # Price below IBH → direction = "down" (SELL)
        now_et = datetime.time(11, 0)
        bt.on_tick(27899.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_called_once()

    def test_momentum_filter_blocks_with_momentum(self):
        """Entry should be blocked when 5-min momentum > 5pts with direction."""
        bt = _make_bot_trader()
        bt._zones = {"FIB_0.236": BotZone("FIB_0.236", 27800.0)}
        bt._broker.can_trade.return_value = (True, "")

        # Set price_5m_ago so momentum = 27800.5 - 27790 = +10.5 for BUY
        bt._price_5m_ago = 27790.0

        now_et = datetime.time(11, 0)
        bt.on_tick(27800.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_not_called()

    def test_momentum_filter_allows_against_momentum(self):
        """Entry should be allowed when momentum is against trade direction."""
        bt = _make_bot_trader()
        bt._zones = {"FIB_0.236": BotZone("FIB_0.236", 27800.0)}
        bt._broker.can_trade.return_value = (True, "")
        bt._broker.submit_bracket.return_value = SimpleNamespace(
            success=True, order_id=1, entry_price=27800.5
        )

        # Price fell to level (against momentum for BUY)
        bt._price_5m_ago = 27810.0

        now_et = datetime.time(11, 0)
        bt.on_tick(27800.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_called_once()

    def test_per_level_cap_blocks_at_limit(self):
        """Entry should be blocked when per-level cap is reached."""
        bt = _make_bot_trader()
        bt._zones = {"FIB_0.618": BotZone("FIB_0.618", 27850.0)}
        bt._broker.can_trade.return_value = (True, "")
        # FIB_0.618 cap is 3
        bt._level_trade_counts = {"FIB_0.618": 3}

        now_et = datetime.time(11, 0)
        bt.on_tick(27850.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_not_called()

    def test_position_open_skips_all_zones(self):
        """When position is open, no zones should be processed."""
        bt = _make_bot_trader()
        bt._broker._position_open = True
        bt._zones = {"FIB_0.236": BotZone("FIB_0.236", 27800.0)}

        now_et = datetime.time(11, 0)
        bt.on_tick(27800.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_not_called()

    def test_daily_loss_limit_skips_all_zones(self):
        """When daily loss limit hit, no zones should be processed."""
        bt = _make_bot_trader()
        bt._broker.can_trade.return_value = (False, "Daily loss limit hit")
        bt._zones = {"FIB_0.236": BotZone("FIB_0.236", 27800.0)}

        now_et = datetime.time(11, 0)
        bt.on_tick(27800.5, tick_rate=1500, now_et=now_et)

        bt._broker.submit_bracket.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Trade close resets all zones
# ---------------------------------------------------------------------------


class TestTradeCloseResetsZones:
    """Tests that when a trade closes, ALL zones reset."""

    def test_trade_close_resets_all_zones(self):
        """When position closes, all zones should reset to in_zone=False."""
        bt = _make_bot_trader()
        bt._active_trade_level = "FIB_0.236"
        bt._broker._position_open = False
        bt._broker._consecutive_losses = 0
        bt._broker.can_trade.return_value = (True, "")

        z1 = BotZone("FIB_0.236", 27800.0)
        z2 = BotZone("FIB_0.764", 27850.0)
        z1.in_zone = True
        z2.in_zone = True
        bt._zones = {"FIB_0.236": z1, "FIB_0.764": z2}

        now_et = datetime.time(11, 0)
        bt.on_tick(27820.0, tick_rate=1500, now_et=now_et)

        assert z1.in_zone is False
        assert z2.in_zone is False
        assert bt._active_trade_level is None

    def test_trade_close_loss_sets_no_cooldown(self):
        """After a loss, global cooldown should NOT be set (cooldown=0)."""
        bt = _make_bot_trader()
        bt._active_trade_level = "FIB_0.236"
        bt._broker._position_open = False
        bt._broker._consecutive_losses = 1  # had a loss
        bt._broker.can_trade.return_value = (True, "")

        bt._zones = {"FIB_0.236": BotZone("FIB_0.236", 27800.0)}

        now_et = datetime.time(11, 0)
        bt.on_tick(27820.0, tick_rate=1500, now_et=now_et)

        # Cooldown is 0 (disabled)
        assert bt._global_cooldown_until == 0.0


# ---------------------------------------------------------------------------
# 3. Broker risk management
# ---------------------------------------------------------------------------


class TestBrokerRiskManagement:
    """Tests that broker risk checks work correctly."""

    def test_can_trade_allows_when_clear(self):
        """can_trade returns True when no limits hit."""
        b = _make_broker()
        allowed, reason = b.can_trade()
        assert allowed is True

    def test_can_trade_blocks_when_position_open(self):
        """can_trade returns False when position is open."""
        b = _make_broker()
        b._position_open = True
        allowed, reason = b.can_trade()
        assert allowed is False
        assert "Position" in reason

    def test_can_trade_blocks_at_loss_limit(self):
        """can_trade returns False when daily loss limit hit."""
        b = _make_broker()
        b._daily_pnl_usd = -200.0  # at $200 limit
        allowed, reason = b.can_trade()
        assert allowed is False
        assert b._stopped_for_day is True

    def test_can_trade_allows_below_loss_limit(self):
        """can_trade allows when loss is below limit."""
        b = _make_broker()
        b._daily_pnl_usd = -150.0  # below $200 limit
        allowed, reason = b.can_trade()
        assert allowed is True

    def test_stopped_for_day_persists(self):
        """Once stopped, can_trade stays False even if P&L recovers."""
        b = _make_broker()
        b._stopped_for_day = True
        b._stop_reason = "Daily loss limit"
        b._daily_pnl_usd = 50.0  # hypothetically recovered
        allowed, reason = b.can_trade()
        assert allowed is False

    def test_reset_daily_state_clears_everything(self):
        """reset_daily_state clears all risk counters."""
        b = _make_broker()
        b._daily_pnl_usd = -150.0
        b._trades_today = 10
        b._wins_today = 5
        b._losses_today = 5
        b._consecutive_losses = 3
        b._stopped_for_day = True
        b._stop_reason = "test"

        b.reset_daily_state()

        assert b._daily_pnl_usd == 0.0
        assert b._trades_today == 0
        assert b._wins_today == 0
        assert b._losses_today == 0
        assert b._consecutive_losses == 0
        assert b._stopped_for_day is False


# ---------------------------------------------------------------------------
# 4. BotTrader reset_daily_state
# ---------------------------------------------------------------------------


class TestBotTraderResetDaily:
    """Tests that daily reset clears all state."""

    def test_reset_clears_zones(self):
        bt = _make_bot_trader()
        bt._zones = {"FIB_0.236": BotZone("FIB_0.236", 27800.0)}
        bt._level_trade_counts = {"FIB_0.236": 5}
        bt._price_window.append((datetime.datetime.now(_ET), 27800.0))
        bt._price_window_5m.append((datetime.datetime.now(_ET), 27800.0))
        bt._price_5m_ago = 27790.0
        bt._active_trade_level = "FIB_0.236"

        bt.reset_daily_state()

        assert len(bt._zones) == 0
        assert len(bt._level_trade_counts) == 0
        assert len(bt._price_window) == 0
        assert len(bt._price_window_5m) == 0
        assert bt._price_5m_ago is None
        assert bt._active_trade_level is None


# ---------------------------------------------------------------------------
# 5. Human alert scoring
# ---------------------------------------------------------------------------


class TestHumanAlertScoring:
    """Tests that human scoring produces correct scores."""

    def test_composite_score_basic(self):
        from scoring import composite_score
        # FIB_EXT_HI_1.272 (+2 level), up (+2 combo), power hour (+2)
        score = composite_score(
            "FIB_EXT_HI_1.272", entry_count=2, now_et=datetime.time(15, 30),
            tick_rate=1500, session_move_pts=15.0, direction="up",
        )
        # level=2 + combo=2 + time=2 + test(#2)=1 + move(10-20)=2 = 9
        assert score == 9

    def test_composite_score_suppressed_window(self):
        from scoring import SUPPRESSED_WINDOWS
        # 13:30-14:00 should be in suppressed windows
        assert any(810 <= 820 < we for ws, we in SUPPRESSED_WINDOWS)

    def test_min_score_threshold(self):
        from scoring import MIN_SCORE
        assert MIN_SCORE == 5

    def test_streak_bonus(self):
        from scoring import composite_score
        # 2+ wins should add +3
        score_no_streak = composite_score(
            "FIB_EXT_HI_1.272", entry_count=2, now_et=datetime.time(11, 0),
            tick_rate=1500, session_move_pts=0.0, direction="up",
        )
        score_with_streak = composite_score(
            "FIB_EXT_HI_1.272", entry_count=2, now_et=datetime.time(11, 0),
            tick_rate=1500, session_move_pts=0.0, direction="up",
            consecutive_wins=3,
        )
        assert score_with_streak == score_no_streak + 3


# ---------------------------------------------------------------------------
# 6. Backtest simulation consistency
# ---------------------------------------------------------------------------


class TestBacktestConsistency:
    """Tests that simulate_day parameters produce expected behavior."""

    def test_momentum_max_zero_disables_filter(self):
        """momentum_max=0 should not filter any entries."""
        # This is a parameter behavior test, not a full simulation
        # Verifies the parameter interface is preserved after refactor
        from backtest.simulate import simulate_day
        import inspect
        sig = inspect.signature(simulate_day)
        assert "momentum_max" in sig.parameters
        assert sig.parameters["momentum_max"].default == 0.0

    def test_simulate_day_has_all_required_params(self):
        """simulate_day must accept all parameters used by the test scripts."""
        from backtest.simulate import simulate_day
        import inspect
        sig = inspect.signature(simulate_day)
        required_params = [
            "dc", "arrays", "zone_factory", "target_fn",
            "max_per_level_map", "exclude_levels",
            "include_ibl", "include_vwap",
            "global_cooldown_after_loss_secs", "direction_filter",
            "momentum_max", "momentum_lookback_ticks",
            "daily_loss", "suppress_1330", "adaptive_caps",
            "direction_caps", "timeout_secs", "vol_filter_pct",
        ]
        for p in required_params:
            assert p in sig.parameters, f"Missing parameter: {p}"

    def test_trade_record_has_required_fields(self):
        """TradeRecord must have all fields used by results.py."""
        from backtest.simulate import TradeRecord
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(TradeRecord)]
        required = ["date", "level", "direction", "entry_count",
                     "outcome", "pnl_usd", "factors", "entry_idx",
                     "exit_idx", "entry_ns"]
        for f in required:
            assert f in field_names, f"Missing field: {f}"


# ---------------------------------------------------------------------------
# 7. Config consistency
# ---------------------------------------------------------------------------


class TestConfigConsistency:
    """Tests that config values are consistent and valid."""

    def test_deployed_levels_have_ts(self):
        """Every deployed level must have target/stop configured."""
        for level in ["FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272",
                       "FIB_0.236", "FIB_0.618", "FIB_0.764"]:
            assert level in BOT_PER_LEVEL_TS

    def test_deployed_levels_have_caps(self):
        """Every deployed level must have a per-level cap."""
        for level in BOT_PER_LEVEL_TS:
            assert level in BOT_PER_LEVEL_MAX_ENTRIES

    def test_loss_limit_is_200(self):
        from config import DAILY_LOSS_LIMIT_USD
        assert DAILY_LOSS_LIMIT_USD == 200.0

    def test_cooldown_is_zero(self):
        from config import BOT_GLOBAL_COOLDOWN_AFTER_LOSS_SECS
        assert BOT_GLOBAL_COOLDOWN_AFTER_LOSS_SECS == 0

    def test_ibh_sell_only(self):
        from config import BOT_DIRECTION_FILTER
        assert BOT_DIRECTION_FILTER.get("IBH") == "down"

    def test_excluded_levels(self):
        from config import BOT_EXCLUDE_LEVELS
        assert "FIB_0.5" in BOT_EXCLUDE_LEVELS

    def test_monday_double_caps_enabled(self):
        from config import BOT_MONDAY_DOUBLE_CAPS
        assert BOT_MONDAY_DOUBLE_CAPS is True


# ---------------------------------------------------------------------------
# 8. Zone lifecycle
# ---------------------------------------------------------------------------


class TestZoneLifecycle:
    """End-to-end zone lifecycle tests."""

    def test_full_lifecycle_entry_exit_reentry(self):
        """Zone: approach → enter → leave → re-approach → re-enter."""
        bz = BotZone("FIB_0.236", 27800.0)

        # Approach and enter
        assert bz.update(27800.5) is True
        assert bz.in_zone is True

        # Stay in zone — no re-fire
        assert bz.update(27800.3) is False

        # Leave zone
        bz.update(27805.0)
        assert bz.in_zone is False

        # Re-approach
        assert bz.update(27800.5) is True
        assert bz.in_zone is True

    def test_reset_allows_immediate_reentry(self):
        """After reset, zone can fire on same tick."""
        bz = BotZone("FIB_0.236", 27800.0)
        bz.update(27800.5)  # enter
        assert bz.in_zone is True

        bz.reset()
        assert bz.in_zone is False

        # Can re-enter immediately
        assert bz.update(27800.5) is True

    def test_multiple_zones_independent(self):
        """Zones track state independently."""
        z1 = BotZone("FIB_0.236", 27800.0)
        z2 = BotZone("FIB_0.764", 27850.0)

        z1.update(27800.5)  # z1 enters
        assert z1.in_zone is True
        assert z2.in_zone is False

        z2.update(27850.5)  # z2 enters
        assert z1.in_zone is True
        assert z2.in_zone is True

        z1.update(27810.0)  # z1 exits
        assert z1.in_zone is False
        assert z2.in_zone is True
