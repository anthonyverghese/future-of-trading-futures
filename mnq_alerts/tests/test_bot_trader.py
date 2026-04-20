"""Tests for bot_trader.py — BotZone zone tracking and entry scoring."""

import sys
import os
import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub ib_insync before importing broker/bot_trader.
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
import importlib

# Force IBKR_TRADING_ENABLED=true so broker imports ib_insync stubs.
with patch.dict(os.environ, {"IBKR_TRADING_ENABLED": "true"}):
    import config as _cfg

    _cfg.IBKR_TRADING_ENABLED = True
    import broker as _broker_mod

    importlib.reload(_broker_mod)
    import bot_trader as _bt_mod

    importlib.reload(_bt_mod)

from bot_trader import BotZone, bot_entry_score
from config import BOT_ENTRY_THRESHOLD

_ET = pytz.timezone("America/New_York")


def _score_at_hour(level, direction, entry_count, trend_60m, hour):
    """Compute score at a specific ET hour with neutral tick rate."""
    now_et = datetime.time(hour, 0)
    return bot_entry_score(
        level, direction, entry_count, trend_60m,
        tick_rate=1500.0,  # neutral bucket (no penalty/bonus)
        now_et=now_et,
    )


# ---------------------------------------------------------------------------
# 1. BotZone
# ---------------------------------------------------------------------------


class TestBotZone:
    def test_entry_when_within_threshold(self):
        bz = BotZone("IBL", 20000.0)
        result = bz.update(20000.0 + BOT_ENTRY_THRESHOLD)
        assert result is True
        assert bz.in_zone is True

    def test_no_entry_when_too_far(self):
        bz = BotZone("IBL", 20000.0)
        result = bz.update(20000.0 + BOT_ENTRY_THRESHOLD + 0.5)
        assert result is False
        assert bz.in_zone is False

    def test_no_reentry_while_in_zone(self):
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # enter
        assert bz.in_zone is True
        result = bz.update(20000.5)  # still within zone
        assert result is False  # no fresh entry

    def test_zone_stays_active_regardless_of_price(self):
        """Zone does NOT exit on price distance — only on reset()."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # enter
        assert bz.in_zone is True
        bz.update(20100.0)  # price moved 100 pts away
        assert bz.in_zone is True  # still active — no exit threshold

    def test_reset_allows_reentry(self):
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # first entry
        assert bz.entry_count == 1
        bz.reset()  # trade closed
        assert bz.in_zone is False
        result = bz.update(20000.0)  # re-enter
        assert result is True
        assert bz.entry_count == 2

    def test_entry_count_increments_across_resets(self):
        bz = BotZone("IBH", 20000.0)
        for i in range(3):
            bz.update(20000.0)  # enter
            bz.reset()  # trade closed
        assert bz.entry_count == 3

    def test_update_returns_false_after_entry(self):
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True
        # Subsequent updates while zone active return False
        assert bz.update(20000.0) is False
        assert bz.update(20000.5) is False


# ---------------------------------------------------------------------------
# 2. bot_entry_score
# ---------------------------------------------------------------------------


class TestBotEntryScore:
    def test_ibl_up_base(self):
        # IBL=+1, IBL×up=0, test#2=+1, h10=0, tick=neutral
        score = _score_at_hour("IBL", "up", 2, 0.0, 10)
        assert score == 2

    def test_ibh_up_gets_penalty(self):
        # IBH=-1, IBH×up=-1, test#2=+1
        score = _score_at_hour("IBH", "up", 2, 0.0, 10)
        assert score == -1

    def test_fib_ext_hi_up(self):
        # FIB_HI=0, FIB_HI×up=+1, test#2=+1
        score = _score_at_hour("FIB_EXT_HI_1.272", "up", 2, 0.0, 10)
        assert score == 2

    def test_first_touch_no_penalty(self):
        # test#1=0 (no penalty in bot weights)
        score = _score_at_hour("IBL", "up", 1, 0.0, 10)
        assert score == 1  # 1 + 0 + 0

    def test_third_touch_penalty(self):
        # test#3=-1
        score = _score_at_hour("IBL", "up", 3, 0.0, 10)
        assert score == 0  # 1 + 0 + (-1)

    def test_fib_lo_down_strong(self):
        # FIB_LO=+1, FIB_LO×down=+2, test#2=+1
        score = _score_at_hour("FIB_EXT_LO_1.272", "down", 2, 0.0, 10)
        assert score == 4

    def test_power_hour_penalty(self):
        # h15=-2
        score = _score_at_hour("IBL", "up", 2, 0.0, 15)
        assert score == 0  # 1 + 0 + 1 + (-2)

    def test_post_ib_bonus(self):
        # h11 (10:31-11:30)=+1
        score = _score_at_hour("IBL", "up", 2, 0.0, 11)
        assert score == 3  # 1 + 0 + 1 + 1

    def test_low_tick_rate_penalty(self):
        # tick<500=-2, no time (now_et=None)
        score = bot_entry_score("IBL", "up", 2, tick_rate=300)
        assert score == 0  # 1 + 0 + 1 + (-2)

    def test_vol_filter_dead_market(self):
        # range_30m_pct=0.10 (<0.15)=-4, no time/tick
        score = bot_entry_score("IBL", "up", 2, tick_rate=1500, range_30m_pct=0.10)
        assert score == -2  # 1 + 0 + 1 + (-4)
