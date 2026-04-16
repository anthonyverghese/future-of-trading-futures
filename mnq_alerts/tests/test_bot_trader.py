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
from config import BOT_ENTRY_THRESHOLD, BOT_EXIT_THRESHOLD

_ET = pytz.timezone("America/New_York")


def _score_at_hour(level, direction, entry_count, trend_60m, hour):
    """Compute score at a specific ET hour."""
    fake_now = datetime.datetime(2026, 4, 15, hour, 0, tzinfo=_ET)
    with patch("bot_trader.datetime") as mock_dt:
        mock_dt.datetime.now.return_value = fake_now
        return bot_entry_score(level, direction, entry_count, trend_60m)


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

    def test_exit_when_beyond_exit_threshold(self):
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # enter
        assert bz.in_zone is True
        bz.update(20000.0 + BOT_EXIT_THRESHOLD + 1)  # drift away
        assert bz.in_zone is False

    def test_reentry_after_exit(self):
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # first entry
        assert bz.entry_count == 1
        bz.update(20000.0 + BOT_EXIT_THRESHOLD + 1)  # exit
        assert bz.in_zone is False
        result = bz.update(20000.0)  # re-enter
        assert result is True
        assert bz.entry_count == 2

    def test_entry_count_increments(self):
        bz = BotZone("IBH", 20000.0)
        for i in range(3):
            bz.update(20000.0)  # enter
            bz.update(20000.0 + BOT_EXIT_THRESHOLD + 1)  # exit
        assert bz.entry_count == 3

    def test_update_returns_false_after_entry(self):
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True
        # Subsequent updates within zone return False
        assert bz.update(20000.0) is False
        assert bz.update(20000.5) is False

    def test_drifting_zone_uses_current_price_for_exit(self):
        """BotZone with drifts=True uses self.price (not self._ref_price) for exit."""
        bz = BotZone("VWAP", 20000.0, drifts=True)
        bz.update(20000.0)  # enter — _ref_price locked at 20000
        assert bz.in_zone is True

        # Simulate VWAP drifting toward the current price
        bz.price = 20010.0  # VWAP moved up

        # Price at 20026 is >15 pts from drifted VWAP (20010) → should exit
        # But only 26 pts from original _ref_price (20000) — if using _ref_price
        # it would also exit. Let's pick a value that distinguishes the two:
        # 20010 + 15.5 = 20025.5 → >15 from drifted (20010) → exits
        # 20000 + 15.5 = only 25.5 from ref → also exits
        # Instead test: price near ref but far from drifted VWAP
        bz2 = BotZone("VWAP", 20000.0, drifts=True)
        bz2.update(20000.0)  # enter, ref=20000
        bz2.price = 20020.0  # VWAP drifted to 20020

        # current_price=20000 → 20 pts from drifted VWAP (20020) > 15 → exit
        # But only 0 pts from ref (20000) → would NOT exit with ref
        bz2.update(20000.0)
        assert bz2.in_zone is False  # exited because drifts=True uses self.price


# ---------------------------------------------------------------------------
# 2. bot_entry_score
# ---------------------------------------------------------------------------


class TestBotEntryScore:
    def test_ibl_up_gets_bonus(self):
        # IBL = +2, direction up with IBL = +1 combo, entry_count != 1 and < 3 = 0
        # At 10:00 AM ET (during IB, no time bonus/penalty at this hour)
        score = _score_at_hour("IBL", "up", 2, 0.0, 10)
        assert score == 3  # 2 (level) + 1 (combo)

    def test_ibh_up_gets_penalty(self):
        # IBH = -1, direction up with IBH = -1 combo
        score = _score_at_hour("IBH", "up", 2, 0.0, 10)
        assert score == -2  # -1 (level) + -1 (combo)

    def test_fib_ext_base_score(self):
        score = _score_at_hour("FIB_EXT_HI_1.272", "up", 2, 0.0, 10)
        assert score == 1  # +1 (level), no combo, no entry_count adj

    def test_first_touch_penalty(self):
        # IBL up = +3 base, entry_count=1 → -2
        score = _score_at_hour("IBL", "up", 1, 0.0, 10)
        assert score == 1  # 2 + 1 - 2

    def test_third_touch_bonus(self):
        # IBL up = +3 base, entry_count=3 → +1
        score = _score_at_hour("IBL", "up", 3, 0.0, 10)
        assert score == 4  # 2 + 1 + 1

    def test_counter_trend_penalty(self):
        # direction="up", trend_60m=-60 (< -50) → -3 penalty
        score = _score_at_hour("IBL", "up", 2, -60.0, 10)
        assert score == 0  # 2 + 1 + (-3)

    def test_with_trend_no_penalty(self):
        # direction="up", trend_60m=+60 → no penalty
        score = _score_at_hour("IBL", "up", 2, 60.0, 10)
        assert score == 3  # 2 + 1, no penalty

    def test_power_hour_bonus(self):
        # At 15:00 ET → +2 time bonus
        score = _score_at_hour("IBL", "up", 2, 0.0, 15)
        assert score == 5  # 2 + 1 + 2

    def test_afternoon_bonus(self):
        # At 13:00 ET → +1 time bonus
        score = _score_at_hour("IBL", "up", 2, 0.0, 13)
        assert score == 4  # 2 + 1 + 1

    def test_post_ib_weakness_penalty(self):
        # At 11:00 ET (10:30-11:30 window) → -1
        score = _score_at_hour("IBL", "up", 2, 0.0, 11)
        assert score == 2  # 2 + 1 - 1
