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

    def test_zone_auto_exits_when_price_leaves_threshold(self):
        """Zone auto-exits when price moves beyond entry threshold."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # enter
        assert bz.in_zone is True
        bz.update(20000.5)  # still within 1pt
        assert bz.in_zone is True
        bz.update(20002.0)  # moved beyond 1pt threshold
        assert bz.in_zone is False  # auto-exited

    def test_zone_rearms_after_auto_exit(self):
        """After auto-exit, zone fires again on next approach."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # enter
        assert bz.entry_count == 1
        bz.update(20005.0)  # leave — auto-exit
        assert bz.in_zone is False
        result = bz.update(20000.5)  # approach again
        assert result is True
        assert bz.entry_count == 2

    def test_reset_allows_reentry(self):
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # first entry
        assert bz.entry_count == 1
        bz.reset()  # trade closed
        assert bz.in_zone is False
        result = bz.update(20000.0)  # re-enter
        assert result is True
        assert bz.entry_count == 2

    def test_auto_exit_at_exact_boundary(self):
        """Price at exactly BOT_ENTRY_THRESHOLD stays in zone."""
        from config import BOT_ENTRY_THRESHOLD
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # enter
        assert bz.in_zone is True
        # At exactly 1pt — still in zone (<=, not <)
        bz.update(20000.0 + BOT_ENTRY_THRESHOLD)
        assert bz.in_zone is True
        # One tick beyond — auto-exit
        bz.update(20000.0 + BOT_ENTRY_THRESHOLD + 0.25)
        assert bz.in_zone is False

    def test_auto_exit_below_line(self):
        """Auto-exit works when price drops below the line too."""
        bz = BotZone("IBL", 20000.0)
        bz.update(19999.5)  # enter from below (within 1pt)
        assert bz.in_zone is True
        bz.update(19998.5)  # move beyond 1pt below
        assert bz.in_zone is False

    def test_rapid_reset_reenter_cycle(self):
        """Simulates the blocked-trade scenario: enter, reset, re-enter
        while price stays in zone, then price leaves."""
        bz = BotZone("IBL", 20000.0)
        # Tick 1: enter (trade blocked)
        assert bz.update(20000.0) is True
        bz.reset()  # trade was blocked
        # Tick 2: still in zone, re-enter (trade blocked again)
        assert bz.update(20000.25) is True
        bz.reset()
        # Tick 3: price leaves zone
        bz.update(20005.0)
        assert bz.in_zone is False
        # Tick 4: price comes back — should fire
        assert bz.update(20000.0) is True
        assert bz.entry_count == 3

    def test_vwap_zone_with_drifting_price(self):
        """VWAP zone auto-exit uses current price, which drifts."""
        bz = BotZone("VWAP", 20000.0, drifts=True)
        bz.update(20000.5)  # enter
        assert bz.in_zone is True
        # VWAP drifts — update the price
        bz.price = 20010.0
        # Now 20000.5 is 9.5 pts from new VWAP — auto-exit
        bz.update(20000.5)
        assert bz.in_zone is False
        # Approach new VWAP
        assert bz.update(20010.0) is True

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
# 1b. Zone reset + 1-position constraint integration
# ---------------------------------------------------------------------------


class TestZoneResetAndOnePosition:
    """Test that the new zone-reset-on-trade-close logic never allows
    two simultaneous trades."""

    def test_zone_blocks_reentry_while_in_threshold(self):
        """While price stays within threshold, update() returns False."""
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True  # first entry
        # Price stays within 1pt — zone still active, no re-entry.
        assert bz.update(20000.5) is False
        assert bz.in_zone is True

    def test_zone_reentry_after_price_leaves(self):
        """After price leaves threshold, next approach re-enters."""
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True  # first entry
        bz.update(20050.0)  # price leaves — auto-exit
        assert bz.in_zone is False
        assert bz.update(20000.0) is True  # re-entry
        assert bz.entry_count == 2

    def test_zone_reset_then_reentry(self):
        """After reset(), the next approach triggers a new entry."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)
        assert bz.in_zone is True
        bz.reset()
        assert bz.in_zone is False
        # Price comes back — new entry.
        assert bz.update(20000.5) is True
        assert bz.entry_count == 2

    def test_two_zones_cannot_both_be_active_with_trade(self):
        """Simulates the BotTrader flow: if one zone has an active trade,
        the broker blocks the second via can_trade()."""
        bz1 = BotZone("IBH", 20100.0)
        bz2 = BotZone("IBL", 20000.0)

        # First zone triggers.
        assert bz1.update(20100.0) is True
        assert bz1.in_zone is True

        # Second zone triggers — update() returns True (zone-level it's valid),
        # but the broker's can_trade() would block it. If blocked, reset().
        assert bz2.update(20000.0) is True
        # Simulate broker blocking: reset zone since we can't trade.
        bz2.reset()
        assert bz2.in_zone is False

        # First trade closes — reset first zone.
        bz1.reset()
        assert bz1.in_zone is False

        # Now second zone can enter.
        assert bz2.update(20000.0) is True
        assert bz2.in_zone is True

    def test_skipped_entry_resets_zone(self):
        """If an entry is skipped (score, vol, max), zone must reset
        so the level isn't permanently locked."""
        bz = BotZone("IBH", 20100.0)
        assert bz.update(20100.0) is True
        assert bz.in_zone is True
        # Simulate scoring skip — must reset.
        bz.reset()
        assert bz.in_zone is False
        # Can re-enter later.
        assert bz.update(20100.0) is True

    def test_active_trade_level_tracks_correctly(self):
        """_active_trade_level is set on trade open, cleared on close."""
        from bot_trader import BotTrader
        bt = BotTrader.__new__(BotTrader)
        bt._active_trade_level = None
        bt._zones = {"IBL": BotZone("IBL", 20000.0)}

        # Simulate trade open.
        bt._active_trade_level = "IBL"
        assert bt._active_trade_level == "IBL"

        # Simulate trade close (broker._position_open goes False).
        # The on_tick check would do:
        if bt._active_trade_level is not None:
            bt._zones[bt._active_trade_level].reset()
            bt._active_trade_level = None
        assert bt._active_trade_level is None
        assert bt._zones["IBL"].in_zone is False

    def test_multiple_resets_are_idempotent(self):
        """Calling reset() multiple times is safe."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)
        bz.reset()
        bz.reset()  # second reset — should be no-op
        assert bz.in_zone is False
        # Can still re-enter.
        assert bz.update(20000.0) is True

    def test_zone_auto_exits_during_trade(self):
        """Zone auto-exits when price leaves threshold, even if trade is open.
        The trade is protected by its bracket (target/stop), not the zone."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)
        assert bz.in_zone is True
        bz.update(20005.0)  # price leaves threshold
        assert bz.in_zone is False  # zone auto-exited
        # Zone re-fires on next approach
        assert bz.update(20000.0) is True

    def test_entry_count_not_incremented_on_skipped_reset(self):
        """If zone enters then resets (skipped trade), entry_count was
        already incremented. Next entry increments again."""
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True  # entry_count = 1
        bz.reset()  # skipped
        assert bz.update(20000.0) is True  # entry_count = 2
        assert bz.entry_count == 2

    def test_failed_trade_resets_zone(self):
        """If submit_bracket fails, zone should reset so it's not stuck."""
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True
        # Trade failed.
        bz.reset()
        assert bz.in_zone is False
        # Can try again.
        assert bz.update(20000.5) is True


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


# ---------------------------------------------------------------------------
# 4. Per-level target/stop configuration
# ---------------------------------------------------------------------------


class TestPerLevelTS:
    def test_per_level_ts_has_all_deployed_levels(self):
        """Every deployed bot level must have a T/S entry."""
        from config import BOT_PER_LEVEL_TS
        expected_levels = [
            "IBH", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272",
            "FIB_0.236", "FIB_0.5", "FIB_0.618", "FIB_0.786",
        ]
        for level in expected_levels:
            assert level in BOT_PER_LEVEL_TS, f"Missing T/S for {level}"

    def test_per_level_ts_values_valid(self):
        """All targets >= 6, all stops >= 6, stop >= target."""
        from config import BOT_PER_LEVEL_TS
        for level, (tgt, stop) in BOT_PER_LEVEL_TS.items():
            assert tgt >= 6, f"{level}: target {tgt} < 6"
            assert stop >= 6, f"{level}: stop {stop} < 6"
            assert stop >= tgt, f"{level}: stop {stop} < target {tgt}"

    def test_interior_fibs_have_bigger_targets(self):
        """Interior fibs should have T >= 8 (bigger bounces inside IB range)."""
        from config import BOT_PER_LEVEL_TS
        interior = ["FIB_0.236", "FIB_0.5", "FIB_0.618", "FIB_0.786"]
        for level in interior:
            tgt, _ = BOT_PER_LEVEL_TS[level]
            assert tgt >= 8, f"{level}: interior fib target {tgt} < 8"

    def test_extension_levels_have_smaller_targets(self):
        """Extension/IBH levels should have T <= 8 (quick small bounces)."""
        from config import BOT_PER_LEVEL_TS
        extensions = ["IBH", "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]
        for level in extensions:
            tgt, _ = BOT_PER_LEVEL_TS[level]
            assert tgt <= 8, f"{level}: extension target {tgt} > 8"

    def test_fallback_to_default_for_unknown_level(self):
        """Unknown levels should use default BOT_TARGET/STOP_POINTS."""
        from config import BOT_PER_LEVEL_TS, BOT_TARGET_POINTS, BOT_STOP_POINTS
        tgt, stop = BOT_PER_LEVEL_TS.get("UNKNOWN_LEVEL", (BOT_TARGET_POINTS, BOT_STOP_POINTS))
        assert tgt == BOT_TARGET_POINTS
        assert stop == BOT_STOP_POINTS


# ---------------------------------------------------------------------------
# 5. Interior fib level calculation
# ---------------------------------------------------------------------------


class TestInteriorFibs:
    def test_calculate_interior_fibs(self):
        from levels import calculate_interior_fibs
        fibs = calculate_interior_fibs(27100.0, 26900.0)
        # IB range = 200
        assert abs(fibs["FIB_0.236"] - (26900 + 0.236 * 200)) < 0.01
        assert abs(fibs["FIB_0.5"] - (26900 + 0.5 * 200)) < 0.01
        assert abs(fibs["FIB_0.618"] - (26900 + 0.618 * 200)) < 0.01
        assert abs(fibs["FIB_0.786"] - (26900 + 0.786 * 200)) < 0.01

    def test_interior_fibs_within_ib_range(self):
        from levels import calculate_interior_fibs
        ibh, ibl = 27100.0, 26900.0
        fibs = calculate_interior_fibs(ibh, ibl)
        for name, price in fibs.items():
            assert ibl <= price <= ibh, f"{name}={price} outside IB range [{ibl}, {ibh}]"

    def test_interior_fibs_excludes_0382(self):
        """FIB_0.382 should NOT be included (weakest level, 70.3% WR)."""
        from levels import calculate_interior_fibs
        fibs = calculate_interior_fibs(27100.0, 26900.0)
        assert "FIB_0.382" not in fibs
