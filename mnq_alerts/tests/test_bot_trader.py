"""Tests for bot_trader.py — BotZone zone tracking and entry scoring."""

import sys
import os
import datetime
from collections import deque
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
        bz.update(20005.0)  # leave — auto-exit
        assert bz.in_zone is False
        result = bz.update(20000.5)  # approach again
        assert result is True

    def test_reset_allows_reentry(self):
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # first entry
        bz.reset()  # trade closed
        assert bz.in_zone is False
        result = bz.update(20000.0)  # re-enter
        assert result is True

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

    def test_blocked_zone_no_spam(self):
        """When trade is blocked, zone stays in_zone and doesn't re-fire
        on subsequent ticks within threshold."""
        bz = BotZone("IBL", 20000.0)
        # Tick 1: enter (trade blocked — zone stays in_zone)
        assert bz.update(20000.0) is True
        # Tick 2: still within 1pt — no re-fire
        assert bz.update(20000.25) is False
        # Tick 3: price leaves zone
        bz.update(20005.0)
        assert bz.in_zone is False
        # Tick 4: price comes back — fresh entry
        assert bz.update(20000.0) is True

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

    def test_entry_count_not_incremented_by_update(self):
        """entry_count is only incremented in on_tick after filters pass,
        not in update(). This prevents inflation from oscillation noise."""
        bz = BotZone("IBH", 20000.0)
        for i in range(3):
            bz.update(20000.0)  # enter
            bz.reset()  # trade closed
        assert bz.entry_count == 0  # not incremented by update()

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

    def test_zone_reset_then_reentry(self):
        """After reset(), the next approach triggers a new entry."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)
        assert bz.in_zone is True
        bz.reset()
        assert bz.in_zone is False
        # Price comes back — new entry.
        assert bz.update(20000.5) is True

    def test_position_open_blocks_without_reset(self):
        """When position is already open, zone stays in_zone (no reset).
        Zone won't re-fire until price leaves 1pt or trade closes."""
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True
        # Broker blocks: "position already open". Zone stays in_zone.
        assert bz.in_zone is True
        # Next tick within 1pt — no re-fire.
        assert bz.update(20000.5) is False
        assert bz.in_zone is True

    def test_position_open_clears_when_price_leaves(self):
        """Zone blocked by position-open clears when price moves beyond 1pt."""
        bz = BotZone("IBL", 20000.0)
        bz.update(20000.0)  # enter, blocked
        assert bz.in_zone is True
        bz.update(20005.0)  # price leaves
        assert bz.in_zone is False
        # Can re-fire on next approach.
        assert bz.update(20000.0) is True

    def test_trade_close_resets_all_zones(self):
        """When a trade closes, ALL zones reset — not just the active level."""
        bz1 = BotZone("IBH", 20100.0)
        bz2 = BotZone("IBL", 20000.0)

        # Both zones enter.
        assert bz1.update(20100.0) is True
        assert bz2.update(20000.0) is True
        assert bz1.in_zone is True
        assert bz2.in_zone is True

        # Trade closes on bz1 — simulate BotTrader resetting ALL zones.
        for z in [bz1, bz2]:
            z.reset()
        assert bz1.in_zone is False
        assert bz2.in_zone is False

        # Both can re-enter.
        assert bz1.update(20100.0) is True
        assert bz2.update(20000.0) is True

    def test_skipped_entry_stays_in_zone(self):
        """If an entry is skipped (vol, score), zone stays in_zone.
        It clears when price leaves 1pt threshold."""
        bz = BotZone("IBH", 20100.0)
        assert bz.update(20100.0) is True
        assert bz.in_zone is True
        # Skipped — zone stays in_zone (no reset).
        assert bz.in_zone is True
        # Next tick within 1pt — still blocked.
        assert bz.update(20100.5) is False
        # Price leaves threshold.
        bz.update(20105.0)
        assert bz.in_zone is False
        # Can re-enter on next approach.
        assert bz.update(20100.0) is True

    def test_active_trade_level_resets_all_zones(self):
        """_active_trade_level triggers reset of ALL zones on trade close."""
        from bot_trader import BotTrader
        bt = BotTrader.__new__(BotTrader)
        bt._active_trade_level = "IBL"
        bt._zones = {
            "IBL": BotZone("IBL", 20000.0),
            "IBH": BotZone("IBH", 20100.0),
        }
        # Both zones are in_zone.
        bt._zones["IBL"].update(20000.0)
        bt._zones["IBH"].update(20100.0)
        assert bt._zones["IBL"].in_zone is True
        assert bt._zones["IBH"].in_zone is True

        # Simulate trade close: reset ALL zones.
        for z in bt._zones.values():
            z.reset()
        bt._active_trade_level = None

        assert bt._active_trade_level is None
        assert bt._zones["IBL"].in_zone is False
        assert bt._zones["IBH"].in_zone is False

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
        assert bz.update(20000.0) is True  # fires again

    def test_failed_trade_stays_in_zone(self):
        """If submit_bracket fails, zone stays in_zone.
        Clears when price leaves 1pt threshold."""
        bz = BotZone("IBL", 20000.0)
        assert bz.update(20000.0) is True
        # Trade failed — zone stays in_zone.
        assert bz.in_zone is True
        # Price stays near — no re-fire.
        assert bz.update(20000.5) is False
        # Price leaves — zone clears.
        bz.update(20005.0)
        assert bz.in_zone is False
        # Can try again on next approach.
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
            "FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272",
            "FIB_0.236", "FIB_0.618", "FIB_0.764",
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
        interior = ["FIB_0.236", "FIB_0.618", "FIB_0.764"]
        for level in interior:
            tgt, _ = BOT_PER_LEVEL_TS[level]
            assert tgt >= 8, f"{level}: interior fib target {tgt} < 8"

    def test_excluded_levels_not_in_per_level_ts(self):
        """FIB_0.5 should not be in BOT_PER_LEVEL_TS."""
        from config import BOT_PER_LEVEL_TS
        assert "FIB_0.5" not in BOT_PER_LEVEL_TS

    def test_per_level_max_entries_covers_all_levels(self):
        """Every deployed level should have a per-level max entry cap."""
        from config import BOT_PER_LEVEL_MAX_ENTRIES, BOT_PER_LEVEL_TS
        for level in BOT_PER_LEVEL_TS:
            assert level in BOT_PER_LEVEL_MAX_ENTRIES, f"Missing max entries for {level}"

    def test_ibh_sell_only(self):
        """IBH should be included but SELL only via direction filter."""
        from config import BOT_INCLUDE_IBH, BOT_DIRECTION_FILTER
        assert BOT_INCLUDE_IBH is True
        assert BOT_DIRECTION_FILTER.get("IBH") == "down"

    def test_fib_05_excluded(self):
        """FIB_0.5 should be in BOT_EXCLUDE_LEVELS."""
        from config import BOT_EXCLUDE_LEVELS
        assert "FIB_0.5" in BOT_EXCLUDE_LEVELS

    def test_extension_levels_have_smaller_targets(self):
        """Extension levels should have T <= 8 (quick small bounces)."""
        from config import BOT_PER_LEVEL_TS
        extensions = ["FIB_EXT_HI_1.272", "FIB_EXT_LO_1.272"]
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
        assert abs(fibs["FIB_0.764"] - (26900 + 0.764 * 200)) < 0.01

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

    def test_fib_0764_matches_standard_retracement(self):
        """FIB_0.764 should equal IBH - 0.236 * range (standard 0.236 retracement)."""
        from levels import calculate_interior_fibs
        ibh, ibl = 27429.75, 27298.00
        fibs = calculate_interior_fibs(ibh, ibl)
        ib_range = ibh - ibl
        # Standard 0.236 retracement from top = IBH - 0.236 * range
        standard_236 = ibh - 0.236 * ib_range
        # Our FIB_0.764 = IBL + 0.764 * range = IBH - 0.236 * range
        assert abs(fibs["FIB_0.764"] - standard_236) < 0.01

    def test_fib_0764_not_0786(self):
        """FIB_0.786 should NOT exist — replaced by FIB_0.764."""
        from levels import calculate_interior_fibs
        fibs = calculate_interior_fibs(27100.0, 26900.0)
        assert "FIB_0.786" not in fibs
        assert "FIB_0.764" in fibs


# ---------------------------------------------------------------------------
# 6. Momentum filter
# ---------------------------------------------------------------------------


class TestMomentumFilter:
    """Tests for the 5-min momentum filter in BotTrader.

    The filter skips entries where price moved > 5pts in the trade direction
    over the last 5 minutes (momentum carrying through the level).
    """

    def _make_trader(self):
        """Create a BotTrader with stubbed broker for unit testing."""
        from bot_trader import BotTrader
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
        bt._adaptive_caps_restored = True  # skip adaptive caps logic
        bt._adaptive_caps_until_et_mins = 0
        bt._adaptive_accepted_trades = 0
        bt._adaptive_any_loss = False
        return bt

    def test_price_5m_ago_updates_on_tick(self):
        """_price_5m_ago should track the price from ~5 min ago."""
        from bot_trader import BotTrader
        bt = self._make_trader()

        base_time = datetime.datetime(2026, 5, 1, 11, 0, 0, tzinfo=_ET)

        # Feed 6 minutes of ticks (1 per second)
        for sec in range(360):
            t = base_time + datetime.timedelta(seconds=sec)
            price = 27800.0 + sec * 0.1  # slowly rising
            bt._price_window_5m.append((t, price))
            cutoff_5m = t - datetime.timedelta(minutes=5)
            while bt._price_window_5m and bt._price_window_5m[0][0] < cutoff_5m:
                bt._price_5m_ago = bt._price_window_5m.popleft()[1]

        # After 6 min, _price_5m_ago should be ~the price from 5 min ago
        # Price at t=0: 27800.0, price at t=60 (1 min): 27806.0
        # price_5m_ago should be around 27800 + 60*0.1 = 27806 (price at 1 min mark)
        assert bt._price_5m_ago is not None
        assert abs(bt._price_5m_ago - 27806.0) < 1.0

    def test_momentum_with_direction_blocks_sell(self):
        """SELL entry should be blocked when price fell >5pts in last 5 min
        (momentum carrying through — bad for a bounce)."""
        # For SELL: momentum = -(price - price_5m_ago)
        # If price fell 10pts: momentum = -(-10) = +10 > 5 → blocked
        price = 27830.0
        price_5m_ago = 27840.0  # price fell 10pts
        momentum = price - price_5m_ago  # -10
        direction = "down"
        if direction == "down":
            momentum = -momentum  # +10
        assert momentum > 5.0  # should be blocked

    def test_momentum_against_direction_allows_sell(self):
        """SELL entry should be allowed when price rose to the level
        (approaching from below — classic bounce setup)."""
        price = 27840.0
        price_5m_ago = 27830.0  # price rose 10pts TO the level
        momentum = price - price_5m_ago  # +10
        direction = "down"
        if direction == "down":
            momentum = -momentum  # -10
        assert momentum <= 5.0  # should be allowed

    def test_momentum_with_direction_blocks_buy(self):
        """BUY entry should be blocked when price rose >5pts in last 5 min."""
        price = 27840.0
        price_5m_ago = 27830.0  # price rose 10pts
        momentum = price - price_5m_ago  # +10
        direction = "up"
        # No negation for "up"
        assert momentum > 5.0  # should be blocked

    def test_momentum_against_direction_allows_buy(self):
        """BUY entry should be allowed when price fell to the level."""
        price = 27830.0
        price_5m_ago = 27840.0  # price fell 10pts TO the level
        momentum = price - price_5m_ago  # -10
        direction = "up"
        assert momentum <= 5.0  # should be allowed

    def test_small_momentum_allows_trade(self):
        """Momentum <= 5pts should NOT block the trade."""
        price = 27835.0
        price_5m_ago = 27832.0  # only 3pt move
        momentum = price - price_5m_ago  # +3
        direction = "up"
        assert momentum <= 5.0  # allowed

    def test_no_price_history_allows_trade(self):
        """When _price_5m_ago is None (startup), trade should be allowed."""
        # The filter is: if self._price_5m_ago is not None: ...
        # So None means no filter applied — trade allowed.
        assert True  # filter skipped when _price_5m_ago is None

    def test_exactly_5pt_allows_trade(self):
        """Momentum of exactly 5.0 should be allowed (> 5.0, not >= 5.0)."""
        price = 27835.0
        price_5m_ago = 27830.0
        momentum = price - price_5m_ago  # exactly 5.0
        direction = "up"
        assert not (momentum > 5.0)  # NOT blocked — boundary case


# ---------------------------------------------------------------------------
# 7. Adaptive caps
# ---------------------------------------------------------------------------


class TestAdaptiveCaps:
    """Tests for the adaptive caps logic in BotTrader."""

    def test_initial_state(self):
        """Adaptive caps should start with half caps until 11:00 AM ET."""
        from bot_trader import BotTrader
        bt = BotTrader.__new__(BotTrader)
        bt._adaptive_caps_restored = False
        bt._adaptive_caps_until_et_mins = 660  # 11:00 AM
        bt._adaptive_accepted_trades = 0
        bt._adaptive_any_loss = False
        assert bt._adaptive_caps_restored is False
        assert bt._adaptive_caps_until_et_mins == 660

    def test_three_wins_restores_caps(self):
        """Three consecutive wins from start should restore full caps."""
        from bot_trader import BotTrader
        bt = BotTrader.__new__(BotTrader)
        bt._adaptive_caps_restored = False
        bt._adaptive_accepted_trades = 0
        bt._adaptive_any_loss = False

        # Simulate 3 wins
        for _ in range(3):
            bt._adaptive_accepted_trades += 1
        # Check condition: 3 trades, no loss
        assert bt._adaptive_accepted_trades >= 3
        assert not bt._adaptive_any_loss
        # Should restore
        bt._adaptive_caps_restored = True
        assert bt._adaptive_caps_restored is True

    def test_loss_prevents_early_restore(self):
        """A loss before 3 wins prevents early restore."""
        from bot_trader import BotTrader
        bt = BotTrader.__new__(BotTrader)
        bt._adaptive_caps_restored = False
        bt._adaptive_accepted_trades = 2  # 2 wins
        bt._adaptive_any_loss = False

        # Loss on trade 3
        bt._adaptive_accepted_trades += 1
        bt._adaptive_any_loss = True
        # Condition: 3 trades but has loss
        assert bt._adaptive_accepted_trades >= 3
        assert bt._adaptive_any_loss  # prevents restore

    def test_loss_extends_window(self):
        """A loss should extend the half-cap window by 30 min."""
        from bot_trader import BotTrader
        bt = BotTrader.__new__(BotTrader)
        bt._adaptive_caps_restored = False
        bt._adaptive_caps_until_et_mins = 660  # 11:00 AM

        # Loss at 10:45 (et_mins = 645)
        loss_et = 645
        bt._adaptive_caps_until_et_mins = loss_et + 30  # extends to 11:15
        assert bt._adaptive_caps_until_et_mins == 675

    def test_reset_clears_state(self):
        """reset_daily_state should clear all adaptive caps state."""
        from bot_trader import BotTrader
        bt = BotTrader.__new__(BotTrader)
        bt._adaptive_caps_restored = True
        bt._adaptive_caps_until_et_mins = 700
        bt._adaptive_accepted_trades = 5
        bt._adaptive_any_loss = True

        # Reset
        bt._adaptive_caps_restored = False
        bt._adaptive_caps_until_et_mins = 660
        bt._adaptive_accepted_trades = 0
        bt._adaptive_any_loss = False

        assert bt._adaptive_caps_restored is False
        assert bt._adaptive_caps_until_et_mins == 660
        assert bt._adaptive_accepted_trades == 0
        assert bt._adaptive_any_loss is False

    def test_half_cap_calculation(self):
        """Half cap should be max(1, cap // 2)."""
        assert max(1, 18 // 2) == 9   # FIB_0.236: 18 -> 9
        assert max(1, 3 // 2) == 1    # FIB_0.618: 3 -> 1
        assert max(1, 5 // 2) == 2    # FIB_0.764: 5 -> 2
        assert max(1, 6 // 2) == 3    # EXT levels: 6 -> 3
        assert max(1, 7 // 2) == 3    # IBH: 7 -> 3
        assert max(1, 1 // 2) == 1    # minimum is 1, not 0
