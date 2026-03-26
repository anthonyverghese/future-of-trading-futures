"""Tests for alert_manager module: LevelState, AlertManager, build_message, helpers."""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch

from alert_manager import (
    LevelState,
    AlertManager,
    build_message,
    _ordinal,
    _time_bucket,
)
from config import ALERT_THRESHOLD_POINTS, ALERT_EXIT_POINTS

# ---------------------------------------------------------------------------
# 1. LevelState zone entry and exit (hysteresis)
# ---------------------------------------------------------------------------


class TestLevelStateZoneEntryExit:
    def test_enters_zone_when_within_threshold(self):
        ls = LevelState(name="IBL", price=20000.0)
        # Price within 7 points of level
        assert ls.update(20005.0) is True
        assert ls.in_zone is True

    def test_enters_zone_at_exact_threshold_boundary(self):
        ls = LevelState(name="IBL", price=20000.0)
        assert ls.update(20000.0 + ALERT_THRESHOLD_POINTS) is True
        assert ls.in_zone is True

    def test_does_not_enter_zone_beyond_threshold(self):
        ls = LevelState(name="IBL", price=20000.0)
        assert ls.update(20000.0 + ALERT_THRESHOLD_POINTS + 1) is False
        assert ls.in_zone is False

    def test_stays_in_zone_no_repeat_alert(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(20003.0)  # enter
        # Still in zone, should return False (no new alert)
        assert ls.update(20004.0) is False
        assert ls.in_zone is True

    def test_exits_zone_when_beyond_exit_threshold(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(20003.0)  # enter zone, reference_price = 20000
        # Move beyond ALERT_EXIT_POINTS (20) from reference
        assert ls.update(20000.0 + ALERT_EXIT_POINTS + 1) is False
        assert ls.in_zone is False
        assert ls.reference_price is None

    def test_does_not_exit_at_exactly_exit_threshold(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(20003.0)  # enter
        # Exactly at exit threshold — should stay in zone
        assert ls.update(20000.0 + ALERT_EXIT_POINTS) is False
        assert ls.in_zone is True

    def test_reentry_after_exit(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(20003.0)  # enter
        ls.update(20000.0 + ALERT_EXIT_POINTS + 1)  # exit
        # Re-enter
        assert ls.update(20002.0) is True
        assert ls.in_zone is True

    def test_reference_price_locked_at_entry(self):
        ls = LevelState(name="VWAP", price=20000.0)
        ls.update(20003.0)  # enter
        assert ls.reference_price == 20000.0
        # Even if level price drifts (VWAP), reference stays locked
        ls.price = 20010.0
        assert ls.reference_price == 20000.0

    def test_below_level_entry(self):
        ls = LevelState(name="IBL", price=20000.0)
        assert ls.update(19995.0) is True
        assert ls.in_zone is True

    def test_below_level_exit(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(19995.0)  # enter, reference = 20000
        ls.update(20000.0 - ALERT_EXIT_POINTS - 1)  # exit below
        assert ls.in_zone is False


# ---------------------------------------------------------------------------
# 2. LevelState entry_count increments
# ---------------------------------------------------------------------------


class TestLevelStateEntryCount:
    def test_first_entry(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(20003.0)
        assert ls.entry_count == 1

    def test_count_increments_on_reentry(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(20003.0)  # entry 1
        ls.update(20000.0 + ALERT_EXIT_POINTS + 1)  # exit
        ls.update(20002.0)  # entry 2
        assert ls.entry_count == 2

    def test_count_does_not_increment_while_in_zone(self):
        ls = LevelState(name="IBL", price=20000.0)
        ls.update(20003.0)
        ls.update(20004.0)
        ls.update(20001.0)
        assert ls.entry_count == 1

    def test_multiple_reentries(self):
        ls = LevelState(name="IBL", price=20000.0)
        for i in range(5):
            ls.update(20003.0)  # enter
            ls.update(20000.0 + ALERT_EXIT_POINTS + 1)  # exit
        assert ls.entry_count == 5


# ---------------------------------------------------------------------------
# 3. AlertManager level registration
# ---------------------------------------------------------------------------


class TestAlertManagerRegistration:
    def _make_manager(self):
        return AlertManager(
            log_fn=lambda **kw: 1,
            notify_fn=lambda t, b: True,
        )

    def test_ibh_excluded_from_levels(self):
        am = self._make_manager()
        am.update_levels(ibh=20100.0, ibl=20000.0, vwap=20050.0)
        assert "IBH" not in am._levels
        assert "IBL" in am._levels
        assert "VWAP" in am._levels

    def test_vwap_updates_on_every_call(self):
        am = self._make_manager()
        am.update_levels(ibh=None, ibl=None, vwap=20050.0)
        assert am._levels["VWAP"].price == 20050.0
        am.update_levels(ibh=None, ibl=None, vwap=20060.0)
        assert am._levels["VWAP"].price == 20060.0

    def test_ibl_updates_on_every_call(self):
        am = self._make_manager()
        am.update_levels(ibh=None, ibl=20000.0, vwap=None)
        am.update_levels(ibh=None, ibl=20010.0, vwap=None)
        assert am._levels["IBL"].price == 20010.0

    def test_none_levels_skipped(self):
        am = self._make_manager()
        am.update_levels(ibh=None, ibl=None, vwap=None)
        assert len(am._levels) == 0

    def test_update_fib_levels(self):
        am = self._make_manager()
        am.update_fib_levels({"FIB_EXT_LO_1.272": 19900.0, "FIB_EXT_HI_1.272": 20200.0})
        assert "FIB_EXT_LO_1.272" in am._levels
        assert "FIB_EXT_HI_1.272" in am._levels
        assert am._levels["FIB_EXT_LO_1.272"].price == 19900.0

    def test_fib_levels_not_overwritten(self):
        am = self._make_manager()
        am.update_fib_levels({"FIB_EXT_LO_1.272": 19900.0})
        am._levels["FIB_EXT_LO_1.272"].entry_count = 3
        am.update_fib_levels({"FIB_EXT_LO_1.272": 19950.0})
        # Should NOT overwrite — fib levels are fixed
        assert am._levels["FIB_EXT_LO_1.272"].price == 19900.0
        assert am._levels["FIB_EXT_LO_1.272"].entry_count == 3


# ---------------------------------------------------------------------------
# 4. AlertManager check_and_notify with mock callbacks
# ---------------------------------------------------------------------------


class TestAlertManagerCheckAndNotify:
    def _make_manager_with_tracking(self):
        log_calls = []
        notify_calls = []

        def mock_log(**kw):
            log_calls.append(kw)
            return len(log_calls)

        def mock_notify(title, body):
            notify_calls.append((title, body))
            return True

        am = AlertManager(log_fn=mock_log, notify_fn=mock_notify)
        return am, log_calls, notify_calls

    def test_fires_notification_on_zone_entry(self):
        am, log_calls, notify_calls = self._make_manager_with_tracking()
        am.update_levels(ibh=None, ibl=20000.0, vwap=None)
        # Use params that produce a high score: power hour, tick_rate in sweet spot,
        # strongly green session, 2+ consecutive wins, 2nd test
        # First, prime entry_count to 1 so next entry is #2
        am._levels["IBL"].entry_count = 1
        fired, all_entries = am.check_and_notify(
            current_price=19995.0,
            now_et=datetime.time(15, 30),  # power hour +2
            tick_rate=1800,  # sweet spot +2
            session_move_pts=-60,  # strongly red +1
            consecutive_wins=3,  # streak +3
        )
        # IBL down: level +1, direction +1, power +2, tick +2, test#2 +1, move +1, streak +3 = 11
        assert len(fired) == 1
        assert fired[0][1] == "IBL"
        assert len(log_calls) == 1
        assert len(notify_calls) == 1

    def test_no_notification_when_not_in_zone(self):
        am, log_calls, notify_calls = self._make_manager_with_tracking()
        am.update_levels(ibh=None, ibl=20000.0, vwap=None)
        fired, all_entries = am.check_and_notify(
            current_price=20050.0,  # far from level
            now_et=datetime.time(15, 30),
            tick_rate=1800,
            session_move_pts=-60,
            consecutive_wins=3,
        )
        assert len(fired) == 0
        assert len(notify_calls) == 0

    def test_log_fn_receives_correct_kwargs(self):
        am, log_calls, notify_calls = self._make_manager_with_tracking()
        am.update_levels(ibh=None, ibl=20000.0, vwap=None)
        am._levels["IBL"].entry_count = 1
        ts = datetime.datetime(2026, 3, 25, 15, 30)
        am.check_and_notify(
            current_price=19996.0,
            now_et=datetime.time(15, 30),
            tick_rate=1800,
            session_move_pts=-60,
            consecutive_wins=3,
            trade_ts=ts,
        )
        assert log_calls[0]["ticker"] == "MNQ"
        assert log_calls[0]["line"] == "IBL"
        assert log_calls[0]["line_price"] == 20000.0
        assert log_calls[0]["current_price"] == 19996.0
        assert log_calls[0]["direction"] == "down"
        assert log_calls[0]["trade_ts"] == ts


# ---------------------------------------------------------------------------
# 5. AlertManager suppresses low-score alerts
# ---------------------------------------------------------------------------


class TestAlertManagerSuppression:
    def test_low_score_suppressed(self):
        notify_calls = []
        am = AlertManager(
            log_fn=lambda **kw: 1,
            notify_fn=lambda t, b: notify_calls.append((t, b)) or True,
        )
        am.update_levels(ibh=None, ibl=None, vwap=20000.0)
        # VWAP level=-1, direction down=-1, entry#1=-1, no time bonus,
        # no tick bonus, mildly green=-3 → score very low
        fired, _ = am.check_and_notify(
            current_price=19997.0,
            now_et=datetime.time(10, 0),
            tick_rate=500,
            session_move_pts=10,  # mildly green → -3
            consecutive_losses=3,  # streak → -4
        )
        assert len(fired) == 0
        assert len(notify_calls) == 0


# ---------------------------------------------------------------------------
# 6. all_zone_entries includes suppressed entries
# ---------------------------------------------------------------------------


class TestAllZoneEntries:
    def test_suppressed_entry_in_all_zone_entries(self):
        am = AlertManager(
            log_fn=lambda **kw: 1,
            notify_fn=lambda t, b: True,
        )
        am.update_levels(ibh=None, ibl=None, vwap=20000.0)
        # Use params that ensure suppression (low score)
        fired, all_entries = am.check_and_notify(
            current_price=19997.0,
            now_et=datetime.time(10, 0),
            tick_rate=500,
            session_move_pts=10,
            consecutive_losses=3,
        )
        assert len(fired) == 0
        assert len(all_entries) == 1
        assert all_entries[0][0] == "VWAP"
        assert all_entries[0][2] == "down"


# ---------------------------------------------------------------------------
# 7. AlertManager advance_state doesn't trigger notifications
# ---------------------------------------------------------------------------


class TestAdvanceState:
    def test_advance_state_no_notifications(self):
        notify_calls = []
        am = AlertManager(
            log_fn=lambda **kw: 1,
            notify_fn=lambda t, b: notify_calls.append((t, b)) or True,
        )
        am.update_levels(ibh=None, ibl=20000.0, vwap=None)
        am.advance_state(20003.0)  # enters zone
        assert am._levels["IBL"].in_zone is True
        assert len(notify_calls) == 0

    def test_advance_state_consumes_entry(self):
        """After advance_state enters the zone, check_and_notify should not re-fire."""
        notify_calls = []
        am = AlertManager(
            log_fn=lambda **kw: 1,
            notify_fn=lambda t, b: notify_calls.append((t, b)) or True,
        )
        am.update_levels(ibh=None, ibl=20000.0, vwap=None)
        am.advance_state(20003.0)  # enters zone silently
        fired, _ = am.check_and_notify(
            current_price=20004.0,
            now_et=datetime.time(15, 30),
            tick_rate=1800,
            session_move_pts=-60,
            consecutive_wins=3,
        )
        assert len(fired) == 0
        assert len(notify_calls) == 0

    def test_advance_state_increments_entry_count(self):
        am = AlertManager(
            log_fn=lambda **kw: 1,
            notify_fn=lambda t, b: True,
        )
        am.update_levels(ibh=None, ibl=20000.0, vwap=None)
        am.advance_state(20003.0)
        assert am._levels["IBL"].entry_count == 1


# ---------------------------------------------------------------------------
# 8. build_message formatting
# ---------------------------------------------------------------------------


class TestBuildMessage:
    def test_buy_scenario_above_level(self):
        title, body = build_message(
            level_name="IBL",
            level_price=20000.0,
            current_price=20005.0,
            entry_count=2,
            now_et=datetime.time(14, 0),
            score=6,
            tier_label="Strong",
            tier_wr="~85%",
        )
        assert "BUY" in title
        assert "20005.00" in title
        assert "IBL @ 20000.00" in body
        assert "Strong" in body
        assert "~85%" in body
        assert "1st retest" in body  # entry_count - 1 = 1
        assert "afternoon" in body

    def test_sell_scenario_below_level(self):
        title, body = build_message(
            level_name="IBL",
            level_price=20000.0,
            current_price=19995.0,
            entry_count=3,
            now_et=datetime.time(12, 0),
            score=5,
            tier_label="Good",
            tier_wr="~85%",
        )
        assert "SELL" in title
        assert "19995.00" in title
        assert "2nd retest" in body
        assert "lunch" in body

    def test_vwap_buy_above(self):
        title, body = build_message(
            level_name="VWAP",
            level_price=20050.0,
            current_price=20055.0,
            entry_count=1,
            score=5,
            tier_label="Good",
            tier_wr="~85%",
        )
        assert "BUY" in title

    def test_vwap_sell_below(self):
        title, body = build_message(
            level_name="VWAP",
            level_price=20050.0,
            current_price=20045.0,
            entry_count=1,
            score=5,
            tier_label="Good",
            tier_wr="~85%",
        )
        assert "SELL" in title

    def test_unknown_level_shows_watch(self):
        title, body = build_message(
            level_name="SOMETHING_NEW",
            level_price=20000.0,
            current_price=20005.0,
            entry_count=1,
            score=5,
            tier_label="Good",
            tier_wr="~85%",
        )
        assert "WATCH" in title

    def test_green_dot_for_buy(self):
        title, _ = build_message(
            level_name="IBL",
            level_price=20000.0,
            current_price=20005.0,
            entry_count=1,
            score=5,
            tier_label="Good",
            tier_wr="~85%",
        )
        assert "\U0001f7e2" in title  # green circle

    def test_red_dot_for_sell(self):
        title, _ = build_message(
            level_name="IBL",
            level_price=20000.0,
            current_price=19995.0,
            entry_count=1,
            score=5,
            tier_label="Good",
            tier_wr="~85%",
        )
        assert "\U0001f534" in title  # red circle

    def test_fib_level_message(self):
        title, body = build_message(
            level_name="FIB_EXT_HI_1.272",
            level_price=20200.0,
            current_price=20205.0,
            entry_count=2,
            score=7,
            tier_label="Elite",
            tier_wr="~88%",
        )
        assert "BUY" in title
        assert "FIB_EXT_HI_1.272" in body


# ---------------------------------------------------------------------------
# 9. _ordinal for various numbers
# ---------------------------------------------------------------------------


class TestOrdinal:
    def test_first(self):
        assert _ordinal(1) == "1st"

    def test_second(self):
        assert _ordinal(2) == "2nd"

    def test_third(self):
        assert _ordinal(3) == "3rd"

    def test_fourth(self):
        assert _ordinal(4) == "4th"

    def test_eleventh(self):
        assert _ordinal(11) == "11th"

    def test_twelfth(self):
        assert _ordinal(12) == "12th"

    def test_thirteenth(self):
        assert _ordinal(13) == "13th"

    def test_twenty_first(self):
        assert _ordinal(21) == "21st"

    def test_twenty_second(self):
        assert _ordinal(22) == "22nd"

    def test_hundredth(self):
        assert _ordinal(100) == "100th"


# ---------------------------------------------------------------------------
# 10. _time_bucket for different times
# ---------------------------------------------------------------------------


class TestTimeBucket:
    def test_none_returns_unknown(self):
        assert _time_bucket(None) == "unknown"

    def test_late_morning(self):
        # Before 11:30 AM
        assert _time_bucket(datetime.time(10, 30)) == "late morning"
        assert _time_bucket(datetime.time(11, 0)) == "late morning"
        assert _time_bucket(datetime.time(11, 29)) == "late morning"

    def test_lunch(self):
        # 11:30 AM to 12:59 PM
        assert _time_bucket(datetime.time(11, 30)) == "lunch"
        assert _time_bucket(datetime.time(12, 30)) == "lunch"
        assert _time_bucket(datetime.time(12, 59)) == "lunch"

    def test_afternoon(self):
        # 1:00 PM to 2:59 PM
        assert _time_bucket(datetime.time(13, 0)) == "afternoon"
        assert _time_bucket(datetime.time(14, 30)) == "afternoon"
        assert _time_bucket(datetime.time(14, 59)) == "afternoon"

    def test_power_hour(self):
        # 3:00 PM onward
        assert _time_bucket(datetime.time(15, 0)) == "power hour"
        assert _time_bucket(datetime.time(15, 30)) == "power hour"
        assert _time_bucket(datetime.time(15, 59)) == "power hour"
