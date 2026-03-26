"""Tests for the pure helper functions in main.py."""

import datetime
import sys
import os
from unittest.mock import MagicMock

import pytest
import pytz

# Allow importing from mnq_alerts package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub out heavy/unavailable third-party modules before importing main.
for mod_name in ("databento",):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from main import is_market_open, ib_period_complete, seconds_until_next_open

ET = pytz.timezone("America/New_York")


def _et(year, month, day, hour, minute, second=0):
    """Create a timezone-aware datetime in ET."""
    naive = datetime.datetime(year, month, day, hour, minute, second)
    return ET.localize(naive)


# ── is_market_open ───────────────────────────────────────────────────────────


class TestIsMarketOpen:
    def test_during_market_hours(self):
        # Wednesday 11:00 AM ET
        assert is_market_open(_et(2026, 3, 25, 11, 0)) is True

    def test_at_market_open(self):
        # Exactly 9:30 AM ET on a weekday
        assert is_market_open(_et(2026, 3, 25, 9, 30)) is True

    def test_just_before_open(self):
        assert is_market_open(_et(2026, 3, 25, 9, 29, 59)) is False

    def test_at_market_close(self):
        # 4:00 PM ET — close is exclusive, so market is NOT open
        assert is_market_open(_et(2026, 3, 25, 16, 0)) is False

    def test_just_before_close(self):
        assert is_market_open(_et(2026, 3, 25, 15, 59, 59)) is True

    def test_after_close(self):
        assert is_market_open(_et(2026, 3, 25, 17, 0)) is False

    def test_before_open(self):
        assert is_market_open(_et(2026, 3, 25, 6, 0)) is False

    def test_saturday(self):
        # 2026-03-28 is a Saturday
        assert is_market_open(_et(2026, 3, 28, 11, 0)) is False

    def test_sunday(self):
        # 2026-03-29 is a Sunday
        assert is_market_open(_et(2026, 3, 29, 11, 0)) is False

    def test_weekend_at_open_time(self):
        # Even at 9:30 AM on a Saturday, market is closed
        assert is_market_open(_et(2026, 3, 28, 9, 30)) is False


# ── ib_period_complete ───────────────────────────────────────────────────────


class TestIbPeriodComplete:
    def test_before_ib_end(self):
        assert ib_period_complete(_et(2026, 3, 25, 10, 0)) is False

    def test_just_before_ib_end(self):
        assert ib_period_complete(_et(2026, 3, 25, 10, 29, 59)) is False

    def test_at_ib_end(self):
        assert ib_period_complete(_et(2026, 3, 25, 10, 30)) is True

    def test_after_ib_end(self):
        assert ib_period_complete(_et(2026, 3, 25, 11, 0)) is True

    def test_at_market_open(self):
        assert ib_period_complete(_et(2026, 3, 25, 9, 30)) is False


# ── seconds_until_next_open ──────────────────────────────────────────────────


class TestSecondsUntilNextOpen:
    def test_before_open_same_day(self):
        # Wednesday 8:30 AM ET -> 1 hour until 9:30 AM
        now = _et(2026, 3, 25, 8, 30)
        assert seconds_until_next_open(now) == 3600.0

    def test_just_before_open(self):
        # Wednesday 9:29 AM ET -> 60 seconds until 9:30
        now = _et(2026, 3, 25, 9, 29)
        assert seconds_until_next_open(now) == 60.0

    def test_after_close_weekday(self):
        # Wednesday 5:00 PM ET -> next day Thursday 9:30 AM = 16.5 hours
        now = _et(2026, 3, 25, 17, 0)
        expected = 16.5 * 3600  # 16 hours 30 minutes
        assert seconds_until_next_open(now) == expected

    def test_during_market_hours(self):
        # Wednesday 11:00 AM ET — already past open, should return time to Thursday 9:30
        now = _et(2026, 3, 25, 11, 0)
        expected = 22.5 * 3600  # 22 hours 30 minutes
        assert seconds_until_next_open(now) == expected

    def test_friday_after_close(self):
        # Friday 5:00 PM ET -> Monday 9:30 AM = 3 days minus 7.5 hours
        # Friday 17:00 -> Monday 09:30 = 64.5 hours
        now = _et(2026, 3, 27, 17, 0)
        expected = 64.5 * 3600
        assert seconds_until_next_open(now) == expected

    def test_saturday(self):
        # Saturday 12:00 PM ET -> Monday 9:30 AM = 1 day 21.5 hours = 45.5 hours
        now = _et(2026, 3, 28, 12, 0)
        expected = 45.5 * 3600
        assert seconds_until_next_open(now) == expected

    def test_sunday(self):
        # Sunday 12:00 PM ET -> Monday 9:30 AM = 21.5 hours
        now = _et(2026, 3, 29, 12, 0)
        expected = 21.5 * 3600
        assert seconds_until_next_open(now) == expected

    def test_midnight_weekday(self):
        # Thursday 00:00 AM ET -> Thursday 9:30 AM = 9.5 hours
        now = _et(2026, 3, 26, 0, 0)
        expected = 9.5 * 3600
        assert seconds_until_next_open(now) == expected
