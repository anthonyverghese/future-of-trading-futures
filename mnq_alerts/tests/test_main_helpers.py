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
        assert ib_period_complete(_et(2026, 3, 25, 10, 30, 59)) is False

    def test_at_ib_end(self):
        # IB_END is 10:31 (includes the 10:30 bar)
        assert ib_period_complete(_et(2026, 3, 25, 10, 31)) is True

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


# ── Incremental range tracking (30-min window) ─────────────────────────────


class TestIncrementalRange:
    """Tests for the incremental high/low tracking used in main.py's
    30-min volatility window. Tests the logic pattern, not main.py itself."""

    def _compute_range(self, prices_with_times):
        """Simulate the incremental range logic from main.py."""
        from collections import deque
        window = deque()
        high = 0.0
        low = 999999.0
        results = []
        window_size = datetime.timedelta(minutes=30)

        for ts, price in prices_with_times:
            window.append((ts, price))
            if price > high:
                high = price
            if price < low:
                low = price

            cutoff = ts - window_size
            recalc = False
            while window and window[0][0] < cutoff:
                evicted = window.popleft()[1]
                if evicted >= high or evicted <= low:
                    recalc = True

            if recalc:
                if len(window) >= 1:
                    high = max(p for _, p in window)
                    low = min(p for _, p in window)
                else:
                    high = price
                    low = price

            if len(window) >= 2:
                results.append(high - low)
            else:
                results.append(None)

        return results, high, low

    def test_basic_range(self):
        """Two ticks should produce correct range."""
        t0 = _et(2026, 5, 1, 11, 0, 0)
        prices = [
            (t0, 27800.0),
            (t0 + datetime.timedelta(seconds=1), 27810.0),
        ]
        results, high, low = self._compute_range(prices)
        assert results[0] is None  # only 1 tick
        assert results[1] == 10.0  # 27810 - 27800
        assert high == 27810.0
        assert low == 27800.0

    def test_new_high_updates(self):
        """New high should update incrementally."""
        t0 = _et(2026, 5, 1, 11, 0, 0)
        prices = [
            (t0, 27800.0),
            (t0 + datetime.timedelta(seconds=1), 27810.0),
            (t0 + datetime.timedelta(seconds=2), 27820.0),
        ]
        results, high, low = self._compute_range(prices)
        assert results[2] == 20.0  # 27820 - 27800
        assert high == 27820.0

    def test_eviction_triggers_recalc(self):
        """When the high is evicted, recalc should find the new high."""
        t0 = _et(2026, 5, 1, 11, 0, 0)
        prices = [
            (t0, 27850.0),  # this will be the high
            (t0 + datetime.timedelta(minutes=15), 27800.0),  # stays in window
            # Jump 31 min from t0 — evicts only t0
            (t0 + datetime.timedelta(minutes=31), 27810.0),
        ]
        results, high, low = self._compute_range(prices)
        # After eviction of 27850, new high should be 27810
        assert high == 27810.0
        assert low == 27800.0
        assert results[2] == 10.0

    def test_eviction_of_low(self):
        """When the low is evicted, recalc should find the new low."""
        t0 = _et(2026, 5, 1, 11, 0, 0)
        prices = [
            (t0, 27750.0),  # this will be the low
            (t0 + datetime.timedelta(minutes=15), 27800.0),  # stays in window
            (t0 + datetime.timedelta(minutes=31), 27810.0),
        ]
        results, high, low = self._compute_range(prices)
        assert low == 27800.0  # 27750 evicted, new low is 27800
        assert high == 27810.0

    def test_gap_resets_correctly(self):
        """After all ticks evicted (gap), high/low should reset."""
        t0 = _et(2026, 5, 1, 11, 0, 0)
        prices = [
            (t0, 27850.0),
            (t0 + datetime.timedelta(seconds=1), 27800.0),
            # 35 min gap — both old ticks evicted
            (t0 + datetime.timedelta(minutes=35), 27900.0),
            (t0 + datetime.timedelta(minutes=35, seconds=1), 27905.0),
        ]
        results, high, low = self._compute_range(prices)
        # After gap, only 27900 and 27905 in window
        assert high == 27905.0
        assert low == 27900.0
        assert results[3] == 5.0

    def test_no_stale_high_after_gap(self):
        """High from before a gap must not persist after eviction."""
        t0 = _et(2026, 5, 1, 11, 0, 0)
        prices = [
            (t0, 27900.0),  # high that will be evicted
            (t0 + datetime.timedelta(seconds=1), 27850.0),
            # Gap — both evicted
            (t0 + datetime.timedelta(minutes=35), 27800.0),
            (t0 + datetime.timedelta(minutes=35, seconds=1), 27810.0),
        ]
        results, high, low = self._compute_range(prices)
        # 27900 must NOT be the high anymore
        assert high == 27810.0
        assert low == 27800.0
        assert results[3] == 10.0  # not 100

    def test_single_tick_after_eviction(self):
        """Single tick after full eviction should give range=None."""
        t0 = _et(2026, 5, 1, 11, 0, 0)
        prices = [
            (t0, 27800.0),
            (t0 + datetime.timedelta(seconds=1), 27810.0),
            (t0 + datetime.timedelta(minutes=35), 27820.0),  # evicts both
        ]
        results, high, low = self._compute_range(prices)
        # Only 1 tick in window after eviction
        assert results[2] is None

    def test_matches_naive_implementation(self):
        """Incremental tracking should match naive max/min scan."""
        import random
        random.seed(42)
        t0 = _et(2026, 5, 1, 11, 0, 0)
        # Generate 100 ticks over 40 minutes
        prices = []
        p = 27800.0
        for i in range(100):
            ts = t0 + datetime.timedelta(seconds=i * 24)  # ~24s apart
            p += random.uniform(-5, 5)
            prices.append((ts, round(p, 2)))

        results, _, _ = self._compute_range(prices)

        # Naive: for each tick, scan the window
        window_size = datetime.timedelta(minutes=30)
        for i, (ts, price) in enumerate(prices):
            window = [(t, p) for t, p in prices[:i+1]
                      if t >= ts - window_size]
            if len(window) >= 2:
                naive_range = max(p for _, p in window) - min(p for _, p in window)
                assert abs(results[i] - naive_range) < 0.01, \
                    f"Tick {i}: incremental={results[i]}, naive={naive_range}"
