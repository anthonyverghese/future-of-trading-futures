"""Tests for levels.py — Fibonacci extension level calculations."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from levels import calculate_fib_levels


class TestCalculateFibLevels:
    def test_basic_values(self):
        """Standard IB range produces correct fib extensions."""
        ibh, ibl = 20000.0, 19900.0
        result = calculate_fib_levels(ibh, ibl)
        ib_range = ibh - ibl  # 100
        assert result["FIB_EXT_LO_1.272"] == ibl - 0.272 * ib_range  # 19872.8
        assert result["FIB_EXT_HI_1.272"] == ibh + 0.272 * ib_range  # 20027.2

    def test_equal_ibh_ibl_zero_range(self):
        """When IBH == IBL, extensions equal the level itself."""
        ibh = ibl = 19500.0
        result = calculate_fib_levels(ibh, ibl)
        assert result["FIB_EXT_LO_1.272"] == 19500.0
        assert result["FIB_EXT_HI_1.272"] == 19500.0

    def test_large_range(self):
        """Large IB range (500 points)."""
        ibh, ibl = 20250.0, 19750.0
        ib_range = 500.0
        result = calculate_fib_levels(ibh, ibl)
        assert result["FIB_EXT_LO_1.272"] == ibl - 0.272 * ib_range
        assert result["FIB_EXT_HI_1.272"] == ibh + 0.272 * ib_range

    def test_small_range(self):
        """Small IB range (10 points)."""
        ibh, ibl = 20005.0, 19995.0
        ib_range = 10.0
        result = calculate_fib_levels(ibh, ibl)
        assert result["FIB_EXT_LO_1.272"] == pytest.approx(ibl - 0.272 * ib_range)
        assert result["FIB_EXT_HI_1.272"] == pytest.approx(ibh + 0.272 * ib_range)

    def test_returns_exactly_two_keys(self):
        """Result dict has exactly the two expected keys."""
        result = calculate_fib_levels(20000.0, 19900.0)
        assert set(result.keys()) == {"FIB_EXT_LO_1.272", "FIB_EXT_HI_1.272"}

    def test_lo_below_ibl_hi_above_ibh(self):
        """Extension low is always below IBL, extension high is always above IBH (positive range)."""
        ibh, ibl = 20100.0, 19800.0
        result = calculate_fib_levels(ibh, ibl)
        assert result["FIB_EXT_LO_1.272"] < ibl
        assert result["FIB_EXT_HI_1.272"] > ibh


# Need pytest for approx
import pytest
